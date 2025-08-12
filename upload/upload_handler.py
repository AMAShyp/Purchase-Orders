"""
upload_handler.py – v6
• Validates columns
• Normalises date columns
• Inventory upload: prevents duplicate item_name / item_barcode
• Maps item_name / item_barcode → item_id for purchases & sales
• Auto-adds new items (on Purchase Invoice only)
• Updates current_stock using bill_type:
    - Purchases:  Purchase Invoice  (+), Purchase Return Invoice (−)
    - Sales:      Sales Invoice     (−), Sales Return Invoice     (+)
"""

from __future__ import annotations
import pandas as pd
from typing import Literal
from sqlalchemy import text
from db_handler import fetch_dataframe, run_transaction

# ───────────────────────── Column rules ──────────────────────────────── #
BASE_REQ = {
    "inventory": {"item_name", "item_barcode", "category", "unit",
                  "initial_stock", "current_stock"},
    "purchases": {"bill_type", "quantity", "purchase_price"},
    "sales":     {"bill_type", "quantity", "sale_price"},
}

DATE_COLS = {
    "purchases": ["purchase_date"],
    "sales":     ["sale_date"],
}

def check_columns(df: pd.DataFrame,
                  table: Literal["inventory", "purchases", "sales"]) -> None:
    required = BASE_REQ[table].copy()
    if table in ("purchases", "sales"):
        if not ({"item_name", "item_barcode", "item_id"} & set(df.columns)):
            raise ValueError("Need item_name, item_barcode or item_id column.")
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")

# ───────────────────────── Date normaliser ──────────────────────────── #
def _normalise_dates(df: pd.DataFrame, table: str) -> pd.DataFrame:
    for col in DATE_COLS.get(table, []):
        if col not in df.columns:  # optional
            continue
        raw = df[col].copy()
        df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True).dt.date
        bad = raw.notna() & df[col].isna()
        if bad.any():
            rows = df.index[bad].tolist()
            raise ValueError(
                f"Invalid {col} in rows {rows}. Use YYYY-MM-DD or recognised date."
            )
    return df

# ───────────────────────── Inventory validators ─────────────────────── #
def _clean_and_validate_inventory_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise text, ensure no duplicates in-file, and no collisions with DB.
    - item_name: checked case-insensitively (after trim)
    - item_barcode: checked as exact (after trim)
    """
    df = df.copy()

    # Normalise strings
    for col in ("item_name", "item_barcode", "category", "unit"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Coerce stocks to non-negative integers
    for col in ("initial_stock", "current_stock"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        if (df[col] < 0).any():
            bad_rows = df.index[df[col] < 0].tolist()
            raise ValueError(f"{col} must be ≥ 0 (rows {bad_rows}).")
        df[col] = df[col].astype(int)

    # Empty name / barcode is not allowed
    if (df["item_name"] == "").any() or (df["item_barcode"] == "").any():
        raise ValueError("item_name and item_barcode must not be empty.")

    # Duplicate check inside the file
    name_lower = df["item_name"].str.lower()
    dup_names = df[name_lower.duplicated(keep=False)]["item_name"].unique().tolist()
    dup_barcodes = df[df["item_barcode"].duplicated(keep=False)]["item_barcode"].unique().tolist()

    msgs = []
    if dup_names:
        msgs.append("Duplicate item_name(s) in upload: " + ", ".join(sorted(dup_names)[:10]))
    if dup_barcodes:
        msgs.append("Duplicate item_barcode(s) in upload: " + ", ".join(sorted(dup_barcodes)[:10]))
    if msgs:
        raise ValueError(" | ".join(msgs))

    # Collision check vs DB
    existing = fetch_dataframe("SELECT item_name, item_barcode FROM inventory")
    existing["item_name_norm"] = existing["item_name"].astype(str).str.strip().str.lower()
    existing["item_barcode_norm"] = existing["item_barcode"].astype(str).str.strip()

    names_in_db = set(existing["item_name_norm"])
    codes_in_db = set(existing["item_barcode_norm"])

    colliding_names = sorted(set(name_lower) & names_in_db)
    colliding_codes = sorted(set(df["item_barcode"]) & codes_in_db)

    msgs = []
    if colliding_names:
        # Show original names for user friendliness
        original = df[df["item_name"].str.lower().isin(colliding_names)]["item_name"].unique().tolist()
        msgs.append("Already exists (item_name): " + ", ".join(sorted(original)[:10]))
    if colliding_codes:
        msgs.append("Already exists (item_barcode): " + ", ".join(sorted(colliding_codes)[:10]))
    if msgs:
        raise ValueError(" | ".join(msgs))

    return df

# ───────────────────────── Sign from bill_type ──────────────────────── #
def _row_sign(table: str, bill_type: str) -> int:
    """
    Returns +1 or -1 for stock delta based on bill_type and table.
    Purchases: base +1; 'return' flips to −1.
    Sales:     base −1; 'return' flips to +1.
    """
    bt = (bill_type or "").strip().lower()
    is_return = "return" in bt
    base = 1 if table == "purchases" else -1
    return -base if is_return else base

# ───────────────────────── Main upsert function ─────────────────────── #
@run_transaction
def upsert_dataframe(conn,
                     df: pd.DataFrame,
                     table: Literal["inventory", "purchases", "sales"]) -> None:
    df = df.dropna(how="all")
    if df.empty:
        raise ValueError("No data rows found.")

    df = _normalise_dates(df, table)

    if table in ("purchases", "sales"):
        df = _handle_purchases_or_sales(conn, df, table)
    else:
        # STRICT inventory insert: prevent duplicates before writing
        df = _clean_and_validate_inventory_df(df)
        _insert_dataframe(conn, df, "inventory")

# ───────────────────────── Purchases / Sales logic ──────────────────── #
def _handle_purchases_or_sales(conn, df: pd.DataFrame, table: str) -> pd.DataFrame:
    inv = fetch_dataframe("SELECT item_id, item_name, item_barcode FROM inventory")
    name2id = inv.set_index("item_name")["item_id"].to_dict()
    code2id = inv.set_index("item_barcode")["item_id"].to_dict()

    inserted_qty: dict[int, int] = {}

    # Resolve item_id and compute per-row sign
    for idx, row in df.iterrows():
        sign = _row_sign(table, row.get("bill_type"))
        df.at[idx, "__sign"] = sign

        item_id = row.get("item_id")
        if not pd.isna(item_id):
            continue

        item_id = name2id.get(row.get("item_name")) \
              or code2id.get(row.get("item_barcode"))

        # New item handling
        if item_id is None:
            if table == "purchases" and sign > 0:
                qty = int(row["quantity"])
                ins = text(
                    "INSERT INTO inventory "
                    "(item_name, item_barcode, category, initial_stock, current_stock, unit) "
                    "VALUES (:name, :code, 'food', :qty, :qty, 'Psc') "
                    "RETURNING item_id;"
                )
                item_id = conn.execute(
                    ins,
                    {"name": row.get("item_name"),
                     "code": row.get("item_barcode"),
                     "qty": qty}
                ).scalar_one()
                inserted_qty[item_id] = qty
                name2id[row.get("item_name")] = item_id
                code2id[row.get("item_barcode")] = item_id
            else:
                # Unknown item on Sales OR on Purchase Return
                raise ValueError(
                    f"Row {idx}: unknown item for {table} with bill_type='{row.get('bill_type')}'."
                )

        df.at[idx, "item_id"] = item_id

    # Compute deltas before dropping helper columns
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0).astype(int)
    df["_delta"] = df["quantity"] * df["__sign"]

    # Insert rows (drop helper columns)
    drop_cols = [c for c in ("item_name", "item_barcode", "__sign", "_delta") if c in df.columns]
    _insert_dataframe(conn, df.drop(columns=drop_cols), table)

    # Aggregate and adjust stock
    deltas = df.groupby("item_id")["_delta"].sum()
    for item_id, delta in deltas.items():
        # Avoid double-adding the first purchase qty for brand-new items
        if table == "purchases" and item_id in inserted_qty:
            delta -= inserted_qty[item_id]
        if delta == 0:
            continue
        conn.execute(
            text("UPDATE inventory SET current_stock = current_stock + :d WHERE item_id = :i;"),
            {"d": int(delta), "i": int(item_id)},
        )
    return df

# ───────────────────────── Generic INSERT helper ────────────────────── #
def _insert_dataframe(conn, df: pd.DataFrame, table: str) -> None:
    cols = list(df.columns)
    placeholders = ", ".join([f":{c}" for c in cols])
    sql = text(f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders});")
    payload = df.where(pd.notnull(df), None).to_dict(orient="records")
    conn.execute(sql, payload)
