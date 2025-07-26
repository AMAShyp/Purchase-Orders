"""
upload_handler.py  – v4
• Validates columns
• Normalises date columns
• Maps item_name / item_barcode → item_id
• Auto‑adds new items to inventory when needed
• Updates current_stock (+ for purchases, – for sales)
"""

from __future__ import annotations
import pandas as pd
from typing import Literal
from sqlalchemy import text
from db_handler import fetch_dataframe, run_transaction

# ───────────────────────── Column rules ──────────────────────────────── #
BASE_REQ = {
    "inventory": {"item_name", "item_barcode", "category",
                  "initial_stock", "current_stock", "unit"},
    "purchases": {"quantity", "purchase_price"},
    "sales":     {"quantity", "sale_price"},
}

DATE_COLS = {
    "purchases": ["purchase_date"],
    "sales":     ["sale_date"],
}

# ───────────────────────── Validation helpers ───────────────────────── #
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
        if col not in df.columns:
            continue
        raw = df[col].copy()
        df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True).dt.date
        bad = raw.notna() & df[col].isna()
        if bad.any():
            rows = df.index[bad].tolist()
            raise ValueError(
                f"Invalid {col} in rows {rows}. Use YYYY‑MM‑DD or recognised date."
            )
    return df

# ───────────────────────── Main upsert function ─────────────────────── #
@run_transaction
def upsert_dataframe(conn,
                     df: pd.DataFrame,
                     table: Literal["inventory", "purchases", "sales"]) -> None:
    df = df.dropna(how="all")
    if df.empty:
        raise ValueError("No data rows found.")

    # Normalise date formats early
    df = _normalise_dates(df, table)

    if table in ("purchases", "sales"):
        df = _handle_purchases_or_sales(conn, df, table)
    else:  # pure inventory bulk load
        _insert_dataframe(conn, df, "inventory")


# ───────────────────────── Purchases / Sales logic ──────────────────── #
def _handle_purchases_or_sales(conn, df: pd.DataFrame, table: str) -> pd.DataFrame:
    """Resolve IDs, create new items, adjust stock, then insert rows."""
    inv = fetch_dataframe("SELECT item_id, item_name, item_barcode FROM inventory")
    name2id = inv.set_index("item_name")["item_id"].to_dict()
    code2id = inv.set_index("item_barcode")["item_id"].to_dict()

    # 1. Resolve / create item_id for every row ------------------------ #
    new_items_payload = []
    for idx, row in df.iterrows():
        item_id = row.get("item_id")
        if not pd.isna(item_id):
            continue

        item_id = name2id.get(row.get("item_name")) \
              or code2id.get(row.get("item_barcode"))

        # If still None → create new inventory row
        if item_id is None:
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
            # update maps for any subsequent rows referencing same item
            name2id[row.get("item_name")] = item_id
            code2id[row.get("item_barcode")] = item_id

        df.at[idx, "item_id"] = item_id

    # 2. Drop helper columns ------------------------------------------ #
    df = df.drop(columns=[c for c in ("item_name", "item_barcode") if c in df.columns])

    # 3. Insert purchase/sale rows ------------------------------------ #
    _insert_dataframe(conn, df, table)

    # 4. Aggregate quantity per item and adjust stock ----------------- #
    sign = 1 if table == "purchases" else -1
    deltas = df.groupby("item_id")["quantity"].sum() * sign
    for item_id, delta in deltas.items():
        conn.execute(
            text(
                "UPDATE inventory "
                "SET current_stock = current_stock + :delta "
                "WHERE item_id = :id;"
            ),
            {"delta": int(delta), "id": int(item_id)},
        )
    return df

# ───────────────────────── Generic INSERT helper ────────────────────── #
def _insert_dataframe(conn, df: pd.DataFrame, table: str) -> None:
    cols = list(df.columns)
    placeholders = ", ".join([f":{c}" for c in cols])
    sql = text(f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders});")
    payload = df.where(pd.notnull(df), None).to_dict(orient="records")
    conn.execute(sql, payload)
