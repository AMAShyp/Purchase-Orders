"""
upload_handler.py – v7
• Inventory upload: robust numeric parsing (commas, spaces, Arabic digits), strict validation,
  barcode normalization, defaults for category/unit, no negative stocks.
• Purchases/Sales: same as v6 (bill_type-aware stock updates & new-item creation),
  plus date normalization.
"""

from __future__ import annotations
import re
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

# ───────────────────────── Dates ─────────────────────────────────────── #
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
                f"Invalid {col} in rows {rows}. Use YYYY-MM-DD or recognised date."
            )
    return df

# ───────────────────────── Inventory sanitizer ──────────────────────── #
_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
def _clean_number_like(s: str) -> str:
    """
    Convert common messy number strings to a standard form:
    - translate Arabic/Extended Arabic digits → ASCII
    - drop spaces and common thousands separators (comma, Arabic comma/sep)
    - normalise Unicode minus to ASCII '-'
    - keep digits and at most one dot
    """
    if s is None:
        return ""
    s = str(s).strip().translate(_ARABIC_DIGITS)
    s = s.replace("٬", "").replace("،", "").replace(",", "").replace(" ", "")
    s = s.replace("−", "-").replace("—", "-")
    # remove any characters except digits, dot and minus
    s = re.sub(r"[^0-9\.\-]", "", s)
    # if multiple dots, keep first
    if s.count(".") > 1:
        first, *rest = s.split(".")
        s = first + "." + "".join(rest)
    return s

def _sanitize_inventory(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(how="all")
    if df.empty:
        raise ValueError("No data rows found.")

    # Ensure text columns are strings
    for col in ["item_name", "category", "unit"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Defaults
    if "category" in df.columns:
        df["category"] = df["category"].replace({"", "nan", "None"}, "Food")
    if "unit" in df.columns:
        df["unit"] = df["unit"].replace({"", "nan", "None"}, "pcs")

    # item_name must not be blank
    name_blank = df["item_name"].isna() | (df["item_name"].str.strip() == "")
    if name_blank.any():
        bad_rows = df.index[name_blank].tolist()
        raise ValueError(f"Blank item_name in rows: {bad_rows}.")

    # Normalise barcodes: keep None, else int-like string
    if "item_barcode" in df.columns:
        def norm_bc(v):
            if pd.isna(v) or str(v).strip() == "":
                return None
            num = pd.to_numeric(str(v).translate(_ARABIC_DIGITS).replace(" ", "").replace(",", ""), errors="coerce")
            if pd.isna(num):
                return str(v).strip()
            return str(int(num))
        df["item_barcode"] = df["item_barcode"].apply(norm_bc)

    # Robust numeric parse for stocks
    for col in ["initial_stock", "current_stock"]:
        df[col] = df[col].apply(_clean_number_like)
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Report non-numeric rows with context
    bad_mask = df["initial_stock"].isna() | df["current_stock"].isna()
    if bad_mask.any():
        bad_rows = df.index[bad_mask].tolist()
        sample = df.loc[bad_mask, ["item_name", "initial_stock", "current_stock"]].head(10)
        raise ValueError(
            f"Non-numeric initial/current stock in rows: {bad_rows[:20]}.\n"
            f"Examples:\n{sample.to_string(index=True)}"
        )

    # No negatives allowed for initial snapshot
    neg_mask = (df["initial_stock"] < 0) | (df["current_stock"] < 0)
    if neg_mask.any():
        bad_rows = df.index[neg_mask].tolist()
        raise ValueError(f"Negative stock values in rows: {bad_rows}.")

    # Round to integers
    df["initial_stock"] = df["initial_stock"].round().astype(int)
    df["current_stock"] = df["current_stock"].round().astype(int)

    return df

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
        clean = _sanitize_inventory(df)
        _insert_dataframe(conn, clean, "inventory")

# ───────────────────────── Purchases / Sales (unchanged logic) ──────── #
def _row_sign(table: str, bill_type: str) -> int:
    bt = (bill_type or "").strip().lower()
    is_return = "return" in bt
    base = 1 if table == "purchases" else -1
    return -base if is_return else base

def _handle_purchases_or_sales(conn, df: pd.DataFrame, table: str) -> pd.DataFrame:
    inv = fetch_dataframe("SELECT item_id, item_name, item_barcode FROM inventory")
    name2id = inv.set_index("item_name")["item_id"].to_dict()
    code2id = inv.set_index("item_barcode")["item_id"].to_dict()

    inserted_qty: dict[int, int] = {}

    for idx, row in df.iterrows():
        sign = _row_sign(table, row.get("bill_type"))
        df.at[idx, "__sign"] = sign

        item_id = row.get("item_id")
        if not pd.isna(item_id):
            continue

        item_id = name2id.get(row.get("item_name")) \
              or code2id.get(row.get("item_barcode"))

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
                raise ValueError(
                    f"Row {idx}: unknown item for {table} with bill_type='{row.get('bill_type')}'."
                )

        df.at[idx, "item_id"] = item_id

    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0).astype(int)
    df["_delta"] = df["quantity"] * df["__sign"]

    drop_cols = [c for c in ("item_name", "item_barcode", "__sign", "_delta") if c in df.columns]
    _insert_dataframe(conn, df.drop(columns=drop_cols), table)

    deltas = df.groupby("item_id")["_delta"].sum()
    for item_id, delta in deltas.items():
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
