"""
Upload-side data cleaners & inserters  (v3 – date normalisation).
"""

from __future__ import annotations
import pandas as pd
from typing import Literal
from sqlalchemy import text
from db_handler import fetch_dataframe, run_transaction

# ───────────────────────── Validation rules ─────────────────────────── #
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

def check_columns(df: pd.DataFrame,
                  table: Literal["inventory", "purchases", "sales"]) -> None:
    required = BASE_REQ[table].copy()

    # identifier column requirement for purchases/sales
    if table in ("purchases", "sales"):
        if not ({"item_name", "item_barcode", "item_id"} & set(df.columns)):
            raise ValueError("File must have item_name, item_barcode or item_id column.")

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")

# ───────────────────────── Helpers ──────────────────────────────────── #
def _inject_item_id(df: pd.DataFrame) -> pd.DataFrame:
    if "item_id" in df.columns:
        return df

    inv = fetch_dataframe(
        "SELECT item_id, item_name, item_barcode FROM inventory"
    )
    name2id = inv.set_index("item_name")["item_id"].to_dict()
    code2id = inv.set_index("item_barcode")["item_id"].to_dict()

    def _resolve(row):
        return name2id.get(row.get("item_name")) \
            or code2id.get(row.get("item_barcode"))

    df["item_id"] = df.apply(_resolve, axis=1)
    if df["item_id"].isna().any():
        bad = df[df["item_id"].isna()].index.tolist()
        raise ValueError(f"Unmatched items in rows: {bad} – check names / barcodes.")
    return df


def _normalise_dates(df: pd.DataFrame, table: str) -> pd.DataFrame:
    """Convert date strings/Excel serials to ISO `YYYY-MM-DD`."""
    for col in DATE_COLS.get(table, []):
        if col not in df.columns:
            continue

        original = df[col].copy()
        df[col] = (
            pd.to_datetime(df[col], errors="coerce", dayfirst=True)
              .dt.date                      # pure python date
        )

        # rows where user typed something but parse failed → error
        bad_mask = original.notna() & df[col].isna()
        if bad_mask.any():
            bad_rows = df.index[bad_mask].tolist()
            raise ValueError(
                f"Invalid {col} format in rows: {bad_rows}. "
                "Use YYYY-MM-DD or a recognised date format."
            )
    return df

# ───────────────────────── Bulk upsert ──────────────────────────────── #
@run_transaction
def upsert_dataframe(conn,
                     df: pd.DataFrame,
                     table: Literal["inventory", "purchases", "sales"]) -> None:
    df = df.dropna(how="all")
    if df.empty:
        raise ValueError("No data rows found.")

    # Resolve IDs & drop helper cols
    if table in ("purchases", "sales"):
        df = _inject_item_id(df)
        df = df.drop(columns=[c for c in ("item_name", "item_barcode") if c in df.columns])

    # Date parsing
    df = _normalise_dates(df, table)

    # Build SQL
    cols = list(df.columns)
    placeholders = ", ".join([f":{c}" for c in cols])
    sql = text(f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders});")

    payload = df.where(pd.notnull(df), None).to_dict(orient="records")
    conn.execute(sql, payload)
