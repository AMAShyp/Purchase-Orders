"""
Upload-side data cleaners & inserters (fixed).
• Keeps item_name & item_barcode for inventory.
• Resolves item_name / item_barcode → item_id for purchases & sales.
• Wraps SQL in sqlalchemy.text() for SQLAlchemy 2.x
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

def check_columns(df: pd.DataFrame,
                  table: Literal["inventory", "purchases", "sales"]) -> None:
    required = BASE_REQ[table].copy()

    # For purchases/sales we need *one* identifier column at minimum
    if table in ("purchases", "sales"):
        if not ({"item_name", "item_barcode", "item_id"} & set(df.columns)):
            raise ValueError("File must have item_name, item_barcode or item_id column.")

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")

# ───────────────────────── Mapping helper ──────────────────────────── #
def _inject_item_id(df: pd.DataFrame) -> pd.DataFrame:
    """Add item_id column using item_name / item_barcode."""
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

# ───────────────────────── Bulk upsert ──────────────────────────────── #
@run_transaction
def upsert_dataframe(conn,
                     df: pd.DataFrame,
                     table: Literal["inventory", "purchases", "sales"]) -> None:
    # Drop empty Excel rows
    df = df.dropna(how="all")
    if df.empty:
        raise ValueError("No data rows found.")

    # Resolve IDs for purchases / sales
    if table in ("purchases", "sales"):
        df = _inject_item_id(df)
        # Drop helper columns only for these tables
        df = df.drop(columns=[c for c in ("item_name", "item_barcode")
                              if c in df.columns])

    # Build executable SQL
    cols = list(df.columns)
    placeholders = ", ".join([f":{c}" for c in cols])
    sql = text(f"INSERT INTO {table} ({', '.join(cols)}) "
               f"VALUES ({placeholders});")

    # Convert NaNs → None
    payload = df.where(pd.notnull(df), None).to_dict(orient="records")
    conn.execute(sql, payload)
