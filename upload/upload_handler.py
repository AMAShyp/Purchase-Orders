"""
Upload-side data cleaners & inserters.
• Translates item_name / item_barcode → item_id for purchases & sales.
• Validates the minimal set of columns per table.
"""

from __future__ import annotations
import pandas as pd
from typing import Literal
from db_handler import execute, fetch_dataframe, run_transaction

# ───────────────────────── Validation rules ─────────────────────────── #
BASE_REQ = {
    "inventory": {"item_name", "item_barcode", "category", "initial_stock", "current_stock", "unit"},
    "purchases": {"quantity", "purchase_price"},
    "sales":     {"quantity", "sale_price"},
}

def check_columns(df: pd.DataFrame, table: Literal["inventory", "purchases", "sales"]) -> None:
    required = BASE_REQ[table].copy()

    # For purchases/sales we need *one* identifier column at minimum.
    if table in ("purchases", "sales"):
        if not ({"item_name", "item_barcode", "item_id"} & set(df.columns)):
            raise ValueError("File must have item_name, item_barcode or item_id column.")

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")

# ───────────────────────── Mapping helpers ──────────────────────────── #
def _inject_item_id(df: pd.DataFrame) -> pd.DataFrame:
    """Add item_id column based on item_name / item_barcode."""
    if "item_id" in df.columns:
        return df  # nothing to do

    inventory = fetch_dataframe(
        "SELECT item_id, item_name, item_barcode FROM inventory"
    )
    name_to_id = inventory.set_index("item_name")["item_id"].to_dict()
    barcode_to_id = inventory.set_index("item_barcode")["item_id"].to_dict()

    def resolver(row):
        return (
            name_to_id.get(row.get("item_name"))
            or barcode_to_id.get(row.get("item_barcode"))
        )

    df["item_id"] = df.apply(resolver, axis=1)
    if df["item_id"].isna().any():
        bad_rows = df[df["item_id"].isna()].index.tolist()
        raise ValueError(f"Unmatched items in rows: {bad_rows} – check names / barcodes.")
    return df


# ───────────────────────── Bulk upsert wrapper ──────────────────────── #
@run_transaction
def upsert_dataframe(conn, df: pd.DataFrame, table: Literal["inventory", "purchases", "sales"]) -> None:
    if df.empty:
        return

    # Purchases & sales need item_id resolution
    if table in ("purchases", "sales"):
        df = _inject_item_id(df)
        # We keep date columns if provided; DB default handles null.

    # Drop helper columns that don't exist in the target table
    drop_cols = {"item_name", "item_barcode"} & set(df.columns)
    if drop_cols:
        df = df.drop(columns=list(drop_cols))

    placeholders = ", ".join([f":{c}" for c in df.columns])
    cols = ", ".join(df.columns)
    sql = f"INSERT INTO {table} ({cols}) VALUES ({placeholders});"

    data = df.where(pd.notnull(df), None).to_dict(orient="records")
    conn.execute(sql, data)
