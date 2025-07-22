"""
Upload-side data cleaners & inserters.
Assumes data frames with columns that match the DB schema exactly
(you can add mapping logic later if needed).
"""

from __future__ import annotations
import pandas as pd
from db_handler import execute, run_transaction


# ───────────────────────── Core upsert helpers ────────────────────────── #
@run_transaction
def upsert_dataframe(conn, df: pd.DataFrame, table: str) -> None:
    """
    Bulk-insert a dataframe into *table*.

    Parameters
    ----------
    conn  : SQLAlchemy connection (provided by run_transaction)
    df    : DataFrame - columns must match table columns
    table : 'inventory' | 'purchases' | 'sales'
    """
    if df.empty:
        return

    # Build VALUES (...)
    placeholders = ", ".join([f":{col}" for col in df.columns])
    cols = ", ".join(df.columns)
    sql = f"INSERT INTO {table} ({cols}) VALUES ({placeholders});"

    # Convert any NaNs to None for SQL
    data = df.where(pd.notnull(df), None).to_dict(orient="records")
    conn.execute(sql, data)  # many-values insert


# ───────────────────────── Simple column sanity checks ────────────────── #
REQUIRED_COLUMNS = {
    "inventory": {"item_name", "item_barcode", "category", "initial_stock", "current_stock", "unit"},
    "purchases": {"item_id", "quantity", "purchase_price"},
    "sales": {"item_id", "quantity", "sale_price"},
}

def check_columns(df: pd.DataFrame, table: str) -> None:
    missing = REQUIRED_COLUMNS[table] - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
