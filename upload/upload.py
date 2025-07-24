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
    required = BASE_REQ[table]
    if table in ("purchases", "sales") and not (
        {"item_name", "item_barcode", "item_id"} & set(df.columns)
    ):
        raise ValueError("File must have item_name, item_barcode or item_id column.")
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")

# ───────────────────────── Mapping helpers ──────────────────────────── #
def _inject_item_id(df: pd.DataFrame) -> pd.DataFrame:
    if "item_id" in df.columns:
        return df
    inv = fetch_dataframe(
        "SELECT item_id, item_name, item_barcode FROM inventory"
    )
    name2id = inv.set_index("item_name")["item_id"].to_dict()
    code2id = inv.set_index("item_barcode")["item_id"].to_dict()

    def resolver(row):
        return name2id.get(row.get("item_name")) \
            or code2id.get(row.get("item_barcode"))

    df["item_id"] = df.apply(resolver, axis=1)
    if df["item_id"].isna().any():
        bad_rows = df[df["item_id"].isna()].index.tolist()
        raise ValueError(f"Unmatched items in rows: {bad_rows} – check names / barcodes.")
    return df

# ───────────────────────── Bulk-insert wrapper ──────────────────────── #
@run_transaction
def upsert_dataframe(conn,
                     df: pd.DataFrame,
                     table: Literal["inventory", "purchases", "sales"]) -> None:
    # 1. Drop rows that are entirely blank (Excel often pads them)
    df = df.dropna(how="all")
    if df.empty:
        raise ValueError("Upload is empty after removing blank rows.")

    # 2. Resolve item_id for purchases / sales
    if table in ("purchases", "sales"):
        df = _inject_item_id(df)
        # After mapping we don’t need the human identifiers anymore
        df = df.drop(columns=[c for c in ("item_name", "item_barcode") if c in df.columns])

    # 3. Build parameterised INSERT
    placeholders = ", ".join([f":{c}" for c in df.columns])
    cols = ", ".join(df.columns)
    sql = text(f"INSERT INTO {table} ({cols}) VALUES ({placeholders});")

    # 4. Convert NaN → None, then execute many
    data = df.where(pd.notnull(df), None).to_dict(orient="records")
    conn.execute(sql, data)
