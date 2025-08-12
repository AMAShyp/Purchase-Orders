"""
upload_handler.py – v6 (performance)
• Validates columns
• Normalises date columns
• Maps item_name / item_barcode → item_id
• Auto-adds new items (on Purchase Invoice only)
• FAST bulk insert via psycopg2.extras.execute_values (batched)
• FAST stock updates via one set-based UPDATE ... FROM (VALUES ...)
• Updates current_stock using bill_type:
    - Purchases:  Purchase Invoice  (+), Purchase Return Invoice (−)
    - Sales:      Sales Invoice     (−), Sales Return Invoice     (+)
"""

from __future__ import annotations
import math
import pandas as pd
from typing import Iterable, List, Literal, Sequence, Tuple
from sqlalchemy import text
from psycopg2.extras import execute_values
from db_handler import fetch_dataframe, run_transaction

# ───────────────────────── Column rules ──────────────────────────────── #
BASE_REQ = {
    "inventory": {"item_name", "item_barcode", "category", "unit",
                  "initial_stock", "current_stock"},
    "purchases": {"bill_type", "quantity", "purchase_price"},
    "sales":     {"bill_type", "sale_price", "quantity"},
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

# ───────────────────────── Bulk helpers (fast) ──────────────────────── #
def _dbapi(conn):
    """Get DBAPI connection from SQLAlchemy Connection in 1.4/2.x."""
    # Works on SQLAlchemy 1.4/2.x
    try:
        return conn.connection.dbapi_connection
    except AttributeError:
        try:
            return conn.connection.connection
        except AttributeError:
            return conn.connection  # last resort

def _iter_rows(df: pd.DataFrame, cols: Sequence[str]) -> Iterable[Tuple]:
    # Ensure None for nulls; tuples for execute_values
    for row in df[cols].itertuples(index=False, name=None):
        yield tuple(None if (isinstance(v, float) and pd.isna(v)) or v is pd.NA else v for v in row)

def _bulk_insert_values(conn, table: str, df: pd.DataFrame, chunk: int = 1000) -> None:
    if df.empty:
        return
    cols: List[str] = list(df.columns)
    dbapi_conn = _dbapi(conn)
    sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES %s"
    it = _iter_rows(df, cols)

    with dbapi_conn.cursor() as cur:
        while True:
            batch = list(next_batch := tuple([x for _, x in zip(range(chunk), it)]))
            if not batch:
                break
            execute_values(cur, sql, batch, page_size=chunk)

def _bulk_update_deltas(conn, deltas: pd.Series) -> None:
    """Apply stock deltas in one statement using VALUES list."""
    values = [(int(k), int(v)) for k, v in deltas.items() if int(v) != 0]
    if not values:
        return
    dbapi_conn = _dbapi(conn)
    sql = (
        "UPDATE inventory AS i "
        "SET current_stock = i.current_stock + v.delta "
        "FROM (VALUES %s) AS v(item_id, delta) "
        "WHERE i.item_id = v.item_id"
    )
    with dbapi_conn.cursor() as cur:
        execute_values(cur, sql, values, template="(%s, %s)", page_size=1000)

# ───────────────────────── Main upsert function ─────────────────────── #
@run_transaction
def upsert_dataframe(conn,
                     df: pd.DataFrame,
                     table: Literal["inventory", "purchases", "sales"]) -> None:
    # Drop fully-empty rows early
    df = df.dropna(how="all")
    if df.empty:
        raise ValueError("No data rows found.")

    # Normalise dates for sales/purchases
    df = _normalise_dates(df, table)

    if table in ("purchases", "sales"):
        df = _handle_purchases_or_sales(conn, df, table)
    else:
        _bulk_insert_values(conn, "inventory", df)

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

    # Compute deltas
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0).astype(int)
    df["_delta"] = df["quantity"] * df["__sign"]

    # Insert rows (drop helper columns) – bulk & batched
    drop_cols = [c for c in ("item_name", "item_barcode", "__sign", "_delta") if c in df.columns]
    _bulk_insert_values(conn, table, df.drop(columns=drop_cols))

    # Aggregate and apply stock deltas as one statement
    deltas = df.groupby("item_id")["_delta"].sum()
    # Avoid double-adding the first purchase qty for brand-new items
    for item_id, qty in inserted_qty.items():
        if item_id in deltas.index:
            deltas.loc[item_id] = int(deltas.loc[item_id]) - int(qty)
    deltas = deltas[deltas != 0]
    _bulk_update_deltas(conn, deltas)
    return df
