# sales_upload_handler.py – FAST v1.0 (Purchases/Sales)
# High-performance bulk load for purchases/sales with:
# - subset headers (only provided columns are used; all must exist in table)
# - resolve item_id from item_name/item_barcode (param batch)
# - create new items on Purchase Invoice (+) only (ON CONFLICT safe)
# - COPY into purchases/sales; then atomic stock update
# - numeric coercion for quantity/price fields

from __future__ import annotations

import io
import re
import time
from typing import Dict, List, Optional, Set

import pandas as pd
from sqlalchemy import text

# robust imports (package vs flat)
try:
    from ..db_handler import fetch_dataframe, run_transaction
except Exception:
    from db_handler import fetch_dataframe, run_transaction


# ───────────────────── constants / helpers ───────────────────── #

BILL_SIGNS = {
    "purchases": {
        "purchase invoice": +1,
        "purchase return invoice": -1,
    },
    "sales": {
        "sales invoice": -1,
        "sales return invoice": +1,
    },
}

_num_clean_re = re.compile(r"[,\s]")

def _num_clean_to_int(series: pd.Series) -> pd.Series:
    if series.dtype.kind in "biufc":
        return pd.to_numeric(series, errors="coerce").fillna(0).round(0).astype("Int64")
    s2 = series.astype(str).map(lambda x: _num_clean_re.sub("", x))
    return pd.to_numeric(s2, errors="coerce").fillna(0).round(0).astype("Int64")

def _num_clean_to_float(series: pd.Series) -> pd.Series:
    if series.dtype.kind in "biufc":
        return pd.to_numeric(series, errors="coerce")
    s2 = series.astype(str).map(lambda x: _num_clean_re.sub("", x))
    return pd.to_numeric(s2, errors="coerce")


def _get_table_columns_ordered(conn, table: str, schema: str = "public") -> List[str]:
    q = text("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = :schema AND table_name = :table
        ORDER BY ordinal_position
    """)
    return [r[0] for r in conn.execute(q, {"schema": schema, "table": table}).fetchall()]

def _get_raw_psycopg2_connection(sqlalchemy_conn) -> Optional[object]:
    raw = getattr(sqlalchemy_conn, "connection", None)
    if raw is None:
        return None
    psyco = getattr(raw, "connection", raw)
    return psyco if hasattr(psyco, "cursor") else None

def _to_csv_buffer(df: pd.DataFrame) -> io.StringIO:
    buf = io.StringIO()
    df.to_csv(buf, index=False, header=False, na_rep='\\N')
    buf.seek(0)
    return buf


# ───────────────────── core pipeline ───────────────────── #

def _strict_sign(table: str, bill_type: str) -> int:
    bt = (bill_type or "").strip().lower()
    try:
        return BILL_SIGNS[table][bt]
    except KeyError:
        raise ValueError(f"Unknown bill_type '{bill_type}' for {table}. "
                         f"Expected: {list(BILL_SIGNS[table].keys())}")

def _align_subset_columns(df: pd.DataFrame, db_cols: List[str]) -> List[str]:
    db_lower = {c.lower(): c for c in db_cols}
    extra = [c for c in df.columns if c.lower() not in db_lower]
    if extra:
        raise ValueError(f"Unknown columns in file: {extra}")
    used = [c for c in db_cols if c.lower() in {x.lower() for x in df.columns}]
    return used

def _resolve_item_ids(conn, df: pd.DataFrame, table: str) -> Dict[int, int]:
    """
    Ensures df has item_id (creating items only for purchases with positive sign).
    Returns map of {new_item_id: initial_qty_inserted} for first positive purchases.
    """
    inserted_qty: Dict[int, int] = {}

    # Build sets
    have_id = "item_id" in df.columns
    need_names = set()
    need_codes = set()
    if "item_name" in df.columns:
        need_names = set(df["item_name"].dropna().astype(str).str.strip())
    if "item_barcode" in df.columns:
        need_codes = set(df["item_barcode"].dropna().astype(str).str.strip())

    # Fetch mapping just for needed values
    inv = pd.DataFrame()
    if need_names or need_codes:
        try:
            inv = fetch_dataframe(
                "SELECT item_id, item_name, item_barcode FROM inventory "
                "WHERE item_name = ANY(%(names)s) OR item_barcode = ANY(%(codes)s);",
                params={"names": list(need_names) or ["__none__"], "codes": list(need_codes) or ["__none__"]},
            )
        except Exception:
            inv = fetch_dataframe("SELECT item_id, item_name, item_barcode FROM inventory;")
    name2id = inv.set_index("item_name")["item_id"].to_dict() if not inv.empty else {}
    code2id = inv.set_index("item_barcode")["item_id"].to_dict() if not inv.empty else {}

    # Resolve per-row
    out_ids = []
    for idx, row in df.iterrows():
        iid = row.get("item_id") if have_id else None
        if pd.notna(iid):
            out_ids.append(int(iid))
            continue

        nm = (row.get("item_name") or "").strip()
        bc = (row.get("item_barcode") or "").strip()
        cand = None
        if nm and nm in name2id:
            cand = name2id[nm]
        if bc and bc in code2id:
            if cand is not None and code2id[bc] != cand:
                raise ValueError(f"Row {idx}: item_name and item_barcode map to different items.")
            cand = code2id[bc]

        if cand is None:
            # Create only for Purchase Invoice (positive)
            sgn = _strict_sign(table, row.get("bill_type", ""))
            if table == "purchases" and sgn > 0:
                qty = int(_num_clean_to_int(pd.Series([row.get("quantity")])).iloc[0])
                ins = text("""
                    INSERT INTO inventory
                    (item_name, item_barcode, category, unit, initial_stock, current_stock)
                    VALUES (:name, :code, COALESCE(:cat,'Uncategorized'), COALESCE(:unit,'Pcs'), :qty, :qty)
                    ON CONFLICT (item_name, item_barcode)
                    DO UPDATE SET updated_at = CURRENT_TIMESTAMP
                    RETURNING item_id;
                """)
                new_id = conn.execute(ins, {
                    "name": nm or None,
                    "code": bc or None,
                    "cat":  row.get("category"),
                    "unit": row.get("unit"),
                    "qty":  qty,
                }).scalar_one()
                inserted_qty[int(new_id)] = qty
                if nm: name2id[nm] = int(new_id)
                if bc: code2id[bc] = int(new_id)
                cand = int(new_id)
            else:
                raise ValueError(f"Row {idx}: unknown item and not creatable for this bill_type/table.")

        out_ids.append(int(cand))

    df["item_id"] = pd.Series(out_ids, index=df.index).astype("Int64")
    return inserted_qty


def _prepare_target_df(conn, df: pd.DataFrame, table: str, schema: str) -> pd.DataFrame:
    """Subset headers → DB casing/order; coerce quantity/price/date."""
    db_cols = _get_table_columns_ordered(conn, table, schema)
    used_cols = _align_subset_columns(df, db_cols)
    lower2db = {c.lower(): c for c in db_cols}
    df2 = df[[c for c in df.columns if c.lower() in {u.lower() for u in used_cols}]].copy()
    df2.rename(columns={c: lower2db[c.lower()] for c in df2.columns}, inplace=True)
    df2 = df2[used_cols]

    # Coercions
    if "quantity" in df2.columns:
        df2["quantity"] = _num_clean_to_int(df2["quantity"])
    price_col = "purchase_price" if table == "purchases" else "sale_price"
    if price_col in df2.columns:
        df2[price_col] = _num_clean_to_float(df2[price_col])

    # Bill sign (for later delta compute)
    if "bill_type" in df2.columns:
        df2["__sign"] = df2["bill_type"].map(lambda v: _strict_sign(table, v))
    else:
        df2["__sign"] = 0

    return df2


def _copy_into_table(conn, df: pd.DataFrame, table: str, schema: str, used_cols: List[str]) -> bool:
    raw = _get_raw_psycopg2_connection(conn)
    if raw is None:
        return False
    # COPY only the visible columns
    csv_buf = _to_csv_buffer(df[used_cols])
    col_list = ", ".join(f'"{c}"' for c in used_cols)
    sql = f'COPY "{schema}"."{table}" ({col_list}) FROM STDIN WITH (FORMAT CSV, NULL \'\\N\')'
    cur = raw.cursor()
    try:
        cur.copy_expert(sql=sql, file=csv_buf)
        return True
    finally:
        cur.close()


def _insert_executemany(conn, df: pd.DataFrame, table: str, schema: str, used_cols: List[str]) -> None:
    placeholders = ", ".join([f":{c}" for c in used_cols])
    ins = text(f'INSERT INTO "{schema}"."{table}" ({", ".join(used_cols)}) VALUES ({placeholders})')
    payload = df[used_cols].where(pd.notnull(df[used_cols]), None).to_dict(orient="records")
    conn.execute(ins, payload)


# ───────────────────── public entrypoint ───────────────────── #

@run_transaction
def bulk_upload_sales_like(conn, *, df: pd.DataFrame, table: str, schema: str = "public") -> Dict[str, object]:
    """
    Fast bulk upload for purchases/sales:
      - resolves item_id (creates new items for Purchase Invoice)
      - COPY into purchases/sales
      - atomic stock update from aggregated deltas
    """
    if table not in ("purchases", "sales"):
        raise ValueError("table must be 'purchases' or 'sales'")

    timings: Dict[str, float] = {}
    t0 = time.perf_counter()

    # Drop all-empty rows
    df = df.dropna(how="all")
    if df.empty:
        raise ValueError("No data rows found.")

    # Ensure item_id exists, possibly creating new inventory on positive purchase
    t = time.perf_counter()
    created_qty = _resolve_item_ids(conn, df, table)
    timings["resolve_item_ids_ms"] = (time.perf_counter() - t) * 1000

    # Prepare DF to match target table (subset headers + coercions)
    t = time.perf_counter()
    df_t = _prepare_target_df(conn, df, table, schema)
    timings["prepare_target_df_ms"] = (time.perf_counter() - t) * 1000

    # Used column list (in DB order)
    db_cols = _get_table_columns_ordered(conn, table, schema)
    used_cols = [c for c in db_cols if c in df_t.columns]

    # Bulk insert
    t = time.perf_counter()
    used_copy = _copy_into_table(conn, df_t, table, schema, used_cols)
    if not used_copy:
        _insert_executemany(conn, df_t, table, schema, used_cols)
    timings["insert_ms"] = (time.perf_counter() - t) * 1000

    # Aggregate deltas and update inventory
    t = time.perf_counter()
    deltas = (df_t.assign(quantity=_num_clean_to_int(df_t["quantity"]) if "quantity" in df_t else 0)
                .assign(_delta=lambda x: x["quantity"] * x["__sign"])
                .groupby("item_id")["_delta"].sum()
                .reset_index())

    if table == "purchases" and created_qty:
        # subtract initial inserted qty for just-created items
        deltas["_delta"] = deltas.apply(
            lambda r: int(r["_delta"] - created_qty.get(int(r["item_id"]), 0)), axis=1
        )

    deltas = deltas[deltas["_delta"] != 0]
    if not deltas.empty:
        upd = text("""
            WITH changes AS (
                SELECT UNNEST(:ids::int[]) AS item_id, UNNEST(:ds::int[]) AS d
            )
            UPDATE inventory i
            SET current_stock = i.current_stock + c.d,
                updated_at = CURRENT_TIMESTAMP
            FROM changes c
            WHERE i.item_id = c.item_id AND c.d <> 0;
        """)
        conn.execute(upd, {"ids": deltas["item_id"].tolist(), "ds": deltas["_delta"].tolist()})
    timings["stock_update_ms"] = (time.perf_counter() - t) * 1000

    timings["total_ms"] = (time.perf_counter() - t0) * 1000
    return {
        "rows": int(df_t.shape[0]),
        "used_copy": bool(used_copy),
        "used_columns": used_cols,
        "created_items": created_qty,
        "timings": timings,
    }
