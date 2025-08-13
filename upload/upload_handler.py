# upload_handler.py – FAST v2.1
# - Generic COPY-based bulk insert (subset headers, numeric coercion)
# - Unified purchases/sales flow:
#     * Accepts Excel with input_quantity / output_quantity
#     * Derives a single `quantity` for DB rows based on bill_type
#     * Resolves item_id from (item_name, item_barcode); creates item on positive purchases
#     * Inserts into purchases & sales; updates inventory.current_stock by bill_type

from __future__ import annotations

import io
import re
import time
from typing import List, Dict, Any, Optional, Set, Tuple

import pandas as pd
from sqlalchemy import text

# Robust imports (package vs flat)
try:
    from ..db_handler import fetch_dataframe, run_transaction
except Exception:
    from db_handler import fetch_dataframe, run_transaction


# ───────────────────────── schema helpers ───────────────────────── #

def _get_table_columns_ordered(conn, table: str, schema: str = "public") -> List[str]:
    q = text("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = :schema AND table_name = :table
        ORDER BY ordinal_position
    """)
    rows = conn.execute(q, {"schema": schema, "table": table}).fetchall()
    return [r[0] for r in rows]

def _get_integer_columns(conn, table: str, schema: str = "public") -> Set[str]:
    q = text("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = :schema AND table_name = :table
          AND data_type IN ('integer','smallint','bigint')
    """)
    return {r[0] for r in conn.execute(q, {"schema": schema, "table": table}).fetchall()}

def _get_numeric_columns(conn, table: str, schema: str = "public") -> Set[str]:
    q = text("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = :schema AND table_name = :table
          AND data_type IN ('numeric','real','double precision')
    """)
    return {r[0] for r in conn.execute(q, {"schema": schema, "table": table}).fetchall()}

def get_insertable_columns(table: str, schema: str = "public") -> List[str]:
    q = f"""
        SELECT column_name, is_identity, column_default
        FROM information_schema.columns
        WHERE table_schema = '{schema}' AND table_name = '{table}'
        ORDER BY ordinal_position
    """
    df = fetch_dataframe(q)
    cols: List[str] = []
    for _, r in df.iterrows():
        name = r["column_name"]
        is_identity = (str(r["is_identity"]).upper() == "YES")
        if is_identity:
            continue
        if name.lower() in ("created_at", "updated_at"):
            continue
        cols.append(name)
    return cols


# ───────────────────────── DF alignment & coercion ───────────────────────── #

_num_clean_re = re.compile(r"[,\s]")

def _align_subset_columns(df: pd.DataFrame, db_cols: List[str]) -> List[str]:
    db_lower_map = {c.lower(): c for c in db_cols}
    file_lowers = [c.lower() for c in df.columns]
    extra = [c for c in df.columns if c.lower() not in db_lower_map]
    if extra:
        raise ValueError(f"Unknown columns in file: {extra}")
    return [c for c in db_cols if c.lower() in file_lowers]

def _coerce_numeric_like(df: pd.DataFrame, cols: List[str], as_int: bool) -> None:
    for c in cols:
        if c not in df.columns:
            continue
        s = df[c]
        if s.dtype.kind in "biufc":
            if as_int:
                df[c] = pd.to_numeric(s, errors="coerce").fillna(0).round(0).astype("Int64")
            else:
                df[c] = pd.to_numeric(s, errors="coerce")
            continue
        s2 = s.astype(str).map(lambda x: _num_clean_re.sub("", x))
        s_num = pd.to_numeric(s2, errors="coerce")
        if as_int:
            df[c] = s_num.fillna(0).round(0).astype("Int64")
        else:
            df[c] = s_num


# ───────────────────────── COPY plumbing ───────────────────────── #

def _get_raw_psycopg2_connection(sqlalchemy_conn) -> Optional[Any]:
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


# ───────────────────────── core COPY insert (reusable) ───────────────────────── #

def _bulk_insert_core(conn, *, df: pd.DataFrame, table: str, schema: str = "public") -> Dict[str, Any]:
    timings: Dict[str, float] = {}
    t0 = time.perf_counter()

    # Columns / types
    t = time.perf_counter()
    db_cols = _get_table_columns_ordered(conn, table, schema)
    int_cols = _get_integer_columns(conn, table, schema)
    num_cols = _get_numeric_columns(conn, table, schema)
    timings["fetch_columns_ms"] = (time.perf_counter() - t) * 1000

    # Align subset
    t = time.perf_counter()
    df = df.dropna(how="all")
    if df.empty:
        raise ValueError("No data rows found.")
    used_cols = _align_subset_columns(df, db_cols)

    lower2db = {c.lower(): c for c in db_cols}
    present_lower = {c.lower() for c in used_cols}
    df2 = df[[c for c in df.columns if c.lower() in present_lower]].copy()
    df2.rename(columns={c: lower2db[c.lower()] for c in df2.columns}, inplace=True)
    df2 = df2[used_cols]
    timings["align_columns_ms"] = (time.perf_counter() - t) * 1000

    # Coerce numerics
    t = time.perf_counter()
    present_int = [c for c in used_cols if c in int_cols]
    present_num = [c for c in used_cols if c in num_cols]
    if present_int:
        _coerce_numeric_like(df2, present_int, as_int=True)
    if present_num:
        _coerce_numeric_like(df2, present_num, as_int=False)
    timings["numeric_coercion_ms"] = (time.perf_counter() - t) * 1000

    # COPY fast path
    rows = len(df2)
    used_copy = False
    t = time.perf_counter()
    raw = _get_raw_psycopg2_connection(conn)
    if raw is not None:
        csv_buf = _to_csv_buffer(df2)
        col_list = ", ".join(f'"{c}"' for c in used_cols)
        sql = f'COPY "{schema}"."{table}" ({col_list}) FROM STDIN WITH (FORMAT CSV, NULL \'\\N\')'
        cur = raw.cursor()
        try:
            cur.copy_expert(sql=sql, file=csv_buf)
            used_copy = True
        finally:
            cur.close()
    timings["copy_or_insert_ms"] = (time.perf_counter() - t) * 1000

    # Fallback executemany
    if not used_copy:
        t = time.perf_counter()
        placeholders = ", ".join([f":{c}" for c in used_cols])
        ins = text(f'INSERT INTO "{schema}"."{table}" ({", ".join(used_cols)}) VALUES ({placeholders})')
        payload = df2.where(pd.notnull(df2), None).to_dict(orient="records")
        conn.execute(ins, payload)
        timings["executemany_ms"] = (time.perf_counter() - t) * 1000

    timings["total_ms"] = (time.perf_counter() - t0) * 1000
    return {"rows": rows, "used_copy": used_copy, "timings": timings, "used_columns": used_cols}


# Public generic entry
@run_transaction
def bulk_insert_exact_headers(conn, *, df: pd.DataFrame, table: str, schema: str = "public") -> Dict[str, Any]:
    return _bulk_insert_core(conn, df=df, table=table, schema=schema)


def get_row_count(table: str, schema: str = "public") -> int:
    q = f'SELECT COUNT(*) FROM "{schema}"."{table}"'
    df = fetch_dataframe(q)
    return int(df.iloc[0, 0])


# ───────────────────────── unified Purchases/Sales flow ───────────────────────── #

SALE_TYPES = {
    "sales invoice",
    "sales return invoice",
}
PURCHASE_TYPES = {
    "purchase invoice direct",
    "purchasing return invoice",
    # aliases
    "purchase invoice",
    "purchase return invoice",
}

# Which bill types use input vs output quantity in the Excel template
INPUT_QTY_TYPES = {
    "purchase invoice direct", "purchase invoice", "sales return invoice"
}
OUTPUT_QTY_TYPES = {
    "sales invoice", "purchasing return invoice", "purchase return invoice"
}

def _normalize_bill_type(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip().str.casefold()

def _bill_sign(bt: str) -> int:
    bt = (bt or "").strip().lower()
    if bt in ("sales invoice",):
        return -1
    if bt in ("sales return invoice",):
        return +1
    if bt in ("purchase invoice direct", "purchase invoice"):
        return +1
    if bt in ("purchasing return invoice", "purchase return invoice"):
        return -1
    return 0

def _coerce_qty_series(s: pd.Series) -> pd.Series:
    """Return integer series: clean commas/spaces, to_numeric, round, fillna 0."""
    if s is None:
        return pd.Series([], dtype="Int64")
    if s.dtype.kind in "biufc":
        return pd.to_numeric(s, errors="coerce").fillna(0).round(0).astype("Int64")
    s2 = s.astype(str).map(lambda x: _num_clean_re.sub("", x))
    return pd.to_numeric(s2, errors="coerce").fillna(0).round(0).astype("Int64")

def _derive_quantity_from_io(df: pd.DataFrame) -> pd.DataFrame:
    """
    Using `input_quantity` and `output_quantity`, compute the DB `quantity` for each row,
    based on `bill_type` rules. Leaves original columns in place; caller can drop them.
    """
    df2 = df.copy()
    bt = _normalize_bill_type(df2.get("bill_type", pd.Series([], dtype="object")))
    in_q = _coerce_qty_series(df2.get("input_quantity", pd.Series([], dtype="object")))
    out_q = _coerce_qty_series(df2.get("output_quantity", pd.Series([], dtype="object")))

    is_input = bt.isin(INPUT_QTY_TYPES)
    # quantity = input_quantity where is_input, else output_quantity
    df2["quantity"] = in_q.where(is_input, out_q)
    df2["bill_type"] = bt
    return df2

def _resolve_item_ids_and_create(
    conn,
    df: pd.DataFrame,
    allow_create_positive_purchases: bool = True,
) -> Tuple[pd.DataFrame, Dict[int, int]]:
    df2 = df.copy()

    names = set(df2.get("item_name", pd.Series([], dtype="object")).dropna().astype(str).str.strip())
    barcodes = set(df2.get("item_barcode", pd.Series([], dtype="object")).dropna().astype(str).str.strip())
    if not names and not barcodes:
        raise ValueError("Need item_name or item_barcode to resolve items.")

    try:
        inv = fetch_dataframe(
            """
            SELECT item_id, item_name, item_barcode
            FROM inventory
            WHERE item_name = ANY(%(names)s) OR item_barcode = ANY(%(codes)s);
            """,
            params={"names": list(names) or ["__none__"], "codes": list(barcodes) or ["__none__"]},
        )
    except Exception:
        inv = fetch_dataframe("SELECT item_id, item_name, item_barcode FROM inventory;")

    name2id = inv.set_index("item_name")["item_id"].to_dict() if not inv.empty else {}
    code2id = inv.set_index("item_barcode")["item_id"].to_dict() if not inv.empty else {}

    created: Dict[int, int] = {}

    if "item_id" not in df2.columns:
        df2["item_id"] = pd.NA

    for idx, row in df2.iterrows():
        if pd.notna(row.get("item_id")):
            continue

        nm = str(row.get("item_name") or "").strip()
        bc = str(row.get("item_barcode") or "").strip()
        bt = str(row.get("bill_type") or "").strip().lower()

        candidate = None
        if nm and nm in name2id:
            candidate = name2id[nm]
        if bc and bc in code2id:
            if candidate is not None and code2id[bc] != candidate:
                raise ValueError(f"Row {idx}: item_name and item_barcode refer to different items.")
            candidate = code2id[bc]

        if candidate is None and allow_create_positive_purchases and bt in ("purchase invoice direct", "purchase invoice"):
            ins = text("""
                INSERT INTO inventory (item_name, item_barcode, category, unit, initial_stock, current_stock)
                VALUES (:name, :code, :cat, :unit, 0, 0)
                ON CONFLICT (item_name, item_barcode) DO UPDATE SET updated_at = CURRENT_TIMESTAMP
                RETURNING item_id;
            """)
            item_id_new = conn.execute(ins, {
                "name": nm or None,
                "code": bc or None,
                "cat":  (row.get("category") or "Uncategorized"),
                "unit": (row.get("unit") or "Psc"),
            }).scalar_one()
            candidate = int(item_id_new)
            if nm: name2id[nm] = candidate
            if bc: code2id[bc] = candidate
            created[candidate] = 0

        if candidate is None:
            raise ValueError(f"Row {idx}: unknown item (name/barcode not found) for bill_type='{bt}'.")

        df2.at[idx, "item_id"] = int(candidate)

    df2["item_id"] = pd.to_numeric(df2["item_id"], errors="raise").astype(int)
    return df2, created
    
def _aggregate_deltas(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp["quantity"] = pd.to_numeric(tmp["quantity"], errors="coerce").fillna(0).round(0).astype(int)
    tmp["__sign"] = tmp["bill_type"].astype(str).map(_bill_sign)
    tmp["__delta"] = tmp["quantity"] * tmp["__sign"]
    agg = tmp.groupby("item_id", as_index=False)["__delta"].sum().rename(columns={"__delta": "delta"})
    agg = agg[agg["delta"] != 0]
    return agg


@run_transaction
def bulk_insert_unified_txns(conn, *, df: pd.DataFrame, schema: str = "public") -> Dict[str, Any]:
    """
    Unified purchases/sales upload with input/output quantity support.
    Expected columns at minimum:
      - bill_type
      - txn_date
      - item_name or item_barcode
      - input_quantity and/or output_quantity
      - (optional) unit_price, category, unit
    Steps:
      - normalize bill_type
      - derive `quantity` from input/output columns by bill_type
      - resolve item_id (auto-create for positive purchases)
      - rename txn_date/unit_price to per-table columns
      - insert into purchases & sales (COPY)
      - update inventory.current_stock using aggregated deltas
    """
    t0 = time.perf_counter()
    out: Dict[str, Any] = {"purchases": None, "sales": None, "inventory_update": None}

    if df.empty:
        raise ValueError("No data rows found.")

    base = df.copy()
    base["bill_type"] = _normalize_bill_type(base.get("bill_type", pd.Series([], dtype="object")))
    base = _derive_quantity_from_io(base)

    # Split routes
    is_sale = base["bill_type"].isin(SALE_TYPES)
    is_purchase = base["bill_type"].isin(PURCHASE_TYPES)
    purchases_raw = base[is_purchase].copy()
    sales_raw = base[is_sale].copy()
    unknown = base[~(is_sale | is_purchase)].copy()

    if not unknown.empty:
        raise ValueError(
            f"{len(unknown)} rows have unknown bill_type. "
            f"Sales: {sorted(SALE_TYPES)} | Purchases: {sorted(PURCHASE_TYPES)}"
        )

    # Purchases
    if not purchases_raw.empty:
        purchases_raw.rename(columns={"txn_date": "purchase_date", "unit_price": "purchase_price"}, inplace=True)
        purchases_resolved, _ = _resolve_item_ids_and_create(conn, purchases_raw, allow_create_positive_purchases=True)
        out["purchases"] = _bulk_insert_core(conn, df=purchases_resolved, table="purchases", schema=schema)

    # Sales
    if not sales_raw.empty:
        sales_raw.rename(columns={"txn_date": "sale_date", "unit_price": "sale_price"}, inplace=True)
        sales_resolved, _ = _resolve_item_ids_and_create(conn, sales_raw, allow_create_positive_purchases=False)
        out["sales"] = _bulk_insert_core(conn, df=sales_resolved, table="sales", schema=schema)

    # Inventory deltas (both)
    both = pd.concat([purchases_raw, sales_raw], ignore_index=True) if (not purchases_raw.empty or not sales_raw.empty) else pd.DataFrame()
    if not both.empty:
        # Ensure item_id present for delta aggregation
        both_ids, _ = _resolve_item_ids_and_create(conn, both, allow_create_positive_purchases=False)
        deltas = _aggregate_deltas(both_ids)
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
            conn.execute(upd, {"ids": deltas["item_id"].tolist(), "ds": deltas["delta"].tolist()})
            out["inventory_update"] = {"items_updated": int(deltas.shape[0])}
        else:
            out["inventory_update"] = {"items_updated": 0}
    else:
        out["inventory_update"] = {"items_updated": 0}

    out["total_ms"] = (time.perf_counter() - t0) * 1000
    return out


# ───────────────────────── convenience wrappers (unchanged) ───────────────────────── #

@run_transaction
def bulk_insert_purchases(conn, *, df: pd.DataFrame, schema: str = "public") -> Dict[str, Any]:
    return _bulk_insert_core(conn, df=df, table="purchases", schema=schema)

@run_transaction
def bulk_insert_sales(conn, *, df: pd.DataFrame, schema: str = "public") -> Dict[str, Any]:
    return _bulk_insert_core(conn, df=df, table="sales", schema=schema)

@run_transaction
def upsert_dataframe(conn, *, df: pd.DataFrame, table: str, schema: str = "public") -> Dict[str, Any]:
    return _bulk_insert_core(conn, df=df, table=table, schema=schema)
