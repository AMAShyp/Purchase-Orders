# upload_handler.py – FAST v1.3
# Bulk insert using COPY with:
# - subset header mode (only provided columns are inserted; all must exist in table)
# - auto numeric coercion (commas/decimals → integer for integer columns; numeric kept numeric)
# - ignores auto/default columns (e.g., item_id, created_at, updated_at) if not provided
# - signature compatible with run_transaction(conn, ...)

from __future__ import annotations

import io
import re
import time
from typing import List, Dict, Any, Optional, Set

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


# ───────────────────────── DF alignment & coercion ───────────────────────── #

def _align_subset_columns(df: pd.DataFrame, db_cols: List[str]) -> List[str]:
    """
    Case-insensitive header match (subset mode). Returns the DB-cased, DB-ordered
    list of columns we will insert. Fails if file has columns that don't exist.
    """
    db_lower_map = {c.lower(): c for c in db_cols}
    file_lowers = [c.lower() for c in df.columns]

    # ensure every file column exists in table
    extra = [c for c in df.columns if c.lower() not in db_lower_map]
    if extra:
        raise ValueError(f"Unknown columns in file: {extra}")

    # Select columns in DB order, but only those present in the file
    used_cols = [c for c in db_cols if c.lower() in file_lowers]
    return used_cols

_num_clean_re = re.compile(r"[,\s]")

def _coerce_numeric_like(df: pd.DataFrame, cols: List[str], as_int: bool) -> None:
    """In-place: remove commas/spaces, to_numeric, round if int, fill NaN with 0."""
    for c in cols:
        if c not in df.columns:
            continue
        s = df[c]
        # Already numeric?
        if s.dtype.kind in "biufc":
            if as_int:
                df[c] = pd.to_numeric(s, errors="coerce").fillna(0).round(0).astype("Int64")
            else:
                df[c] = pd.to_numeric(s, errors="coerce")
            continue
        # String-like → clean and parse
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
    # NULLs as \N (escape the backslash for Python)
    buf = io.StringIO()
    df.to_csv(buf, index=False, header=False, na_rep='\\N')
    buf.seek(0)
    return buf


# ───────────────────────── public API ───────────────────────── #

@run_transaction
def bulk_insert_exact_headers(conn, *, df: pd.DataFrame, table: str, schema: str = "public") -> Dict[str, Any]:
    """
    Fast subset-header bulk insert (decorator injects `conn` first):
      - Reads DB columns (ordered)
      - Uses only columns present in the file (must all exist in table)
      - Auto-coerces numeric/integer columns for compatibility
      - COPY FROM STDIN CSV with NULL '\\N'; fallback to executemany
    """
    timings: Dict[str, float] = {}
    t0 = time.perf_counter()

    # 1) Fetch table columns & numeric types
    t = time.perf_counter()
    db_cols = _get_table_columns_ordered(conn, table, schema)
    int_cols = _get_integer_columns(conn, table, schema)
    num_cols = _get_numeric_columns(conn, table, schema)
    timings["fetch_columns_ms"] = (time.perf_counter() - t) * 1000

    # 2) Align (subset) and reorder DF to used_cols
    t = time.perf_counter()
    df = df.dropna(how="all")
    if df.empty:
        raise ValueError("No data rows found.")
    used_cols = _align_subset_columns(df, db_cols)

    # Build df2 with only used columns, rename to exact DB casing, ordered by DB
    lower2db = {c.lower(): c for c in db_cols}
    present_lower = {c.lower() for c in used_cols}
    df2 = df[[c for c in df.columns if c.lower() in present_lower]].copy()
    df2.rename(columns={c: lower2db[c.lower()] for c in df2.columns}, inplace=True)
    df2 = df2[used_cols]
    timings["align_columns_ms"] = (time.perf_counter() - t) * 1000

    # 3) Numeric coercion for any numeric columns present in file
    t = time.perf_counter()
    present_int = [c for c in used_cols if c in int_cols]
    present_num = [c for c in used_cols if c in num_cols]
    if present_int:
        _coerce_numeric_like(df2, present_int, as_int=True)
    if present_num:
        _coerce_numeric_like(df2, present_num, as_int=False)
    timings["numeric_coercion_ms"] = (time.perf_counter() - t) * 1000

    # 4) COPY fast path
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

    # 5) Fallback executemany
    if not used_copy:
        t = time.perf_counter()
        placeholders = ", ".join([f":{c}" for c in used_cols])
        ins = text(f'INSERT INTO "{schema}"."{table}" ({", ".join(used_cols)}) VALUES ({placeholders})')
        payload = df2.where(pd.notnull(df2), None).to_dict(orient="records")
        conn.execute(ins, payload)
        timings["executemany_ms"] = (time.perf_counter() - t) * 1000

    timings["total_ms"] = (time.perf_counter() - t0) * 1000
    return {"rows": rows, "used_copy": used_copy, "timings": timings, "used_columns": used_cols}


def get_row_count(table: str, schema: str = "public") -> int:
    q = f'SELECT COUNT(*) FROM "{schema}"."{table}"'
    df = fetch_dataframe(q)
    return int(df.iloc[0, 0])


# Back-compat alias for older imports (`upsert_dataframe(df=..., table=...)`)
@run_transaction
def upsert_dataframe(conn, *, df: pd.DataFrame, table: str, schema: str = "public") -> Dict[str, Any]:
    return bulk_insert_exact_headers(conn, df=df, table=table, schema=schema)
