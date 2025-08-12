# upload_handler.py – FAST v1
# Purpose: bulk insert if headers match DB table; no per-cell checks.
from __future__ import annotations

import io
import time
from typing import List, Tuple, Dict, Any, Optional

import pandas as pd
from sqlalchemy import text

# robust imports whether used as package or flat
try:
    from ..db_handler import fetch_dataframe, run_transaction
except Exception:
    from db_handler import fetch_dataframe, run_transaction


# ───────────────────────── helpers ───────────────────────── #

def _get_table_columns_ordered(conn, table: str, schema: str = "public") -> List[str]:
    """
    DB column order for COPY; excludes generated/identity defaults (COPY needs explicit list anyway).
    """
    q = text("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = :schema AND table_name = :table
        ORDER BY ordinal_position
    """)
    rows = conn.execute(q, {"schema": schema, "table": table}).fetchall()
    return [r[0] for r in rows]


def _match_and_reorder_columns(df: pd.DataFrame, db_cols: List[str]) -> pd.DataFrame:
    # Case-insensitive match, but we output exact DB names and order.
    map_lower_to_db = {c.lower(): c for c in db_cols}
    df_lower = [c.lower() for c in df.columns]

    # exact set equality required (no extras, no missing)
    if set(df_lower) != set(map_lower_to_db.keys()):
        missing = [c for c in db_cols if c.lower() not in df_lower]
        extra = [c for c in df.columns if c.lower() not in map_lower_to_db]
        raise ValueError(
            f"Header mismatch. Missing in file: {missing or '[]'}; "
            f"Extra in file: {extra or '[]'}.\n"
            f"Expected columns (order irrelevant): {db_cols}"
        )

    # reorder to DB order and rename to exact DB casing
    ordered = [map_lower_to_db[c.lower()] for c in df.columns]  # file order with DB casing
    # Now reorder strictly to DB physical order:
    df = df.rename(columns={c: c for c in df.columns})  # no-op; clarity
    # rebuild dataframe in DB order using mapping
    proj = [map_lower_to_db[c.lower()] for c in db_cols]
    rename_map = {c: map_lower_to_db[c.lower()] for c in df.columns}
    df2 = df.rename(columns=rename_map)
    return df2[proj]


def _get_raw_psycopg2_connection(sqlalchemy_conn) -> Optional[Any]:
    """
    Best-effort unwrap to psycopg2 connection for COPY.
    Supports SQLAlchemy 1.x and 2.x typical stacks.
    """
    raw = getattr(sqlalchemy_conn, "connection", None)
    if raw is None:
        return None
    # SQLAlchemy Connection -> DBAPI connection (psycopg2)
    # sqlalchemy_conn.connection is a ConnectionFairy in old versions; .connection again gives raw
    psyco = getattr(raw, "connection", raw)
    # Check if psycopg2-like (has cursor() and copy_expert via cursor)
    if hasattr(psyco, "cursor"):
        return psyco
    return None


def _to_csv_buffer(df: pd.DataFrame) -> io.StringIO:
    """
    Convert DF to CSV for COPY. Use \N as NULL marker (fast, avoids quoting hell).
    """
    buf = io.StringIO()
    # header=False (we supply column list to COPY), na_rep='\\N' emits literal \N
    df.to_csv(buf, index=False, header=False, na_rep='\\N')
    buf.seek(0)
    return buf


# ───────────────────────── public API ───────────────────────── #

@run_transaction
def bulk_insert_exact_headers(conn, df: pd.DataFrame, table: str, schema: str = "public") -> Dict[str, Any]:
    """
    Ultra-fast bulk insert:
      1) Read DB columns (ordered)
      2) Require exact header match (case-insensitive); reorder to DB order
      3) COPY FROM STDIN CSV with NULL '\N'
    If COPY unavailable, fallback to executemany INSERT.
    Returns timing info.
    """
    timings: Dict[str, float] = {}
    t0 = time.perf_counter()

    # 1) table columns (ordered)
    t = time.perf_counter()
    db_cols = _get_table_columns_ordered(conn, table, schema)
    timings["fetch_columns_ms"] = (time.perf_counter() - t) * 1000

    # 2) align columns (no per-cell checks)
    t = time.perf_counter()
    df = df.dropna(how="all")
    if df.empty:
        raise ValueError("No data rows found.")
    df_aligned = _match_and_reorder_columns(df, db_cols)
    timings["align_columns_ms"] = (time.perf_counter() - t) * 1000

    # 3) COPY (fast path)
    t = time.perf_counter()
    raw = _get_raw_psycopg2_connection(conn)
    rows = len(df_aligned)
    used_copy = False
    if raw is not None:
        csv_buf = _to_csv_buffer(df_aligned)
        col_list = ", ".join(f'"{c}"' for c in db_cols)
        sql = f'COPY "{schema}"."{table}" ({col_list}) FROM STDIN WITH (FORMAT CSV, NULL \'\\N\')'
        cur = raw.cursor()
        try:
            cur.copy_expert(sql=sql, file=csv_buf)
            used_copy = True
        finally:
            cur.close()
    timings["copy_or_insert_ms"] = (time.perf_counter() - t) * 1000

    # Fallback if COPY not used
    if not used_copy:
        t = time.perf_counter()
        placeholders = ", ".join([f":{c}" for c in db_cols])
        ins = text(f'INSERT INTO "{schema}"."{table}" ({", ".join(db_cols)}) VALUES ({placeholders})')
        payload = df_aligned.where(pd.notnull(df_aligned), None).to_dict(orient="records")
        conn.execute(ins, payload)  # executemany
        timings["executemany_ms"] = (time.perf_counter() - t) * 1000

    timings["total_ms"] = (time.perf_counter() - t0) * 1000
    return {"rows": rows, "used_copy": used_copy, "timings": timings}


def get_row_count(table: str, schema: str = "public") -> int:
    q = text(f'SELECT COUNT(*) FROM "{schema}"."{table}"')
    df = fetch_dataframe(q)
    return int(df.iloc[0, 0])
