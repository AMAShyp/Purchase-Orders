# upload_handler.py – FAST v1.1
# Purpose: bulk insert if headers match DB table; no cell-level checks.

from __future__ import annotations

import io
import time
from typing import List, Dict, Any, Optional

import pandas as pd
from sqlalchemy import text

# Robust imports whether used as package or flat
try:
    from ..db_handler import fetch_dataframe, run_transaction
except Exception:
    from db_handler import fetch_dataframe, run_transaction


# ───────────────────────── helpers ───────────────────────── #

def _get_table_columns_ordered(conn, table: str, schema: str = "public") -> List[str]:
    """
    Return DB column order for COPY (physical order from information_schema).
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
    """
    Case-insensitive header match against DB; disallow extras; no missing.
    Reorder to DB order and return a DataFrame whose columns exactly equal db_cols.
    """
    map_lower_to_db = {c.lower(): c for c in db_cols}
    df_lower = [c.lower() for c in df.columns]

    if set(df_lower) != set(map_lower_to_db.keys()):
        missing = [c for c in db_cols if c.lower() not in df_lower]
        extra = [c for c in df.columns if c.lower() not in map_lower_to_db]
        raise ValueError(
            "Header mismatch.\n"
            f"Missing in file: {missing or '[]'}\n"
            f"Extra in file:   {extra or '[]'}\n"
            f"Expected columns (order irrelevant): {db_cols}"
        )

    # Rename to exact DB casing, then project in DB order
    rename_map = {c: map_lower_to_db[c.lower()] for c in df.columns}
    df2 = df.rename(columns=rename_map)
    return df2[db_cols]


def _get_raw_psycopg2_connection(sqlalchemy_conn) -> Optional[Any]:
    """
    Best-effort unwrap to psycopg2 connection for COPY.
    """
    raw = getattr(sqlalchemy_conn, "connection", None)
    if raw is None:
        return None
    psyco = getattr(raw, "connection", raw)
    return psyco if hasattr(psyco, "cursor") else None


def _to_csv_buffer(df: pd.DataFrame) -> io.StringIO:
    """
    Convert DF to CSV for COPY. Use \\N as NULL marker.
    """
    buf = io.StringIO()
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
      3) COPY FROM STDIN CSV with NULL '\\N'
    Fallback: executemany INSERT if COPY unavailable.
    Returns timing info.
    """
    timings: Dict[str, float] = {}
    t0 = time.perf_counter()

    # Columns
    t = time.perf_counter()
    db_cols = _get_table_columns_ordered(conn, table, schema)
    timings["fetch_columns_ms"] = (time.perf_counter() - t) * 1000

    # Align
    t = time.perf_counter()
    df = df.dropna(how="all")
    if df.empty:
        raise ValueError("No data rows found.")
    df_aligned = _match_and_reorder_columns(df, db_cols)
    timings["align_columns_ms"] = (time.perf_counter() - t) * 1000

    # COPY fast path
    rows = len(df_aligned)
    used_copy = False
    t = time.perf_counter()
    raw = _get_raw_psycopg2_connection(conn)
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

    # Fallback executemany
    if not used_copy:
        t = time.perf_counter()
        placeholders = ", ".join([f":{c}" for c in db_cols])
        ins = text(f'INSERT INTO "{schema}"."{table}" ({", ".join(db_cols)}) VALUES ({placeholders})')
        payload = df_aligned.where(pd.notnull(df_aligned), None).to_dict(orient="records")
        conn.execute(ins, payload)
        timings["executemany_ms"] = (time.perf_counter() - t) * 1000

    timings["total_ms"] = (time.perf_counter() - t0) * 1000
    return {"rows": rows, "used_copy": used_copy, "timings": timings}


def get_row_count(table: str, schema: str = "public") -> int:
    q = f'SELECT COUNT(*) FROM "{schema}"."{table}"'
    df = fetch_dataframe(q)
    return int(df.iloc[0, 0])


# ─────────── Back-compat alias (your __init__.py imports this) ───────── #
# Make your existing `from .upload_handler import upsert_dataframe` work.
def upsert_dataframe(*, df: pd.DataFrame, table: str, **kwargs) -> Dict[str, Any]:
    """
    Compatibility wrapper -> calls bulk_insert_exact_headers.
    """
    # run_transaction wrapper expects a connection first; our bulk_* already wrapped.
    return bulk_insert_exact_headers(df=df, table=table)
