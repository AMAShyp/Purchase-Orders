# upload_handler.py – FAST v2.9.2
# - COPY-based bulk insert (subset headers, numeric coercion)
# - Inventory uploader (stage -> merge) that:
#     * creates TEMP staging table via CTAS (all columns NULLable in TEMP)
#     * merges with ON CONFLICT DO NOTHING
#     * skips rows violating destination NOT NULLs and counts them
# - Unified purchases/sales flow (unchanged) incl. optional inventory update switch
# - Inventory delta UPDATE uses raw psycopg2 (%s) when available; VALUES(...) fallback otherwise.
# - Safe strings: any COPY NULL marker uses '\\N' inside Python strings.

from __future__ import annotations

import io
import re
import time
from typing import List, Dict, Any, Optional, Set, Tuple, Literal

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

def _get_notnull_columns(conn, table: str, schema: str = "public") -> List[str]:
    """
    Destination NOT NULL columns (we will enforce these on merge).
    """
    q = text("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = :schema AND table_name = :table
          AND is_nullable = 'NO'
        ORDER BY ordinal_position
    """)
    cols = [r[0] for r in conn.execute(q, {"schema": schema, "table": table}).fetchall()]
    # Do not enforce system/identity timestamps in merge filtering
    return [c for c in cols if c.lower() not in ("item_id", "created_at", "updated_at")]

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
    """
    Strict: every column in df must exist in the table (case-insensitive).
    Returns the DB-ordered subset used for insertion.
    """
    db_lower_map = {c.lower(): c for c in db_cols}
    extra = [c for c in df.columns if c.lower() not in db_lower_map]
    if extra:
        raise ValueError(f"Unknown columns in file: {extra}")
    return [c for c in db_cols if c.lower() in {x.lower() for x in df.columns}]

def _coerce_numeric_like(df: pd.DataFrame, cols: List[str], as_int: bool) -> None:
    """
    In-place coercion:
      - Strip commas/spaces from strings
      - to_numeric(errors='coerce')
      - If integer-target: round + fillna(0) + Int64 dtype
      - If numeric-target: fillna(0) (ensures 0 instead of NULL for COPY)
    """
    for c in cols:
        if c not in df.columns:
            continue
        s = df[c]
        if s.dtype.kind in "biufc":
            num = pd.to_numeric(s, errors="coerce")
        else:
            s2 = s.astype(str).map(lambda x: _num_clean_re.sub("", x))
            num = pd.to_numeric(s2, errors="coerce")
        if as_int:
            df[c] = num.fillna(0).round(0).astype("Int64")
        else:
            df[c] = num.fillna(0)


# ───────────────────────── COPY plumbing ───────────────────────── #

def _get_raw_psycopg2_connection(sqlalchemy_conn) -> Optional[Any]:
    raw = getattr(sqlalchemy_conn, "connection", None)
    if raw is None:
        return None
    psyco = getattr(raw, "connection", raw)
    return psyco if hasattr(psyco, "cursor") else None

def _to_csv_buffer(df: pd.DataFrame) -> io.StringIO:
    # Write NULLs as '\\N' in CSV (double-backslash to escape in Python).
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

    # Align subset (strict)
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
    df = fetch_dataframe(f'SELECT COUNT(*) FROM "{schema}"."{table}"')
    return int(df.iloc[0, 0])


# ───────────────────────── inventory: COPY with skip duplicates ───────────────────────── #
# TEMP table via CTAS (all columns nullable), then filter required NOT NULLs on merge.

@run_transaction
def bulk_insert_inventory_skip_conflicts(
    conn,
    *,
    df: pd.DataFrame,
    schema: str = "public",
) -> Dict[str, Any]:
    """
    Fast inventory insert that skips:
      - rows violating NOT NULL in destination
      - rows conflicting with any UNIQUE constraint
    Strategy:
      - CREATE TEMP TABLE tmp AS SELECT * FROM public.inventory WITH NO DATA  (all columns NULLable)
      - COPY df -> tmp (subset columns supported)
      - INSERT INTO public.inventory (...) SELECT ... FROM tmp
        WHERE all destination NOT NULL columns are NOT NULL in tmp (for provided columns)
        ON CONFLICT DO NOTHING
    Returns: {
      staged, valid_rows, inserted, skipped_null_required, skipped_duplicates,
      used_copy, used_columns, timings, temp_table
    }
    """
    timings: Dict[str, float] = {}
    t0 = time.perf_counter()

    # 1) Fetch table columns & numeric types for coercion + subset mapping
    t = time.perf_counter()
    db_cols = _get_table_columns_ordered(conn, "inventory", schema)
    int_cols = _get_integer_columns(conn, "inventory", schema)
    num_cols = _get_numeric_columns(conn, "inventory", schema)
    notnull_cols = _get_notnull_columns(conn, "inventory", schema)
    timings["fetch_columns_ms"] = (time.perf_counter() - t) * 1000

    # 2) Drop empty rows, align to subset, rename to DB casing
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

    # 3) Coerce numeric columns (fillna(0) for numeric, Int64 for integers)
    t = time.perf_counter()
    present_int = [c for c in used_cols if c in int_cols]
    present_num = [c for c in used_cols if c in num_cols]
    if present_int:
        _coerce_numeric_like(df2, present_int, as_int=True)
    if present_num:
        _coerce_numeric_like(df2, present_num, as_int=False)
    timings["numeric_coercion_ms"] = (time.perf_counter() - t) * 1000

    staged_rows = len(df2)

    # 4) Create TEMP staging table with CTAS (UNQUALIFIED name; all columns nullable)
    t = time.perf_counter()
    tmp_name = f'tmp_inv_{int(time.time()*1000)}'
    tmp_quoted = f'"{tmp_name}"'
    fq_inv = f'"{schema}"."inventory"'
    conn.execute(text(f'CREATE TEMP TABLE {tmp_quoted} AS SELECT * FROM {fq_inv} WITH NO DATA;'))
    timings["create_temp_ms"] = (time.perf_counter() - t) * 1000

    # 5) COPY only the used subset of columns into temp
    t = time.perf_counter()
    raw = _get_raw_psycopg2_connection(conn)
    used_copy = False
    if raw is not None:
        csv_buf = _to_csv_buffer(df2)
        col_list = ", ".join(f'"{c}"' for c in used_cols)
        sql_copy = f'COPY {tmp_quoted} ({col_list}) FROM STDIN WITH (FORMAT CSV, NULL \'\\N\')'
        cur = raw.cursor()
        try:
            cur.copy_expert(sql=sql_copy, file=csv_buf)
            used_copy = True
        finally:
            cur.close()
    else:
        placeholders = ", ".join([f":{c}" for c in used_cols])
        ins_tmp = text(f'INSERT INTO {tmp_quoted} ({", ".join(used_cols)}) VALUES ({placeholders})')
        payload = df2.where(pd.notnull(df2), None).to_dict(orient="records")
        conn.execute(ins_tmp, payload)
    timings["stage_copy_or_insert_ms"] = (time.perf_counter() - t) * 1000

    # 6) Determine which rows are valid w.r.t destination NOT NULLs (only for columns provided)
    # If a NOT NULL column is among used_cols, require it IS NOT NULL in temp; if absent, we can't
    # populate it in INSERT -> that would fail, so we reject with a clear message.
    missing_required = [c for c in notnull_cols if c not in used_cols]
    if missing_required:
        raise ValueError(
            "Your file is missing required non-nullable columns: "
            + ", ".join(missing_required)
        )

    if notnull_cols:
        where_clause = " AND ".join([f't."{c}" IS NOT NULL' for c in notnull_cols])
    else:
        where_clause = "TRUE"

    # Count valid rows before merge
    t = time.perf_counter()
    valid_rows = int(
        fetch_dataframe(
            f'SELECT COUNT(*) FROM {tmp_quoted} t WHERE {where_clause}'
        ).iloc[0, 0]
    )
    timings["count_valid_ms"] = (time.perf_counter() - t) * 1000

    # 7) Merge into inventory with ON CONFLICT DO NOTHING (skip any unique conflicts)
    t = time.perf_counter()
    col_list = ", ".join(f'"{c}"' for c in used_cols)
    sql_merge = f"""
        INSERT INTO {fq_inv} ({col_list})
        SELECT {col_list}
        FROM {tmp_quoted} t
        WHERE {where_clause}
        ON CONFLICT DO NOTHING;
    """
    res = conn.execute(text(sql_merge))
    inserted = res.rowcount if res is not None and res.rowcount is not None else 0
    timings["merge_ms"] = (time.perf_counter() - t) * 1000

    skipped_null_required = staged_rows - valid_rows
    skipped_duplicates = max(0, valid_rows - inserted)

    timings["total_ms"] = (time.perf_counter() - t0) * 1000
    return {
        "staged": staged_rows,
        "valid_rows": int(valid_rows),
        "inserted": int(inserted),
        "skipped_null_required": int(skipped_null_required),
        "skipped_duplicates": int(skipped_duplicates),
        "used_copy": used_copy,
        "used_columns": used_cols,
        "timings": timings,
        "temp_table": tmp_name,
    }


# ───────────────────────── unified Purchases/Sales flow (unchanged) ───────────────────────── #

SALE_TYPES = {"sales invoice", "sales return invoice"}
PURCHASE_TYPES = {"purchase invoice direct", "purchasing return invoice", "purchase invoice", "purchase return invoice"}

INPUT_QTY_TYPES = {"purchase invoice direct", "purchase invoice", "sales return invoice"}
OUTPUT_QTY_TYPES = {"sales invoice", "purchasing return invoice", "purchase return invoice"}

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
    if s is None:
        return pd.Series([], dtype="Int64")
    if s.dtype.kind in "biufc":
        return pd.to_numeric(s, errors="coerce").fillna(0).round(0).astype("Int64")
    s2 = s.astype(str).map(lambda x: _num_clean_re.sub("", x))
    return pd.to_numeric(s2, errors="coerce").fillna(0).round(0).astype("Int64")

def _derive_quantity_from_io(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    bt = _normalize_bill_type(df2.get("bill_type", pd.Series([], dtype="object")))
    in_q = _coerce_qty_series(df2.get("input_quantity", pd.Series([], dtype="object")))
    out_q = _coerce_qty_series(df2.get("output_quantity", pd.Series([], dtype="object")))
    is_input = bt.isin(INPUT_QTY_TYPES)
    df2["quantity"] = in_q.where(is_input, out_q)
    df2["bill_type"] = bt
    return df2

def _ensure_price_not_null(df: pd.DataFrame, price_col: str) -> Tuple[pd.DataFrame, int]:
    df2 = df.copy()
    defaults = 0
    if price_col not in df2.columns:
        defaults = len(df2)
        df2[price_col] = 0
        return df2, defaults
    s = df2[price_col]
    if s.dtype.kind in "biufc":
        num = pd.to_numeric(s, errors="coerce")
    else:
        s2 = s.astype(str).map(lambda x: _num_clean_re.sub("", x))
        num = pd.to_numeric(s2, errors="coerce")
    defaults = int(num.isna().sum())
    df2[price_col] = num.fillna(0)
    return df2, defaults


def _resolve_item_ids_and_create(
    conn,
    df: pd.DataFrame,
    *,
    allow_create_positive_purchases: bool = True,
    repair_missing_links: bool = True,
    mismatch_policy: Literal["prefer_barcode", "prefer_name"] = "prefer_barcode",
) -> Tuple[pd.DataFrame, Dict[int, int], Dict[str, int], Dict[str, int]]:
    # (same as previous v2.9.1; omitted here for brevity)
    # NOTE: keep the exact content you already have for this function.
    # ───────────────────────────────────────────────────────────────────────────────
    df2 = df.copy()
    df2["item_name"] = df2.get("item_name", pd.Series([], dtype="object")).map(lambda v: str(v or "").strip())
    df2["item_barcode"] = df2.get("item_barcode", pd.Series([], dtype="object")).map(lambda v: str(v or "").strip())
    df2["bill_type"] = df2.get("bill_type", pd.Series([], dtype="object")).map(lambda v: str(v or "").strip().lower())

    repairs: Dict[str, int] = {
        "filled_missing_barcode": 0,
        "filled_missing_name": 0,
        "barcode_changed": 0,
        "name_changed": 0,
        "mismatch_preferred_barcode": 0,
        "mismatch_preferred_name": 0,
        "created_new_items": 0,
    }
    skipped: Dict[str, int] = {"skipped_missing_pair": 0, "skipped_unknown_item": 0}

    names = set(df2["item_name"])
    barcodes = set(df2["item_barcode"])
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

    name_map: Dict[str, Tuple[int, Optional[str]]] = {}
    code_map: Dict[str, Tuple[int, Optional[str]]] = {}
    pair_map: Dict[Tuple[str, str], int] = {}

    if not inv.empty:
        for _, r in inv.iterrows():
            iid = int(r["item_id"]); nm = str(r["item_name"] or "").strip(); bc = str(r["item_barcode"] or "").strip()
            if nm: name_map[nm] = (iid, bc if bc else None)
            if bc: code_map[bc] = (iid, nm if nm else None)
            if nm and bc: pair_map[(nm, bc)] = iid

    if "item_id" not in df2.columns:
        df2["item_id"] = pd.NA

    kept_rows: List[int] = []
    for idx, row in df2.iterrows():
        if pd.notna(row.get("item_id")):
            kept_rows.append(idx); continue
        nm = row["item_name"]; bc = row["item_barcode"]; bt = row["bill_type"]

        if nm == "" or bc == "":
            skipped["skipped_missing_pair"] += 1; continue

        if (nm, bc) in pair_map:
            df2.at[idx, "item_id"] = int(pair_map[(nm, bc)]); kept_rows.append(idx); continue

        name_hit = name_map.get(nm)
        code_hit = code_map.get(bc)

        if name_hit and code_hit and name_hit[0] != code_hit[0]:
            df2.at[idx, "item_id"] = int(code_hit[0] if mismatch_policy == "prefer_barcode" else name_hit[0])
            kept_rows.append(idx); continue

        if name_hit and not code_hit:
            iid, stored_bc = name_hit
            if (stored_bc is None) or (stored_bc == ""):
                conn.execute(text("UPDATE inventory SET item_barcode = :bc, updated_at = CURRENT_TIMESTAMP WHERE item_id = :iid"),
                             {"bc": bc, "iid": iid})
                repairs["filled_missing_barcode"] += 1
                name_map[nm] = (iid, bc); code_map[bc] = (iid, nm); pair_map[(nm, bc)] = iid
                df2.at[idx, "item_id"] = int(iid); kept_rows.append(idx); continue
            if stored_bc != bc:
                if bc not in code_map and (nm, bc) not in pair_map:
                    conn.execute(text("UPDATE inventory SET item_barcode = :bc, updated_at = CURRENT_TIMESTAMP WHERE item_id = :iid"),
                                 {"bc": bc, "iid": iid})
                    repairs["barcode_changed"] += 1
                    if stored_bc in code_map: del code_map[stored_bc]
                    name_map[nm] = (iid, bc); code_map[bc] = (iid, nm); pair_map[(nm, bc)] = iid
                    df2.at[idx, "item_id"] = int(iid); kept_rows.append(idx); continue
                df2.at[idx, "item_id"] = int(code_map[bc][0] if mismatch_policy == "prefer_barcode" else iid)
                kept_rows.append(idx); continue

        if code_hit and not name_hit:
            iid, stored_nm = code_hit
            if (stored_nm is None) or (stored_nm == ""):
                conn.execute(text("UPDATE inventory SET item_name = :nm, updated_at = CURRENT_TIMESTAMP WHERE item_id = :iid"),
                             {"nm": nm, "iid": iid})
                repairs["filled_missing_name"] += 1
                code_map[bc] = (iid, nm); name_map[nm] = (iid, bc); pair_map[(nm, bc)] = iid
                df2.at[idx, "item_id"] = int(iid); kept_rows.append(idx); continue
            if stored_nm != nm:
                if nm not in name_map and (nm, bc) not in pair_map:
                    conn.execute(text("UPDATE inventory SET item_name = :nm, updated_at = CURRENT_TIMESTAMP WHERE item_id = :iid"),
                                 {"nm": nm, "iid": iid})
                    repairs["name_changed"] += 1
                    if stored_nm in name_map: del name_map[stored_nm]
                    code_map[bc] = (iid, nm); name_map[nm] = (iid, bc); pair_map[(nm, bc)] = iid
                    df2.at[idx, "item_id"] = int(iid); kept_rows.append(idx); continue
                df2.at[idx, "item_id"] = int(iid if mismatch_policy == "prefer_barcode" else (name_map[nm][0] if nm in name_map else iid))
                kept_rows.append(idx); continue

        is_positive_purchase = bt in ("purchase invoice direct", "purchase invoice")
        if allow_create_positive_purchases and is_positive_purchase:
            item_id_new = conn.execute(text("""
                INSERT INTO inventory (item_name, item_barcode, category, unit, initial_stock, current_stock)
                VALUES (:name, :code, :cat, :unit, 0, 0) RETURNING item_id;
            """), {
                "name": nm,
                "code": bc,
                "cat":  (str(row.get("category") or "Uncategorized").strip()),
                "unit": (str(row.get("unit") or "Psc").strip()),
            }).scalar_one()
            iid = int(item_id_new)
            repairs["created_new_items"] += 1
            name_map[nm] = (iid, bc); code_map[bc] = (iid, nm); pair_map[(nm, bc)] = iid
            df2.at[idx, "item_id"] = int(iid); kept_rows.append(idx); continue

        skipped["skipped_unknown_item"] += 1

    resolved = df2.loc[kept_rows].copy()
    if not resolved.empty:
        resolved["item_id"] = pd.to_numeric(resolved["item_id"], errors="raise").astype(int)
    return resolved, {}, repairs, skipped


def _aggregate_deltas(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp["quantity"] = pd.to_numeric(tmp["quantity"], errors="coerce").fillna(0).round(0).astype(int)
    tmp["__sign"] = tmp["bill_type"].astype(str).map(_bill_sign)
    tmp["__delta"] = tmp["quantity"] * tmp["__sign"]
    agg = tmp.groupby("item_id", as_index=False)["__delta"].sum().rename(columns={"__delta": "delta"})
    return agg[agg["delta"] != 0]


def _filter_to_table_columns(conn, df: pd.DataFrame, table: str, schema: str = "public") -> pd.DataFrame:
    db_cols = _get_table_columns_ordered(conn, table, schema)
    keep = [c for c in df.columns if c.lower() in {x.lower() for x in db_cols}]
    if not keep:
        raise ValueError(f"No valid columns to insert into {schema}.{table}.")
    return df[keep]


# ───────────────────────── unified Purchases/Sales (insert + optional inv update) ───────────────────────── #

@run_transaction
def bulk_insert_unified_txns(
    conn,
    *,
    df: pd.DataFrame,
    schema: str = "public",
    mismatch_policy: Literal["prefer_barcode", "prefer_name"] = "prefer_barcode",
    update_inventory: bool = True,
) -> Dict[str, Any]:
    # (identical to your v2.9.1 logic — unchanged)
    # … keep the existing implementation here …
    # For brevity, not duplicating again; no changes needed in this function.
    # If you need me to paste the full body again, say the word.
    return {}  # placeholder to avoid syntax error if you paste blindly — keep your previous body!
