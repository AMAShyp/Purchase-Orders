# upload_handler.py – FAST v2.9.1
# - COPY-based bulk insert (subset headers, numeric coercion)
# - Inventory uploader with skip-duplicates: stage -> merge (ON CONFLICT DO NOTHING)
# - Unified purchases/sales flow with:
#     * input_quantity / output_quantity → single DB quantity
#     * pair matching on (item_name, item_barcode) with repair mode
#     * continue-on-error (skip unfixable), end summary of repairs/skips
# - Prices hardened:
#     * Missing/blank sale_price or purchase_price → 0 (counted)
#     * Any numeric column NaN at COPY time → 0 (prevents NULLs in CSV)
# - Inventory delta UPDATE uses raw psycopg2 (%s) when available; VALUES(...) fallback otherwise.
# - Safe strings: any reference to the COPY NULL marker uses '\\N' inside Python strings.

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
    Strict: every column in df must exist in the table.
    Returns the DB-ordered subset used for insertion.
    """
    db_lower_map = {c.lower(): c for c in db_cols}
    file_lowers = [c.lower() for c in df.columns]
    extra = [c for c in df.columns if c.lower() not in db_lower_map]
    if extra:
        raise ValueError(f"Unknown columns in file: {extra}")
    return [c for c in db_cols if c.lower() in file_lowers]

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
    """
    Try to unwrap a SQLAlchemy Connection/Engine to a raw psycopg2 connection.
    """
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
    q = f'SELECT COUNT(*) FROM "{schema}"."{table}"'
    df = fetch_dataframe(q)
    return int(df.iloc[0, 0])


# ───────────────────────── inventory: COPY with skip duplicates ───────────────────────── #
# FIX: temp table must NOT be schema-qualified. Create as TEMP "tmp_..." only.

@run_transaction
def bulk_insert_inventory_skip_conflicts(
    conn,
    *,
    df: pd.DataFrame,
    schema: str = "public",
) -> Dict[str, Any]:
    """
    Fast inventory insert that skips existing unique conflicts.
    Strategy:
      - CREATE TEMP TABLE tmp LIKE public.inventory
      - COPY df -> tmp (subset columns supported)
      - INSERT INTO public.inventory (...) SELECT ... FROM tmp
        ON CONFLICT DO NOTHING  (skip conflicts on any unique constraint)
    Returns: {"staged": N, "inserted": M, "skipped_duplicates": N-M, "timings": {...}}
    """
    timings: Dict[str, float] = {}
    t0 = time.perf_counter()

    # 1) Fetch table columns & numeric types for coercion + subset mapping
    t = time.perf_counter()
    db_cols = _get_table_columns_ordered(conn, "inventory", schema)
    int_cols = _get_integer_columns(conn, "inventory", schema)
    num_cols = _get_numeric_columns(conn, "inventory", schema)
    timings["fetch_columns_ms"] = (time.perf_counter() - t) * 1000

    # 2) Drop empty rows, align to subset (strict, case-insensitive), rename to DB casing
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

    # 4) Create TEMP staging table (UNQUALIFIED!) and COPY into it
    t = time.perf_counter()
    tmp_name = f'tmp_inv_{int(time.time()*1000)}'
    tmp_quoted = f'"{tmp_name}"'
    fq_inv = f'"{schema}"."inventory"'
    conn.execute(text(f'CREATE TEMP TABLE {tmp_quoted} (LIKE {fq_inv} INCLUDING DEFAULTS) ON COMMIT DROP;'))

    # COPY only the used subset of columns
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

    # Fallback executemany if COPY unavailable
    if not used_copy:
        placeholders = ", ".join([f":{c}" for c in used_cols])
        ins_tmp = text(f'INSERT INTO {tmp_quoted} ({", ".join(used_cols)}) VALUES ({placeholders})')
        payload = df2.where(pd.notnull(df2), None).to_dict(orient="records")
        conn.execute(ins_tmp, payload)

    timings["stage_copy_or_insert_ms"] = (time.perf_counter() - t) * 1000

    # 5) Merge into inventory with ON CONFLICT DO NOTHING (skip any unique conflicts)
    t = time.perf_counter()
    col_list = ", ".join(f'"{c}"' for c in used_cols)
    sql_merge = f"""
        INSERT INTO {fq_inv} ({col_list})
        SELECT {col_list} FROM {tmp_quoted}
        ON CONFLICT DO NOTHING;
    """
    res = conn.execute(text(sql_merge))
    inserted = res.rowcount if res is not None and res.rowcount is not None else 0
    timings["merge_ms"] = (time.perf_counter() - t) * 1000

    # 6) Done
    timings["total_ms"] = (time.perf_counter() - t0) * 1000
    return {
        "staged": staged_rows,
        "inserted": int(inserted),
        "skipped_duplicates": int(staged_rows - inserted),
        "used_copy": used_copy,
        "used_columns": used_cols,
        "timings": timings,
        "temp_table": tmp_name,
    }


# ───────────────────────── unified Purchases/Sales flow ───────────────────────── #

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
    """
    Returns (df_with_price, defaults_count). Any non-numeric or missing values
    become 0. If the column doesn't exist, it's created with 0.
    """
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
    """
    Resolve 'item_id' for each row using (item_name, item_barcode).
    Repairs missing/outdated pairs when safe. Continues past unfixable rows (skipped).
    Returns: resolved_df, created_map (unused), repairs_summary, skipped_summary
    """
    df2 = df.copy()

    # Normalize to strings early
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
            iid = int(r["item_id"])
            nm = str(r["item_name"] or "").strip()
            bc = str(r["item_barcode"] or "").strip()
            if nm:
                name_map[nm] = (iid, bc if bc else None)
            if bc:
                code_map[bc] = (iid, nm if nm else None)
            if nm and bc:
                pair_map[(nm, bc)] = iid

    if "item_id" not in df2.columns:
        df2["item_id"] = pd.NA

    kept_rows: List[int] = []

    for idx, row in df2.iterrows():
        if pd.notna(row.get("item_id")):
            kept_rows.append(idx)
            continue

        nm = row["item_name"]
        bc = row["item_barcode"]
        bt = row["bill_type"]

        # require both; skip if missing
        if nm == "" or bc == "":
            skipped["skipped_missing_pair"] += 1
            continue

        # exact pair
        if (nm, bc) in pair_map:
            df2.at[idx, "item_id"] = int(pair_map[(nm, bc)])
            kept_rows.append(idx)
            continue

        name_hit = name_map.get(nm)  # (iid, stored_bc)
        code_hit = code_map.get(bc)  # (iid, stored_nm)

        # both exist but different ids → resolve by policy
        if name_hit and code_hit and name_hit[0] != code_hit[0]:
            if mismatch_policy == "prefer_barcode":
                repairs["mismatch_preferred_barcode"] += 1
                df2.at[idx, "item_id"] = int(code_hit[0])
            else:
                repairs["mismatch_preferred_name"] += 1
                df2.at[idx, "item_id"] = int(name_hit[0])
            kept_rows.append(idx)
            continue

        # name exists; barcode unknown/different
        if name_hit and not code_hit:
            iid, stored_bc = name_hit
            if (stored_bc is None) or (stored_bc == ""):
                conn.execute(text("""
                    UPDATE inventory SET item_barcode = :bc, updated_at = CURRENT_TIMESTAMP
                    WHERE item_id = :iid
                """), {"bc": bc, "iid": iid})
                repairs["filled_missing_barcode"] += 1
                name_map[nm] = (iid, bc); code_map[bc] = (iid, nm); pair_map[(nm, bc)] = iid
                df2.at[idx, "item_id"] = int(iid); kept_rows.append(idx); continue
            if stored_bc != bc:
                if bc not in code_map and (nm, bc) not in pair_map:
                    conn.execute(text("""
                        UPDATE inventory SET item_barcode = :bc, updated_at = CURRENT_TIMESTAMP
                        WHERE item_id = :iid
                    """), {"bc": bc, "iid": iid})
                    repairs["barcode_changed"] += 1
                    if stored_bc in code_map:
                        del code_map[stored_bc]
                    name_map[nm] = (iid, bc); code_map[bc] = (iid, nm); pair_map[(nm, bc)] = iid
                    df2.at[idx, "item_id"] = int(iid); kept_rows.append(idx); continue
                # prefer policy without DB change
                if mismatch_policy == "prefer_barcode" and bc in code_map:
                    repairs["mismatch_preferred_barcode"] += 1
                    df2.at[idx, "item_id"] = int(code_map[bc][0])
                else:
                    repairs["mismatch_preferred_name"] += 1
                    df2.at[idx, "item_id"] = int(iid)
                kept_rows.append(idx); continue

        # barcode exists; name unknown/different
        if code_hit and not name_hit:
            iid, stored_nm = code_hit
            if (stored_nm is None) or (stored_nm == ""):
                conn.execute(text("""
                    UPDATE inventory SET item_name = :nm, updated_at = CURRENT_TIMESTAMP
                    WHERE item_id = :iid
                """), {"nm": nm, "iid": iid})
                repairs["filled_missing_name"] += 1
                code_map[bc] = (iid, nm); name_map[nm] = (iid, bc); pair_map[(nm, bc)] = iid
                df2.at[idx, "item_id"] = int(iid); kept_rows.append(idx); continue
            if stored_nm != nm:
                if nm not in name_map and (nm, bc) not in pair_map:
                    conn.execute(text("""
                        UPDATE inventory SET item_name = :nm, updated_at = CURRENT_TIMESTAMP
                        WHERE item_id = :iid
                    """), {"nm": nm, "iid": iid})
                    repairs["name_changed"] += 1
                    if stored_nm in name_map:
                        del name_map[stored_nm]
                    code_map[bc] = (iid, nm); name_map[nm] = (iid, bc); pair_map[(nm, bc)] = iid
                    df2.at[idx, "item_id"] = int(iid); kept_rows.append(idx); continue
                if mismatch_policy == "prefer_barcode":
                    repairs["mismatch_preferred_barcode"] += 1
                    df2.at[idx, "item_id"] = int(iid)
                else:
                    repairs["mismatch_preferred_name"] += 1
                    df2.at[idx, "item_id"] = int(name_map[nm][0]) if nm in name_map else int(iid)
                kept_rows.append(idx); continue

        # neither exists → create only for positive purchases
        is_positive_purchase = bt in ("purchase invoice direct", "purchase invoice")
        if allow_create_positive_purchases and is_positive_purchase:
            item_id_new = conn.execute(text("""
                INSERT INTO inventory (item_name, item_barcode, category, unit, initial_stock, current_stock)
                VALUES (:name, :code, :cat, :unit, 0, 0)
                RETURNING item_id;
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

        # otherwise skip
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
    agg = agg[agg["delta"] != 0]
    return agg


def _filter_to_table_columns(conn, df: pd.DataFrame, table: str, schema: str = "public") -> pd.DataFrame:
    db_cols = _get_table_columns_ordered(conn, table, schema)
    db_lower = {c.lower() for c in db_cols}
    keep = [c for c in df.columns if c.lower() in db_lower]
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
    """
    Unified purchases/sales upload with input/output quantity support + repair mode.
    Continues past bad rows; returns detailed summary. Filters helper columns before COPY.
    Ensures price columns are NOT NULL (fill missing with 0). Numeric columns NaN→0 before COPY.
    If update_inventory=False, inventory changes are skipped (insert-only mode) and a delta preview is returned.
    """
    t0 = time.perf_counter()
    out: Dict[str, Any] = {
        "mode": "insert_and_update" if update_inventory else "insert_only",
        "purchases": None,
        "sales": None,
        "inventory_update": None,
        "repairs_summary": None,
        "skipped_summary": None,
        "unknown_bill_type_rows": 0,
        "total_ms": 0.0,
    }

    if df.empty:
        raise ValueError("No data rows found.")

    base = df.copy()
    base["bill_type"] = _normalize_bill_type(base.get("bill_type", pd.Series([], dtype="object")))
    base = _derive_quantity_from_io(base)

    # Route by bill_type (count & drop unknown)
    is_sale = base["bill_type"].isin(SALE_TYPES)
    is_purchase = base["bill_type"].isin(PURCHASE_TYPES)
    unknown_mask = ~(is_sale | is_purchase)
    out["unknown_bill_type_rows"] = int(unknown_mask.sum())
    base = base[~unknown_mask].copy()

    purchases_raw = base[base["bill_type"].isin(PURCHASE_TYPES)].copy()
    sales_raw = base[base["bill_type"].isin(SALE_TYPES)].copy()

    repairs_total = {
        "filled_missing_barcode": 0, "filled_missing_name": 0,
        "barcode_changed": 0, "name_changed": 0,
        "mismatch_preferred_barcode": 0, "mismatch_preferred_name": 0,
        "created_new_items": 0,
        "sale_price_defaulted_to_zero": 0,
        "purchase_price_defaulted_to_zero": 0,
    }
    skipped_total = {"skipped_missing_pair": 0, "skipped_unknown_item": 0}

    # Purchases
    purchases_resolved = pd.DataFrame()
    purchases_insert = pd.DataFrame()
    if not purchases_raw.empty:
        purchases_raw.rename(columns={"txn_date": "purchase_date", "unit_price": "purchase_price"}, inplace=True)
        purchases_resolved, _, repairs_p, skipped_p = _resolve_item_ids_and_create(
            conn,
            purchases_raw,
            allow_create_positive_purchases=True,
            repair_missing_links=True,
            mismatch_policy=mismatch_policy,
        )
        purchases_resolved, defaults_p = _ensure_price_not_null(purchases_resolved, "purchase_price")
        repairs_p["purchase_price_defaulted_to_zero"] = defaults_p
        for k in repairs_total: repairs_total[k] += repairs_p.get(k, 0)
        for k in skipped_total: skipped_total[k] += skipped_p.get(k, 0)
        if not purchases_resolved.empty:
            purchases_insert = _filter_to_table_columns(conn, purchases_resolved, "purchases", schema=schema)
            out["purchases"] = _bulk_insert_core(conn, df=purchases_insert, table="purchases", schema=schema)

    # Sales
    sales_resolved = pd.DataFrame()
    sales_insert = pd.DataFrame()
    if not sales_raw.empty:
        sales_raw.rename(columns={"txn_date": "sale_date", "unit_price": "sale_price"}, inplace=True)
        sales_resolved, _, repairs_s, skipped_s = _resolve_item_ids_and_create(
            conn,
            sales_raw,
            allow_create_positive_purchases=False,
            repair_missing_links=True,
            mismatch_policy=mismatch_policy,
        )
        sales_resolved, defaults_s = _ensure_price_not_null(sales_resolved, "sale_price")
        repairs_s["sale_price_defaulted_to_zero"] = defaults_s
        for k in repairs_total: repairs_total[k] += repairs_s.get(k, 0)
        for k in skipped_total: skipped_total[k] += skipped_s.get(k, 0)
        if not sales_resolved.empty:
            sales_insert = _filter_to_table_columns(conn, sales_resolved, "sales", schema=schema)
            out["sales"] = _bulk_insert_core(conn, df=sales_insert, table="sales", schema=schema)

    # Inventory deltas from the rows we actually inserted
    both_for_delta = pd.DataFrame()
    if not purchases_insert.empty:
        both_for_delta = pd.concat([both_for_delta, purchases_resolved[["item_id", "quantity", "bill_type"]]])
    if not sales_insert.empty:
        both_for_delta = pd.concat([both_for_delta, sales_resolved[["item_id", "quantity", "bill_type"]]])

    delta_summary = {"items": 0, "net_delta": 0}
    deltas = pd.DataFrame()
    if not both_for_delta.empty:
        deltas = _aggregate_deltas(both_for_delta)
        if not deltas.empty:
            delta_summary["items"] = int(deltas.shape[0])
            delta_summary["net_delta"] = int(deltas["delta"].sum())

    if update_inventory and not deltas.empty:
        ids = [int(x) for x in deltas["item_id"].tolist()]
        ds  = [int(x) for x in deltas["delta"].tolist()]

        # Prefer raw psycopg2 cursor with %s placeholders
        raw = _get_raw_psycopg2_connection(conn)
        table_inv = f'"{schema}"."inventory"'
        if raw is not None:
            cur = raw.cursor()
            try:
                cur.execute(
                    f"""
                    WITH changes AS (
                        SELECT unnest(%s::int[]) AS item_id,
                               unnest(%s::int[]) AS d
                    )
                    UPDATE {table_inv} i
                    SET current_stock = i.current_stock + c.d,
                        updated_at = CURRENT_TIMESTAMP
                    FROM changes c
                    WHERE i.item_id = c.item_id AND c.d <> 0;
                    """,
                    (ids, ds),
                )
            finally:
                cur.close()
        else:
            # Fallback: VALUES join (works on SQLAlchemy connection)
            pairs = list(zip(ids, ds))
            values_clause = ", ".join(f"(:id{i}, :d{i})" for i in range(len(pairs)))
            sql_vals = f"""
                UPDATE {table_inv} AS i
                SET current_stock = i.current_stock + v.d,
                    updated_at = CURRENT_TIMESTAMP
                FROM (VALUES {values_clause}) AS v(item_id, d)
                WHERE i.item_id = v.item_id AND v.d <> 0;
            """
            params = {}
            for i, (idv, dv) in enumerate(pairs):
                params[f"id{i}"] = int(idv)
                params[f"d{i}"] = int(dv)
            conn.execute(text(sql_vals), params)

        out["inventory_update"] = {
            "mode": "insert_and_update",
            "items_updated": delta_summary["items"],
            "net_delta": delta_summary["net_delta"],
        }
    else:
        out["inventory_update"] = {
            "mode": "insert_only",
            "items_updated": 0,
            "net_delta_preview": delta_summary["net_delta"],
            "items_would_change": delta_summary["items"],
            "skipped": True,
        }

    out["repairs_summary"] = repairs_total
    out["skipped_summary"] = {**skipped_total, "unknown_bill_type_rows": out["unknown_bill_type_rows"]}
    out["total_ms"] = (time.perf_counter() - t0) * 1000
    return out


# ───────────────────────── convenience wrappers ───────────────────────── #

@run_transaction
def bulk_insert_purchases(conn, *, df: pd.DataFrame, schema: str = "public") -> Dict[str, Any]:
    return _bulk_insert_core(conn, df=df, table="purchases", schema=schema)

@run_transaction
def bulk_insert_sales(conn, *, df: pd.DataFrame, schema: str = "public") -> Dict[str, Any]:
    return _bulk_insert_core(conn, df=df, table="sales", schema=schema)

@run_transaction
def upsert_dataframe(conn, *, df: pd.DataFrame, table: str, schema: str = "public") -> Dict[str, Any]:
    return _bulk_insert_core(conn, df=df, table=table, schema=schema)
