# upload_handler.py – FAST v2.3
# - Generic COPY-based bulk insert (subset headers, numeric coercion)
# - Unified purchases/sales flow (input/output quantity, item_id resolved)
# - Repair mode (fills missing OR mismatched name/barcode when safe), counting repairs

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

SALE_TYPES = {"sales invoice", "sales return invoice"}
PURCHASE_TYPES = {
    "purchase invoice direct",
    "purchasing return invoice",
    # aliases
    "purchase invoice",
    "purchase return invoice",
}

# Excel IO quantity mapping
INPUT_QTY_TYPES  = {"purchase invoice direct", "purchase invoice", "sales return invoice"}
OUTPUT_QTY_TYPES = {"sales invoice", "purchasing return invoice", "purchase return invoice"}

def _normalize_bill_type(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip().str.casefold()

def _bill_sign(bt: str) -> int:
    bt = (bt or "").strip().lower()
    if bt in ("sales invoice",): return -1
    if bt in ("sales return invoice",): return +1
    if bt in ("purchase invoice direct", "purchase invoice"): return +1
    if bt in ("purchasing return invoice", "purchase return invoice"): return -1
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
    df2["quantity"] = in_q.where(bt.isin(INPUT_QTY_TYPES), out_q)
    df2["bill_type"] = bt
    return df2


def _resolve_item_ids_and_create(
    conn,
    df: pd.DataFrame,
    *,
    allow_create_positive_purchases: bool = True,
    repair_missing_links: bool = True,      # fill NULLs
    repair_mismatched_pairs: bool = True,   # change mismatched values safely
) -> Tuple[pd.DataFrame, Dict[int, int], int]:
    """
    STRICT pair matching + REPAIR:
      - Each row MUST have both item_name and item_barcode (non-empty after str/strip).
      - If exact (name, barcode) pair exists → use it.
      - If name or barcode exists but NOT the exact pair:
          * If both map to different item_ids → error.
          * If one side exists and the other is NULL OR mismatched → if repair mode ON
            and no conflict, UPDATE that inventory row to set/correct the missing/mismatched field.
      - If neither exists:
          * create item ONLY for positive purchases (purchase invoice direct / purchase invoice),
            otherwise error.
    Returns (df_with_item_id, created_map, repairs_count).
    """
    df2 = df.copy()

    # Normalize to strings early
    df2["item_name"] = df2.get("item_name", pd.Series([], dtype="object")).map(lambda v: str(v or "").strip())
    df2["item_barcode"] = df2.get("item_barcode", pd.Series([], dtype="object")).map(lambda v: str(v or "").strip())
    df2["bill_type"] = df2.get("bill_type", pd.Series([], dtype="object")).map(lambda v: str(v or "").strip().lower())

    # Require both fields per row
    missing_mask = (df2["item_name"] == "") | (df2["item_barcode"] == "")
    if missing_mask.any():
        bad_idx = missing_mask[missing_mask].index.tolist()[:5]
        raise ValueError(
            f"Rows {bad_idx} missing item_name and/or item_barcode. "
            "Both are required for strict pair matching."
        )

    # Build lookup sets
    names = set(df2["item_name"])
    barcodes = set(df2["item_barcode"])

    # Fetch inventory
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

    # Build maps
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

    created: Dict[int, int] = {}
    repairs = 0

    if "item_id" not in df2.columns:
        df2["item_id"] = pd.NA

    # Process rows
    for idx, row in df2.iterrows():
        if pd.notna(row.get("item_id")):
            continue

        nm = row["item_name"]
        bc = row["item_barcode"]
        bt = row["bill_type"]

        # 1) Exact pair exists
        if (nm, bc) in pair_map:
            df2.at[idx, "item_id"] = int(pair_map[(nm, bc)])
            continue

        # 2) Pair not found → check separate hits
        name_hit = name_map.get(nm)  # (id, stored_bc)
        code_hit = code_map.get(bc)  # (id, stored_nm)

        if name_hit and code_hit:
            # Both exist separately
            if name_hit[0] != code_hit[0]:
                raise ValueError(
                    f"Row {idx}: item_name and item_barcode refer to different existing items "
                    f"(name→id {name_hit[0]}, barcode→id {code_hit[0]})."
                )
            # Same id but pair incomplete or mismatched → repair if enabled and safe
            iid = name_hit[0]
            stored_bc = name_hit[1]
            stored_nm = code_hit[1]
            need_bc_fix = (stored_bc is None) or (stored_bc != bc)
            need_nm_fix = (stored_nm is None) or (stored_nm != nm)
            if (need_bc_fix or need_nm_fix) and (repair_missing_links or repair_mismatched_pairs):
                # Conflicts?
                other_for_pair = pair_map.get((nm, bc))
                if other_for_pair is not None and other_for_pair != iid:
                    raise ValueError(
                        f"Row {idx}: cannot repair to pair (name,barcode) already used by item {other_for_pair}."
                    )
                if bc in code_map and code_map[bc][0] != iid:
                    raise ValueError(f"Row {idx}: cannot repair; barcode '{bc}' belongs to item {code_map[bc][0]}.")
                if nm in name_map and name_map[nm][0] != iid:
                    raise ValueError(f"Row {idx}: cannot repair; name '{nm}' belongs to item {name_map[nm][0]}.")

                # Remove old reverse mappings if changing
                if stored_bc and stored_bc != bc:
                    try:
                        if code_map.get(stored_bc, (None,))[0] == iid:
                            del code_map[stored_bc]
                    except Exception:
                        pass
                if stored_nm and stored_nm != nm:
                    try:
                        if name_map.get(stored_nm, (None,))[0] == iid:
                            del name_map[stored_nm]
                    except Exception:
                        pass

                upd = text("""
                    UPDATE inventory
                    SET item_name = :nm,
                        item_barcode = :bc,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE item_id = :iid
                """)
                conn.execute(upd, {"nm": nm, "bc": bc, "iid": iid})
                name_map[nm] = (iid, bc)
                code_map[bc] = (iid, nm)
                pair_map[(nm, bc)] = iid
                repairs += 1
            df2.at[idx, "item_id"] = int(iid)
            continue

        if name_hit and not code_hit:
            iid, stored_bc = name_hit
            if stored_bc is None and repair_missing_links:
                if bc in code_map and code_map[bc][0] != iid:
                    raise ValueError(f"Row {idx}: cannot repair; barcode '{bc}' belongs to item {code_map[bc][0]}.")
                upd = text("""
                    UPDATE inventory
                    SET item_barcode = :bc, updated_at = CURRENT_TIMESTAMP
                    WHERE item_id = :iid
                """)
                conn.execute(upd, {"bc": bc, "iid": iid})
                name_map[nm] = (iid, bc)
                code_map[bc] = (iid, nm)
                pair_map[(nm, bc)] = iid
                repairs += 1
                df2.at[idx, "item_id"] = int(iid)
                continue
            # Mismatch (stored_bc different) → change if allowed and safe
            if repair_mismatched_pairs and stored_bc and stored_bc != bc:
                if bc in code_map and code_map[bc][0] != iid:
                    raise ValueError(f"Row {idx}: cannot repair; barcode '{bc}' belongs to item {code_map[bc][0]}.")
                # Drop old mapping for previous barcode
                try:
                    if code_map.get(stored_bc, (None,))[0] == iid:
                        del code_map[stored_bc]
                except Exception:
                    pass
                upd = text("""
                    UPDATE inventory
                    SET item_barcode = :bc, updated_at = CURRENT_TIMESTAMP
                    WHERE item_id = :iid
                """)
                conn.execute(upd, {"bc": bc, "iid": iid})
                name_map[nm] = (iid, bc)
                code_map[bc] = (iid, nm)
                pair_map[(nm, bc)] = iid
                repairs += 1
                df2.at[idx, "item_id"] = int(iid)
                continue
            raise ValueError(
                f"Row {idx}: item_name exists (id {iid}) but provided barcode '{bc}' "
                f"does not match stored barcode '{stored_bc or ''}'."
            )

        if code_hit and not name_hit:
            iid, stored_nm = code_hit
            if stored_nm is None and repair_missing_links:
                if nm in name_map and name_map[nm][0] != iid:
                    raise ValueError(f"Row {idx}: cannot repair; name '{nm}' belongs to item {name_map[nm][0]}.")
                upd = text("""
                    UPDATE inventory
                    SET item_name = :nm, updated_at = CURRENT_TIMESTAMP
                    WHERE item_id = :iid
                """)
                conn.execute(upd, {"nm": nm, "iid": iid})
                name_map[nm] = (iid, bc)
                code_map[bc] = (iid, nm)
                pair_map[(nm, bc)] = iid
                repairs += 1
                df2.at[idx, "item_id"] = int(iid)
                continue
            if repair_mismatched_pairs and stored_nm and stored_nm != nm:
                if nm in name_map and name_map[nm][0] != iid:
                    raise ValueError(f"Row {idx}: cannot repair; name '{nm}' belongs to item {name_map[nm][0]}.")
                try:
                    if name_map.get(stored_nm, (None,))[0] == iid:
                        del name_map[stored_nm]
                except Exception:
                    pass
                upd = text("""
                    UPDATE inventory
                    SET item_name = :nm, updated_at = CURRENT_TIMESTAMP
                    WHERE item_id = :iid
                """)
                conn.execute(upd, {"nm": nm, "iid": iid})
                name_map[nm] = (iid, bc)
                code_map[bc] = (iid, nm)
                pair_map[(nm, bc)] = iid
                repairs += 1
                df2.at[idx, "item_id"] = int(iid)
                continue
            raise ValueError(
                f"Row {idx}: item_barcode exists (id {iid}) but provided name '{nm}' "
                f"does not match stored name '{stored_nm or ''}'."
            )

        # 3) Neither exists → allow create for positive purchases only
        is_positive_purchase = bt in ("purchase invoice direct", "purchase invoice")
        if allow_create_positive_purchases and is_positive_purchase:
            ins = text("""
                INSERT INTO inventory (item_name, item_barcode, category, unit, initial_stock, current_stock)
                VALUES (:name, :code, :cat, :unit, 0, 0)
                RETURNING item_id;
            """)
            item_id_new = conn.execute(ins, {
                "name": nm,
                "code": bc,
                "cat":  (str(row.get("category") or "Uncategorized").strip()),
                "unit": (str(row.get("unit") or "Psc").strip()),
            }).scalar_one()
            iid = int(item_id_new)
            df2.at[idx, "item_id"] = iid
            name_map[nm] = (iid, bc)
            code_map[bc] = (iid, nm)
            pair_map[(nm, bc)] = iid
            created[iid] = 0
            continue

        raise ValueError(
            f"Row {idx}: unknown item (name+barcode pair not found). "
            f"For sales/returns, the item must already exist. For positive purchases, creation is allowed."
        )

    df2["item_id"] = pd.to_numeric(df2["item_id"], errors="raise").astype(int)
    return df2, created, repairs


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
    Unified purchases/sales upload with input/output quantity support + repair mode.
    Returns per-table insert stats and a repairs summary.
    """
    t0 = time.perf_counter()
    out: Dict[str, Any] = {"purchases": None, "sales": None, "inventory_update": None, "repairs": None}

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

    repairs_p = repairs_s = 0

    # Purchases (repair ON, create allowed)
    purchases_resolved = None
    if not purchases_raw.empty:
        purchases_raw.rename(columns={"txn_date": "purchase_date", "unit_price": "purchase_price"}, inplace=True)
        purchases_resolved, _created_p, repairs_p = _resolve_item_ids_and_create(
            conn, purchases_raw, allow_create_positive_purchases=True,
            repair_missing_links=True, repair_mismatched_pairs=True
        )
        out["purchases"] = _bulk_insert_core(conn, df=purchases_resolved, table="purchases", schema=schema)

    # Sales (repair ON, create NOT allowed)
    sales_resolved = None
    if not sales_raw.empty:
        sales_raw.rename(columns={"txn_date": "sale_date", "unit_price": "sale_price"}, inplace=True)
        sales_resolved, _created_s, repairs_s = _resolve_item_ids_and_create(
            conn, sales_raw, allow_create_positive_purchases=False,
            repair_missing_links=True, repair_mismatched_pairs=True
        )
        out["sales"] = _bulk_insert_core(conn, df=sales_resolved, table="sales", schema=schema)

    # Inventory deltas (use already-resolved frames to avoid re-repairing)
    both_ids = []
    if purchases_resolved is not None:
        both_ids.append(purchases_resolved[["item_id", "quantity", "bill_type"]])
    if sales_resolved is not None:
        both_ids.append(sales_resolved[["item_id", "quantity", "bill_type"]])
    if both_ids:
        both = pd.concat(both_ids, ignore_index=True)
        deltas = _aggregate_deltas(both)
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

    out["repairs"] = {"purchases": int(repairs_p), "sales": int(repairs_s), "total": int(repairs_p + repairs_s)}
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
