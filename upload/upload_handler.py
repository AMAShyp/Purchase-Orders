# upload_handler.py – v6 (final, with robust imports)
from __future__ import annotations
import pandas as pd
from typing import Literal, Optional, Dict, List
from sqlalchemy import text

# Robust import for db_handler (works whether run as a package or flat script)
try:
    from ..db_handler import fetch_dataframe, run_transaction  # if upload/ is a package inside a project
except Exception:
    from db_handler import fetch_dataframe, run_transaction     # if db_handler.py is on PYTHONPATH

# ───────────────────────── Column rules ──────────────────────────────── #
BASE_REQ = {
    "inventory": {"item_name", "item_barcode", "category", "unit",
                  "initial_stock", "current_stock"},
    "purchases": {"bill_type", "quantity", "purchase_price"},
    "sales":     {"bill_type", "quantity", "sale_price"},
}

DATE_COLS = {
    "purchases": ["purchase_date"],
    "sales":     ["sale_date"],
}

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

# ───────────────────────── Debug sink ────────────────────────────────── #
class DebugSink:
    """Collects debug messages; UI can display these."""
    def __init__(self):
        self.events: List[str] = []

    def log(self, msg: str):
        self.events.append(str(msg))

    def extend(self, msgs: List[str]):
        self.events.extend([str(m) for m in msgs])

    def dump(self) -> List[str]:
        return list(self.events)

# ───────────────────────── Public helpers ────────────────────────────── #
def check_columns(df: pd.DataFrame,
                  table: Literal["inventory", "purchases", "sales"]) -> None:
    required = BASE_REQ[table].copy()
    # For purchases/sales, require at least one identifier
    if table in ("purchases", "sales"):
        if not ({"item_name", "item_barcode", "item_id"} & set(df.columns)):
            raise ValueError("Need item_name, item_barcode or item_id column.")
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")

# ───────────────────────── Normalizers ───────────────────────────────── #
def _normalize_dates(df: pd.DataFrame, table: str, dbg: Optional[DebugSink]) -> pd.DataFrame:
    for col in DATE_COLS.get(table, []):
        if col not in df.columns:  # optional
            if dbg: dbg.log(f"[dates] Optional date column '{col}' not present; skipping.")
            continue
        raw = df[col].copy()
        df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True).dt.date
        bad = raw.notna() & df[col].isna()
        if bad.any():
            rows = df.index[bad].tolist()
            raise ValueError(
                f"Invalid {col} in rows {rows}. Use YYYY-MM-DD or a recognised date."
            )
        if dbg: dbg.log(f"[dates] Normalised '{col}' OK.")
    return df

def _strip_strings(df: pd.DataFrame, dbg: Optional[DebugSink]) -> pd.DataFrame:
    for col in ("item_name", "item_barcode", "bill_type", "unit", "category"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    if dbg: dbg.log("[norm] Trimmed strings for item_name/item_barcode/bill_type/unit/category.")
    return df

def _row_sign(table: str, bill_type: str) -> int:
    bt = (bill_type or "").strip().lower()
    try:
        return BILL_SIGNS[table][bt]
    except KeyError:
        raise ValueError(f"Unknown bill_type '{bill_type}' for {table}. "
                         f"Expected one of: {list(BILL_SIGNS[table].keys())}")

# ───────────────────────── Main upsert function ─────────────────────── #
@run_transaction
def upsert_dataframe(conn,
                     df: pd.DataFrame,
                     table: Literal["inventory", "purchases", "sales"],
                     debug: Optional[DebugSink] = None) -> None:
    dbg = debug or DebugSink()

    # Preflight
    orig_rows = df.shape[0]
    df = df.dropna(how="all")
    if df.empty:
        raise ValueError("No data rows found.")
    dbg.log(f"[pre] Received {orig_rows} rows; {df.shape[0]} non-empty rows remain.")

    check_columns(df, table)
    dbg.log(f"[pre] Column check passed for table '{table}'.")

    df = _strip_strings(df, dbg)
    df = _normalize_dates(df, table, dbg)

    if table in ("purchases", "sales"):
        _handle_purchases_or_sales(conn, df, table, dbg)
    else:
        dbg.log("[inventory] Inserting raw inventory rows.")
        inserted = _insert_dataframe(conn, df, "inventory", dbg)
        dbg.log(f"[inventory] Inserted {inserted} rows into inventory.")

    # Final DB snapshot (cheap sanity)
    try:
        if table == "inventory":
            count = fetch_dataframe("SELECT COUNT(*) AS c FROM inventory;").iloc[0]["c"]
            dbg.log(f"[post] Inventory row count now: {count}.")
        elif table == "purchases":
            count = fetch_dataframe("SELECT COUNT(*) AS c FROM purchases;").iloc[0]["c"]
            dbg.log(f"[post] Purchases row count now: {count}.")
        else:
            count = fetch_dataframe("SELECT COUNT(*) AS c FROM sales;").iloc[0]["c"]
            dbg.log(f"[post] Sales row count now: {count}.")
    except Exception as e:
        dbg.log(f"[post] Snapshot count failed: {e}")

# ───────────────────────── Purchases / Sales logic ──────────────────── #
def _handle_purchases_or_sales(conn, df: pd.DataFrame, table: str, dbg: DebugSink) -> None:
    need_names = set(df["item_name"].dropna().astype(str).str.strip()) if "item_name" in df.columns else set()
    need_codes = set(df["item_barcode"].dropna().astype(str).str.strip()) if "item_barcode" in df.columns else set()

    dbg.log(f"[resolve] Need mapping for {len(need_names)} names, {len(need_codes)} barcodes.")

    inv = pd.DataFrame()
    try:
        inv = fetch_dataframe(
            "SELECT item_id, item_name, item_barcode FROM inventory "
            "WHERE item_name = ANY(%(names)s) OR item_barcode = ANY(%(codes)s);",
            params={"names": list(need_names) or ["__none__"], "codes": list(need_codes) or ["__none__"]},
        )
    except Exception as e:
        dbg.log(f"[resolve] Param query failed ({e}); falling back to full scan for mapping.")
        inv = fetch_dataframe("SELECT item_id, item_name, item_barcode FROM inventory;")

    name2id: Dict[str, int] = {}
    code2id: Dict[str, int] = {}
    if not inv.empty:
        name2id = inv.set_index("item_name")["item_id"].to_dict()
        code2id = inv.set_index("item_barcode")["item_id"].to_dict()

    dbg.log(f"[resolve] Pre-resolved {len(name2id)} by name, {len(code2id)} by barcode.")

    inserted_qty: Dict[int, int] = {}

    for idx, row in df.iterrows():
        sign = _row_sign(table, row.get("bill_type"))
        df.at[idx, "__sign"] = sign

        item_id = row.get("item_id")
        if pd.notna(item_id):
            df.at[idx, "item_id"] = int(item_id)
            continue

        candidate = None
        nm = (row.get("item_name") or "").strip()
        bc = (row.get("item_barcode") or "").strip()

        if nm and nm in name2id:
            candidate = name2id[nm]
        if bc and bc in code2id:
            if candidate is not None and code2id[bc] != candidate:
                raise ValueError(f"Row {idx}: item_name and item_barcode map to different items.")
            candidate = code2id[bc]

        if candidate is None:
            if table == "purchases" and sign > 0:
                qty = int(pd.to_numeric(row["quantity"], errors="coerce"))
                if qty < 0:
                    raise ValueError(f"Row {idx}: negative quantity not allowed.")
                ins = text("""
                    INSERT INTO inventory
                    (item_name, item_barcode, category, initial_stock, current_stock, unit)
                    VALUES (:name, :code, :cat, :qty, :qty, :unit)
                    ON CONFLICT (item_name, item_barcode)
                    DO UPDATE SET updated_at = CURRENT_TIMESTAMP
                    RETURNING item_id;
                """)
                item_id_new = conn.execute(ins, {
                    "name": nm or None,
                    "code": bc or None,
                    "cat":  (row.get("category") or "Uncategorized"),
                    "unit": (row.get("unit") or "Psc"),
                    "qty":  qty
                }).scalar_one()
                inserted_qty[int(item_id_new)] = qty
                if nm: name2id[nm] = int(item_id_new)
                if bc: code2id[bc] = int(item_id_new)
                df.at[idx, "item_id"] = int(item_id_new)
                dbg.log(f"[new-item] Row {idx}: created item_id={item_id_new} (qty={qty}).")
            else:
                raise ValueError(
                    f"Row {idx}: unknown item for {table} with bill_type='{row.get('bill_type')}'."
                )
        else:
            df.at[idx, "item_id"] = int(candidate)

    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0).astype(int)
    if (df["quantity"] < 0).any():
        bad = df.index[df["quantity"] < 0].tolist()
        raise ValueError(f"Negative quantity not allowed (rows {bad}).")

    price_col = "purchase_price" if table == "purchases" else "sale_price"
    if price_col in df.columns:
        df[price_col] = pd.to_numeric(df[price_col], errors="coerce")

    df["_delta"] = df["quantity"] * df["__sign"]

    drop_cols = [c for c in ("item_name", "item_barcode", "__sign", "_delta") if c in df.columns]
    to_insert = df.drop(columns=drop_cols).copy()
    dbg.log(f"[insert] Preparing to insert {to_insert.shape[0]} rows into '{table}'. Columns: {list(to_insert.columns)}")

    inserted = _insert_dataframe(conn, to_insert, table, dbg)
    dbg.log(f"[insert] Inserted {inserted} rows into '{table}'.")

    deltas = df.groupby("item_id")["_delta"].sum().reset_index()
    dbg.log(f"[stock] Computed deltas for {len(deltas)} items. Sample: {deltas.head(5).to_dict(orient='records')}")

    if table == "purchases" and inserted_qty:
        deltas["_delta"] = deltas.apply(
            lambda r: int(r["_delta"] - inserted_qty.get(int(r["item_id"]), 0)), axis=1
        )
        dbg.log(f"[stock] Adjusted deltas for brand-new items: {inserted_qty}")

    deltas = deltas[deltas["_delta"] != 0]
    if deltas.empty:
        dbg.log("[stock] No net stock change required.")
        return

    upd_sql = text("""
        WITH changes AS (
            SELECT UNNEST(:ids::int[]) AS item_id, UNNEST(:ds::int[]) AS d
        )
        UPDATE inventory i
        SET current_stock = i.current_stock + c.d,
            updated_at = CURRENT_TIMESTAMP
        FROM changes c
        WHERE i.item_id = c.item_id AND c.d <> 0;
    """)
    conn.execute(upd_sql, {"ids": deltas["item_id"].tolist(), "ds": deltas["_delta"].tolist()})
    dbg.log(f"[stock] Applied stock updates for {len(deltas)} items.")

# ───────────────────────── Generic INSERT helper ────────────────────── #
def _insert_dataframe(conn, df: pd.DataFrame, table: str, dbg: Optional[DebugSink]) -> int:
    if df.empty:
        if dbg: dbg.log(f"[insert] No rows to insert into '{table}'.")
        return 0
    cols = list(df.columns)
    placeholders = ", ".join([f":{c}" for c in cols])
    sql = text(f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders});")

    payload = df.where(pd.notnull(df), None).to_dict(orient="records")

    if dbg:
        sample = payload[:3]
        dbg.log(f"[insert] SQL: INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders});")
        dbg.log(f"[insert] First 3 payload rows: {sample}")

    conn.execute(sql, payload)
    return len(payload)
