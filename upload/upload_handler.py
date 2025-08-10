"""
upload_handler.py – v10
Changes vs v9:
• Inventory now UPSERTs on unique (item_name) using the existing constraint
  "inventory_item_name_key" so duplicates update instead of failing.
• Treat 'nan'/'none'/'null'/'' item_name as blank → explicit error with row indices.
• Intra-file de-dupe on item_name (keep='last').
• Keeps v9 fixes: numeric parsing, optional sign flip, barcode normalization,
  purchases/sales bill-type logic, new-item creation only on positive Purchase Invoice.
"""

from __future__ import annotations
import re
import pandas as pd
from typing import Literal, Dict, Any
from sqlalchemy import text
from db_handler import fetch_dataframe, run_transaction

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

# Toggle: when True, inventory uses ON CONFLICT … DO UPDATE (replace existing rows)
UPSERT_EXISTING_INVENTORY = True
# If most of the sheet has BOTH stocks negative, auto-flip signs to positive
AUTO_FLIP_NEGATIVE_THRESHOLD = 0.80

# ───────────────────────── Dates ─────────────────────────────────────── #
def check_columns(df: pd.DataFrame,
                  table: Literal["inventory", "purchases", "sales"]) -> None:
    required = BASE_REQ[table].copy()
    if table in ("purchases", "sales"):
        if not ({"item_name", "item_barcode", "item_id"} & set(df.columns)):
            raise ValueError("Need item_name, item_barcode or item_id column.")
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")

def _normalise_dates(df: pd.DataFrame, table: str) -> pd.DataFrame:
    for col in DATE_COLS.get(table, []):
        if col not in df.columns:
            continue
        raw = df[col].copy()
        df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True).dt.date
        bad = raw.notna() & df[col].isna()
        if bad.any():
            rows = df.index[bad].tolist()
            raise ValueError(
                f"Invalid {col} in rows {rows}. Use YYYY-MM-DD or a recognised date."
            )
    return df

# ───────────────────────── Helpers ───────────────────────────────────── #
_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

def _clean_number_like(s: Any) -> str:
    """Normalise messy number strings (Arabic digits, commas, spaces, weird dashes)."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    s = str(s).strip().translate(_ARABIC_DIGITS)
    s = s.replace("٬", "").replace("،", "").replace(",", "").replace(" ", "")
    s = s.replace("−", "-").replace("—", "-").replace("–", "-")
    s = re.sub(r"[^0-9\.\-]", "", s)
    if s.count(".") > 1:
        first, *rest = s.split(".")
        s = first + "." + "".join(rest)
    return s

def _norm_barcode_value(v: Any) -> str | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip()
    if s == "":
        return None
    s = s.translate(_ARABIC_DIGITS).replace(" ", "").replace(",", "")
    n = pd.to_numeric(s, errors="coerce")
    return str(int(n)) if pd.notna(n) else s

def _norm_name(v: Any) -> str | None:
    """Turn a raw name into a trimmed string or None if blank/placeholder."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip()
    if s == "":
        return None
    low = s.casefold()
    if low in {"nan", "none", "null"}:
        return None
    return s

# ───────────────────────── Inventory sanitizer ──────────────────────── #
def _sanitize_inventory(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(how="all")
    if df.empty:
        raise ValueError("No data rows found.")

    # item_name: keep None for missing so we can catch it; don't cast to str first
    if "item_name" not in df.columns:
        raise ValueError("Missing required column: item_name")
    df["item_name"] = df["item_name"].apply(_norm_name)

    # Explicitly error on any blank names
    name_blank = df["item_name"].isna()
    if name_blank.any():
        bad_rows = df.index[name_blank].tolist()
        raise ValueError(f"Blank/invalid item_name in rows: {bad_rows}.")

    # Normalise category & unit (safe defaults)
    if "category" in df.columns:
        # cast then clean
        df["category"] = df["category"].astype(str).str.strip()
        df["category"] = df["category"].replace({"", "nan", "None"}, "Food")
        # fix accidental numeric categories
        df.loc[df["category"].str.match(r"^\d+$", na=False), "category"] = "Food"

    if "unit" in df.columns:
        df["unit"] = df["unit"].astype(str).str.strip()
        df["unit"] = df["unit"].replace({"", "nan", "None"}, "pcs")

    # Barcodes to consistent string-or-None
    if "item_barcode" in df.columns:
        df["item_barcode"] = df["item_barcode"].apply(_norm_barcode_value)

    # Robust numeric parse for stocks
    for col in ["initial_stock", "current_stock"]:
        df[col] = df[col].apply(_clean_number_like)
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Heal missing values:
    #  - if one present, copy to the other
    #  - if both missing, set 0 (we'll upsert using these numbers)
    init_na = df["initial_stock"].isna()
    curr_na = df["current_stock"].isna()
    df.loc[init_na & ~curr_na, "initial_stock"] = df.loc[init_na & ~curr_na, "current_stock"]
    df.loc[~init_na & curr_na, "current_stock"] = df.loc[~init_na & curr_na, "initial_stock"]
    both_na = df["initial_stock"].isna() & df["current_stock"].isna()
    df.loc[both_na, ["initial_stock", "current_stock"]] = 0

    # Optional sheet-wide sign flip
    both_negative = (df["initial_stock"] < 0) & (df["current_stock"] < 0)
    if both_negative.mean() > AUTO_FLIP_NEGATIVE_THRESHOLD:
        df[["initial_stock", "current_stock"]] = -df[["initial_stock", "current_stock"]]

    # Round → int (allow negatives)
    df["initial_stock"] = df["initial_stock"].fillna(0).round().astype(int)
    df["current_stock"] = df["current_stock"].fillna(0).round().astype(int)

    # Intra-file de-dupe by item_name: keep last occurrence
    df = df.drop_duplicates(subset=["item_name"], keep="last").reset_index(drop=True)

    return df

# ───────────────────────── Main upsert function ─────────────────────── #
@run_transaction
def upsert_dataframe(conn,
                     df: pd.DataFrame,
                     table: Literal["inventory", "purchases", "sales"]) -> None:
    df = df.dropna(how="all")
    if df.empty:
        raise ValueError("No data rows found.")

    df = _normalise_dates(df, table)

    if table in ("purchases", "sales"):
        _handle_purchases_or_sales(conn, df, table)
    else:
        clean = _sanitize_inventory(df)
        _insert_inventory(conn, clean)

# ───────────────────────── Purchases / Sales logic ──────────────────── #
def _row_sign(table: str, bill_type: str) -> int:
    bt = (bill_type or "").strip().lower()
    is_return = "return" in bt  # Purchase Return / Sales Return
    base = 1 if table == "purchases" else -1
    return -base if is_return else base

def _handle_purchases_or_sales(conn, df: pd.DataFrame, table: str) -> None:
    inv = fetch_dataframe("SELECT item_id, item_name, item_barcode FROM inventory")

    name2id: Dict[str, int] = inv.set_index("item_name")["item_id"].to_dict()
    inv = inv.copy()
    inv["__bc_norm"] = inv["item_barcode"].apply(_norm_barcode_value)
    code2id: Dict[str, int] = (
        inv.dropna(subset=["__bc_norm"]).set_index("__bc_norm")["item_id"].to_dict()
    )

    inserted_qty: Dict[int, int] = {}

    for idx, row in df.iterrows():
        sign = _row_sign(table, row.get("bill_type"))
        df.at[idx, "__sign"] = sign

        item_id = row.get("item_id")
        if not pd.isna(item_id):
            continue

        candidate = name2id.get(row.get("item_name"))
        if candidate is None:
            bc_key = _norm_barcode_value(row.get("item_barcode"))
            candidate = code2id.get(bc_key)

        if candidate is None:
            # Create only on positive Purchase Invoice
            if table == "purchases" and sign > 0:
                qty = int(pd.to_numeric(row.get("quantity"), errors="coerce") or 0)
                ins = text(
                    "INSERT INTO inventory "
                    "(item_name, item_barcode, category, initial_stock, current_stock, unit) "
                    "VALUES (:name, :code, 'Food', :qty, :qty, 'pcs') "
                    "RETURNING item_id;"
                )
                new_id = conn.execute(
                    ins,
                    {"name": row.get("item_name"),
                     "code": _norm_barcode_value(row.get("item_barcode")),
                     "qty": qty}
                ).scalar_one()
                item_id = new_id
                inserted_qty[item_id] = qty
                name2id[row.get("item_name")] = item_id
                code = _norm_barcode_value(row.get("item_barcode"))
                if code:
                    code2id[code] = item_id
            else:
                raise ValueError(
                    f"Row {idx}: unknown item for {table} with bill_type='{row.get('bill_type')}'. "
                    f"Provide an existing item_name/barcode (or use a positive Purchase Invoice to create)."
                )
        else:
            item_id = candidate

        df.at[idx, "item_id"] = item_id

    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0).round().astype(int)
    df["_delta"] = df["quantity"] * df["__sign"]
    deltas = df.groupby("item_id")["_delta"].sum()

    drop_cols = [c for c in ("item_name", "item_barcode", "__sign", "_delta") if c in df.columns]
    _insert_dataframe(conn, df.drop(columns=drop_cols), table)

    for item_id, delta in deltas.items():
        if table == "purchases" and item_id in inserted_qty:
            delta -= inserted_qty[item_id]
        if delta == 0:
            continue
        conn.execute(
            text("UPDATE inventory SET current_stock = current_stock + :d WHERE item_id = :i;"),
            {"d": int(delta), "i": int(item_id)},
        )

# ───────────────────────── INSERT helpers ───────────────────────────── #
def _insert_inventory(conn, df: pd.DataFrame) -> None:
    """
    Inventory UPSERT by item_name. If the record exists, update metadata and stock.
    Barcode is updated only when provided (keeps existing otherwise).
    """
    cols = ["item_name", "item_barcode", "category", "unit", "initial_stock", "current_stock"]
    placeholders = ", ".join([f":{c}" for c in cols])

    if UPSERT_EXISTING_INVENTORY:
        sql = text(f"""
            INSERT INTO inventory ({', '.join(cols)}) VALUES ({placeholders})
            ON CONFLICT ON CONSTRAINT inventory_item_name_key DO UPDATE
            SET
                item_barcode = COALESCE(EXCLUDED.item_barcode, inventory.item_barcode),
                category     = EXCLUDED.category,
                unit         = EXCLUDED.unit,
                initial_stock = EXCLUDED.initial_stock,
                current_stock = EXCLUDED.current_stock;
        """)
    else:
        sql = text(f"INSERT INTO inventory ({', '.join(cols)}) VALUES ({placeholders});")

    payload = df.where(pd.notnull(df), None).to_dict(orient="records")
    conn.execute(sql, payload)

def _insert_dataframe(conn, df: pd.DataFrame, table: str) -> None:
    cols = list(df.columns)
    placeholders = ", ".join([f":{c}" for c in cols])
    sql = text(f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders});")
    payload = df.where(pd.notnull(df), None).to_dict(orient="records")
    conn.execute(sql, payload)
