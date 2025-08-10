"""
upload_handler.py – v9
• Inventory upload:
  - robust numeric parsing (commas, Arabic digits, spaces)
  - heals missing initial/current stock (copy the other; if both missing → 0)
  - allows negative values (no hard stop) and auto-flips sign if >80% rows have both negative
  - barcode normalization
• Purchases/Sales:
  - bill_type-aware stock updates
  - date normalization
  - new-item creation only on Purchase Invoice (no double-count)
  - tolerant barcode lookups (normalize numeric vs string)
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

def check_columns(df: pd.DataFrame,
                  table: Literal["inventory", "purchases", "sales"]) -> None:
    required = BASE_REQ[table].copy()
    if table in ("purchases", "sales"):
        # need at least one way to resolve an item
        if not ({"item_name", "item_barcode", "item_id"} & set(df.columns)):
            raise ValueError("Need item_name, item_barcode or item_id column.")
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")

# ───────────────────────── Dates ─────────────────────────────────────── #
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
    s = re.sub(r"[^0-9\.\-]", "", s)  # keep digits, dot, minus
    if s.count(".") > 1:
        first, *rest = s.split(".")
        s = first + "." + "".join(rest)
    return s

def _norm_barcode_value(v: Any) -> str | None:
    """Return a normalised string barcode (e.g., '6212558011153') or None."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip()
    if s == "":
        return None
    s = s.translate(_ARABIC_DIGITS).replace(" ", "").replace(",", "")
    n = pd.to_numeric(s, errors="coerce")
    return str(int(n)) if pd.notna(n) else s

# ───────────────────────── Inventory sanitizer ──────────────────────── #
def _sanitize_inventory(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(how="all")
    if df.empty:
        raise ValueError("No data rows found.")

    # Ensure text columns are strings
    for col in ["item_name", "category", "unit"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Defaults / clean-ups
    if "category" in df.columns:
        df["category"] = df["category"].replace({"", "nan", "None"}, "Food")
        # fix accidental numeric categories
        df.loc[df["category"].str.match(r"^\d+$", na=False), "category"] = "Food"
    if "unit" in df.columns:
        df["unit"] = df["unit"].replace({"", "nan", "None"}, "pcs")

    # item_name must not be blank
    name_blank = df["item_name"].isna() | (df["item_name"].str.strip() == "")
    if name_blank.any():
        bad_rows = df.index[name_blank].tolist()
        raise ValueError(f"Blank item_name in rows: {bad_rows}.")

    # Normalize barcodes to a consistent string-or-None
    if "item_barcode" in df.columns:
        df["item_barcode"] = df["item_barcode"].apply(_norm_barcode_value)

    # Robust numeric parse for stocks
    for col in ["initial_stock", "current_stock"]:
        df[col] = df[col].apply(_clean_number_like)
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Heal missing values
    init_na = df["initial_stock"].isna()
    curr_na = df["current_stock"].isna()
    df.loc[init_na & ~curr_na, "initial_stock"] = df.loc[init_na & ~curr_na, "current_stock"]
    df.loc[~init_na & curr_na, "current_stock"] = df.loc[~init_na & curr_na, "initial_stock"]
    both_na = df["initial_stock"].isna() & df["current_stock"].isna()
    df.loc[both_na, ["initial_stock", "current_stock"]] = 0

    # If most rows have BOTH quantities negative, assume the whole sheet is sign-flipped → flip.
    both_negative = (df["initial_stock"] < 0) & (df["current_stock"] < 0)
    if both_negative.mean() > 0.80:  # 80% threshold
        df[["initial_stock", "current_stock"]] = -df[["initial_stock", "current_stock"]]

    # Final type: integers (allow negatives), fill any stray NaNs with 0 before cast
    df["initial_stock"] = df["initial_stock"].fillna(0).round().astype(int)
    df["current_stock"] = df["current_stock"].fillna(0).round().astype(int)

    # NOTE: We no longer hard-stop on negatives. If you want to forbid them, tell me and
    # I'll add a toggle in the UI or a strict mode flag here.
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
        _insert_dataframe(conn, clean, "inventory")

# ───────────────────────── Purchases / Sales logic ──────────────────── #
def _row_sign(table: str, bill_type: str) -> int:
    """
    +1 / −1 for stock delta based on bill_type and table.
    Purchases:  base +1; 'return' flips to −1.
    Sales:      base −1; 'return' flips to +1.
    """
    bt = (bill_type or "").strip().lower()
    is_return = "return" in bt
    base = 1 if table == "purchases" else -1
    return -base if is_return else base

def _handle_purchases_or_sales(conn, df: pd.DataFrame, table: str) -> None:
    inv = fetch_dataframe("SELECT item_id, item_name, item_barcode FROM inventory")

    # Build tolerant lookup maps: by name, and by normalised barcode-string
    name2id: Dict[str, int] = inv.set_index("item_name")["item_id"].to_dict()

    inv = inv.copy()
    inv["__bc_norm"] = inv["item_barcode"].apply(_norm_barcode_value)
    code2id: Dict[str, int] = (
        inv.dropna(subset=["__bc_norm"]).set_index("__bc_norm")["item_id"].to_dict()
    )

    inserted_qty: Dict[int, int] = {}

    # Resolve item_id and compute sign
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
            # Not found in existing inventory
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
                # extend lookups for further rows in same file
                name2id[row.get("item_name")] = item_id
                code = _norm_barcode_value(row.get("item_barcode"))
                if code:
                    code2id[code] = item_id
            else:
                raise ValueError(
                    f"Row {idx}: unknown item for {table} with bill_type='{row.get('bill_type')}'. "
                    f"Provide item_name or item_barcode that exists in inventory (or use a Purchase Invoice to create)."
                )
        else:
            item_id = candidate

        df.at[idx, "item_id"] = item_id

    # Quantity & deltas
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0).round().astype(int)
    df["_delta"] = df["quantity"] * df["__sign"]

    # Compute aggregate deltas BEFORE dropping helper columns
    deltas = df.groupby("item_id")["_delta"].sum()

    # Insert rows (drop helper columns)
    drop_cols = [c for c in ("item_name", "item_barcode", "__sign", "_delta") if c in df.columns]
    _insert_dataframe(conn, df.drop(columns=drop_cols), table)

    # Adjust inventory current_stock
    for item_id, delta in deltas.items():
        if table == "purchases" and item_id in inserted_qty:
            # We already created the item with qty as initial+current; don't double-count
            delta -= inserted_qty[item_id]
        if delta == 0:
            continue
        conn.execute(
            text("UPDATE inventory SET current_stock = current_stock + :d WHERE item_id = :i;"),
            {"d": int(delta), "i": int(item_id)},
        )

# ───────────────────────── Generic INSERT helper ────────────────────── #
def _insert_dataframe(conn, df: pd.DataFrame, table: str) -> None:
    cols = list(df.columns)
    placeholders = ", ".join([f":{c}" for c in cols])
    sql = text(f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders});")
    payload = df.where(pd.notnull(df), None).to_dict(orient="records")
    conn.execute(sql, payload)
