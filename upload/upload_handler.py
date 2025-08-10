import re
from decimal import Decimal, getcontext
from typing import Dict, Any, List

import pandas as pd
from sqlalchemy import MetaData, Table, select, func
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert as pg_insert

# 👇 adapt this import to your app
from .db import engine

REQUIRED = {
    "inventory": ["item_name", "item_barcode", "category", "unit", "initial_stock", "current_stock"],
    "purchases": ["bill_type", "purchase_date", "item_name", "item_barcode", "quantity", "purchase_price"],
    "sales": ["bill_type", "sale_date", "item_name", "item_barcode", "quantity", "sale_price"],
}

# ---------- public helpers ----------
def check_columns(df: pd.DataFrame, table: str) -> None:
    need = REQUIRED[table]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for {table}: {', '.join(missing)}")

# ---------- cleaning helpers ----------
_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

def _digits_only(s: str) -> str:
    return re.sub(r"\D", "", s)

def _normalize_barcode(v) -> str | None:
    """Return normalized ASCII-digit barcode or None for blanks."""
    if v is None:
        return None
    s = str(v).strip().translate(_ARABIC_DIGITS)
    if s == "" or s.lower() in {"nan", "none"}:
        return None
    # Fix scientific notation "8.69057E+12" from Excel
    if re.search(r"[eE][+-]?\d+$", s):
        getcontext().prec = 50
        s = format(Decimal(s), "f")
    s = _digits_only(s)
    return s or None

def _as_int(x) -> int:
    try:
        # handles "5", "5.0", "", None
        return int(float(str(x).strip()))
    except Exception:
        return 0

def _clean_inventory_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # make sure all expected columns exist
    for c in REQUIRED["inventory"]:
        if c not in df.columns:
            df[c] = None

    df["item_name"] = df["item_name"].astype(str).str.strip()
    df["item_barcode"] = df["item_barcode"].map(_normalize_barcode)
    df["category"] = df["category"].astype(str).str.strip()
    df["unit"] = df["unit"].astype(str).str.strip()
    df["initial_stock"] = df["initial_stock"].map(_as_int)
    df["current_stock"] = df["current_stock"].map(_as_int)

    # Drop total-empty rows (no name AND no barcode)
    df = df[~(df["item_name"].eq("") & df["item_barcode"].isna())].reset_index(drop=True)
    return df

# ---------- core upserts ----------
def _upsert_inventory(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Upsert by item_name, while skipping any row whose barcode is already
    used by a different item (either in DB or within the same file).
    This prevents UNIQUE(item_barcode) violations.
    """
    df = _clean_inventory_df(df)

    meta = MetaData()
    inv = Table("inventory", meta, autoload_with=engine)

    with Session(engine) as s:
        # Build a mapping: barcode -> item_name from DB, so we know ownership
        db_rows = s.execute(select(inv.c.item_name, inv.c.item_barcode)).all()
        db_bc_to_name: Dict[str, str] = {}
        db_names: set[str] = set()
        for name, bc in db_rows:
            if name:
                db_names.add(str(name).strip())
            if bc:
                db_bc_to_name[str(bc)] = str(name).strip() if name else ""

        # Prepare batch screening
        batch_bc_to_name: Dict[str, str] = {}
        pending_by_name: Dict[str, Dict[str, Any]] = {}
        skipped_rows: List[Dict[str, Any]] = []

        for rec in df.to_dict(orient="records"):
            name = (rec.get("item_name") or "").strip()
            bc = rec.get("item_barcode")

            # if barcode already used by *another* item in DB → skip
            if bc and (bc in db_bc_to_name) and (db_bc_to_name[bc] != name):
                rec["_skip_reason"] = "barcode already used by another item in DB"
                skipped_rows.append(rec)
                continue

            # if barcode duplicated in this same file for different item → skip
            if bc and (bc in batch_bc_to_name) and (batch_bc_to_name[bc] != name):
                rec["_skip_reason"] = "barcode duplicated in file for another item"
                skipped_rows.append(rec)
                continue

            if bc:
                batch_bc_to_name[bc] = name

            # keep the *last* occurrence for each name (last wins)
            pending_by_name[name] = rec

        to_write = list(pending_by_name.values())
        written = 0

        if to_write:
            stmt = pg_insert(inv).values(to_write)
            stmt = stmt.on_conflict_do_update(
                constraint="inventory_item_name_key",  # upsert by item_name
                set_={
                    # Keep existing barcode if the new one is NULL
                    "item_barcode": func.coalesce(stmt.excluded.item_barcode, inv.c.item_barcode),
                    "category": stmt.excluded.category,
                    "unit": stmt.excluded.unit,
                    "initial_stock": stmt.excluded.initial_stock,
                    "current_stock": stmt.excluded.current_stock,
                },
            )
            s.execute(stmt)
            s.commit()
            written = len(to_write)

        return {
            "written": written,                         # inserted + updated
            "skipped_barcode": len(skipped_rows),       # count of skipped for barcode conflicts
            "skipped_rows": skipped_rows,               # the actual rows (for download)
        }

def _upsert_passthrough(df: pd.DataFrame, table: str) -> Dict[str, Any]:
    """Simple write-through for purchases/sales (keep your own logic if needed)."""
    meta = MetaData()
    tab = Table(table, meta, autoload_with=engine)

    df = df.copy()
    recs = df.to_dict(orient="records")

    with Session(engine) as s:
        stmt = pg_insert(tab).values(recs)
        s.execute(stmt)
        s.commit()

    return {"written": len(recs)}

# ---------- public entry ----------
def upsert_dataframe(df: pd.DataFrame, table: str) -> Dict[str, Any]:
    if table == "inventory":
        return _upsert_inventory(df)
    else:
        return _upsert_passthrough(df, table)
