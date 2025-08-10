import re
from decimal import Decimal, getcontext
from typing import Dict, Any, List

import pandas as pd
from sqlalchemy import MetaData, Table, select, func
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert as pg_insert

# 👇 adjust to your project
from .db import engine  # make sure this imports a SQLAlchemy Engine

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

# ---------- internal helpers ----------
def _as_int(s) -> int:
    try:
        return int(float(str(s).strip()))  # handles "5", "5.0", "5.00"
    except Exception:
        return 0

def _normalize_barcode(v) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return None
    # Excel scientific notation -> full digits
    if re.search(r"[eE][+-]?\d+$", s):
        getcontext().prec = 50
        s = format(Decimal(s), "f").split(".")[0]
    # keep only digits; preserve leading zeros by not casting to int
    s = re.sub(r"\D", "", s)
    return s or None

def _clean_inventory_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # ensure required columns exist (some may be dtype=object from Excel)
    for c in REQUIRED["inventory"]:
        if c not in df:
            df[c] = None

    df["item_name"] = df["item_name"].astype(str).str.strip()
    df["item_barcode"] = df["item_barcode"].map(_normalize_barcode)
    df["category"] = df["category"].astype(str).str.strip()
    df["unit"] = df["unit"].astype(str).str.strip()
    df["initial_stock"] = df["initial_stock"].map(_as_int)
    df["current_stock"] = df["current_stock"].map(_as_int)

    # optional: drop total-empty rows (no name and no barcode)
    df = df[~(df["item_name"].str.strip().eq("") & df["item_barcode"].isna())].reset_index(drop=True)
    return df

# ---------- core upserts ----------
def _upsert_inventory(df: pd.DataFrame) -> Dict[str, Any]:
    df = _clean_inventory_df(df)

    meta = MetaData()
    inv = Table("inventory", meta, autoload_with=engine)

    with Session(engine) as s:
        # existing barcodes & names once
        existing_barcodes = {
            v for (v,) in s.execute(select(inv.c.item_barcode)).all() if v is not None
        }
        existing_names = {
            v.strip() for (v,) in s.execute(select(inv.c.item_name)).all() if v is not None
        }

        to_write: List[Dict[str, Any]] = []
        skipped_rows: List[Dict[str, Any]] = []
        seen_in_batch: set[str] = set()

        will_insert = 0
        will_update = 0

        for rec in df.to_dict(orient="records"):
            bc = rec.get("item_barcode")
            name = (rec.get("item_name") or "").strip()

            # Skip duplicate barcodes (either already in DB or repeated in this same file)
            if bc and (bc in existing_barcodes or bc in seen_in_batch):
                skipped_rows.append(rec)
                continue

            if bc:
                seen_in_batch.add(bc)

            if name in existing_names:
                will_update += 1
            else:
                will_insert += 1

            to_write.append(rec)

        written = 0
        if to_write:
            stmt = (
                pg_insert(inv)
                .values(to_write)
                .on_conflict_do_update(
                    constraint="inventory_item_name_key",  # upsert by item_name
                    set_={
                        "item_barcode": func.coalesce(
                            pg_insert(inv).excluded.item_barcode,
                            inv.c.item_barcode,
                        ),
                        "category": pg_insert(inv).excluded.category,
                        "unit": pg_insert(inv).excluded.unit,
                        "initial_stock": pg_insert(inv).excluded.initial_stock,
                        "current_stock": pg_insert(inv).excluded.current_stock,
                    },
                )
            )
            s.execute(stmt)
            s.commit()
            written = len(to_write)

        return {
            "written": written,  # inserted + updated
            "will_insert": will_insert,
            "will_update": will_update,
            "skipped_barcode": len(skipped_rows),
            "skipped_rows": skipped_rows,
        }

def _upsert_passthrough(df: pd.DataFrame, table: str) -> Dict[str, Any]:
    """
    Keep your existing logic here if you already had purchases/sales upserts.
    This simple example just writes the rows as-is (no barcode handling).
    """
    meta = MetaData()
    tab = Table(table, meta, autoload_with=engine)

    # light cleaning
    df = df.copy()
    recs = df.to_dict(orient="records")

    with Session(engine) as s:
        # if you already had ON CONFLICT logic for these tables, plug it here
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
