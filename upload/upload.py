# inventory_upload.py
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation, getcontext

import pandas as pd
from sqlalchemy import create_engine, MetaData, Table, select, func
from sqlalchemy.dialects.postgresql import insert as pg_insert


# ───────────── Config ─────────────
DB_URL = "postgresql+psycopg2://USER:PASS@HOST:PORT/DBNAME"  # ← update this
CHUNK_SIZE = 5000


# ───────────── Helpers ─────────────
def normalize_barcode(v):
    """Return a clean string barcode or None."""
    if v is None:
        return None
    s = str(v).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return None
    # Fix Excel scientific notation like 8.69057E+12
    try:
        if "e" in s.lower():
            getcontext().prec = 40
            s = format(Decimal(s), "f").split(".")[0]
    except InvalidOperation:
        pass
    # Keep digits only (drop spaces, punctuation)
    s = re.sub(r"\D", "", s)
    return s or None


def coerce_nonneg_int(v, default=0):
    """Turn v into a non-negative int; return None if negative/invalid."""
    try:
        n = pd.to_numeric([v], errors="coerce")[0]
    except Exception:
        n = None
    if n is None or pd.isna(n):
        n = default
    n = int(Decimal(str(n)).to_integral_value(rounding="ROUND_DOWN"))
    if n < 0:
        return None
    return n


def load_file(path):
    """Read CSV/XLS/XLSX and return a cleaned DataFrame + count of bad-stock rows."""
    dtype = {"item_barcode": str}
    ext = path.lower().rsplit(".", 1)[-1]
    if ext in ("xlsx", "xls"):
        df = pd.read_excel(path, dtype=dtype)
    elif ext == "csv":
        df = pd.read_csv(path, dtype=dtype)
    else:
        raise ValueError("Only CSV, XLSX, or XLS are supported.")

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]
    required = ["item_name", "item_barcode", "category", "unit", "initial_stock", "current_stock"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Basic cleaning
    df["item_name"] = df["item_name"].astype(str).str.strip()
    df["item_barcode"] = df["item_barcode"].map(normalize_barcode)
    df["category"] = df["category"].astype(str).str.strip().replace({"nan": ""})
    df["unit"] = df["unit"].astype(str).str.strip().replace({"nan": "pcs"}).fillna("pcs")

    # Stocks (non-negative ints)
    df["initial_stock"] = df["initial_stock"].map(coerce_nonneg_int)
    df["current_stock"] = df["current_stock"].map(coerce_nonneg_int)
    bad = df["initial_stock"].isna() | df["current_stock"].isna()
    skipped_bad_stock = int(bad.sum())
    df = df[~bad].copy()

    return df, skipped_bad_stock


# ───────────── Main uploader ─────────────
def upload_inventory_file(path, *, db_url=DB_URL, chunk_size=CHUNK_SIZE):
    engine = create_engine(db_url, future=True)
    md = MetaData()
    inventory = Table("inventory", md, autoload_with=engine)

    df, skipped_bad_stock = load_file(path)

    with engine.begin() as conn:
        # Snapshot existing names and barcodes once
        rows = conn.execute(select(inventory.c.item_name, inventory.c.item_barcode)).all()
        name_to_barcode = {n: b for n, b in rows}
        existing_barcodes = {b for _, b in rows if b}

        kept_rows = []
        seen_batch_barcodes = set()
        skipped_dupe_barcode = 0

        for r in df.itertuples(index=False):
            row = {
                "item_name": r.item_name.strip(),
                "item_barcode": normalize_barcode(r.item_barcode),
                "category": r.category if r.category and str(r.category).lower() != "nan" else None,
                "unit": (r.unit or "pcs"),
                "initial_stock": int(r.initial_stock),
                "current_stock": int(r.current_stock),
            }

            name = row["item_name"]
            bc = row["item_barcode"]

            if name in name_to_barcode:
                # Existing product (upsert by name).
                # If incoming barcode belongs to someone else, don't change it.
                existing_bc_for_name = name_to_barcode[name]
                if bc and bc != existing_bc_for_name and bc in existing_barcodes:
                    row["item_barcode"] = None  # avoid unique violation on barcode
                kept_rows.append(row)
                continue

            # New product → skip if barcode already exists (DB or earlier in this batch)
            if bc and (bc in existing_barcodes or bc in seen_batch_barcodes):
                skipped_dupe_barcode += 1
                continue
            if bc:
                seen_batch_barcodes.add(bc)

            kept_rows.append(row)

        # Upsert by item_name in chunks
        for i in range(0, len(kept_rows), chunk_size):
            chunk = kept_rows[i:i + chunk_size]
            if not chunk:
                continue

            ins = pg_insert(inventory).values(chunk)
            stmt = ins.on_conflict_do_update(
                constraint="inventory_item_name_key",
                set_={
                    # keep existing barcode if incoming is NULL
                    "item_barcode": func.coalesce(ins.excluded.item_barcode, inventory.c.item_barcode),
                    "category": ins.excluded.category,
                    "unit": ins.excluded.unit,
                    "initial_stock": ins.excluded.initial_stock,
                    "current_stock": ins.excluded.current_stock,
                },
            )
            conn.execute(stmt)

    return {
        "processed": int(df.shape[0]),
        "inserted_or_updated": len(kept_rows),
        "skipped_duplicate_barcodes": int(skipped_dupe_barcode),
        "skipped_bad_stock": int(skipped_bad_stock),
    }


if __name__ == "__main__":
    stats = upload_inventory_file("inventory_template.xlsx")  # ← path to your file
    print(
        f"Done. Processed {stats['processed']} rows; "
        f"inserted/updated {stats['inserted_or_updated']}. "
        f"Skipped {stats['skipped_duplicate_barcodes']} rows due to duplicate barcodes "
        f"and {stats['skipped_bad_stock']} due to invalid stock."
    )
