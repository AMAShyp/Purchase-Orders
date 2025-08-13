# sale_upload.py – Unified template for Purchases & Sales (FAST)
import time
from io import StringIO, BytesIO
from typing import List, Tuple

import pandas as pd
import streamlit as st

# Prefer the convenience wrappers; falls back to generic if needed
try:
    from .upload_handler import (
        bulk_insert_purchases,
        bulk_insert_sales,
        get_row_count,
    )
except Exception:
    from upload_handler import (
        bulk_insert_purchases,
        bulk_insert_sales,
        get_row_count,
    )

# -------- Bill type routing (case-insensitive) --------
SALE_TYPES = {
    "sales invoice",
    "sales return invoice",
}

PURCHASE_TYPES = {
    "purchase invoice direct",
    "purchasing return invoice",
    # add aliases here if needed:
    "purchase invoice",            # optional alias
    "purchase return invoice",     # optional alias
}

UNIFIED_TEMPLATE_COLS = [
    "bill_type",     # routes the row
    "txn_date",      # will be renamed to purchase_date / sale_date
    "item_id",       # optional; can be omitted if your schema allows
    "item_name",
    "item_barcode",
    "quantity",
    "unit_price",    # will be renamed to purchase_price / sale_price
]

# ---------- file I/O ----------
def _read_file(file) -> pd.DataFrame:
    t0 = time.perf_counter()
    if file.type == "text/csv":
        try:
            df = pd.read_csv(StringIO(file.getvalue().decode("utf-8")), dtype_backend="pyarrow")
        except TypeError:
            df = pd.read_csv(StringIO(file.getvalue().decode("utf-8")))
    else:
        df = pd.read_excel(BytesIO(file.getvalue()))
    st.caption(f"📥 File read in {(time.perf_counter() - t0)*1000:,.0f} ms")
    return df

def _make_template(columns: List[str]) -> bytes:
    buf = BytesIO()
    pd.DataFrame(columns=columns).to_excel(buf, index=False, engine="openpyxl")
    buf.seek(0)
    return buf.read()

def _normalize_bill_type(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.casefold()

def _split_routes(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (purchases_df, sales_df, unknown_df)."""
    bt = _normalize_bill_type(df.get("bill_type", pd.Series([], dtype="object")))
    is_purchase = bt.isin(PURCHASE_TYPES)
    is_sale = bt.isin(SALE_TYPES)
    purchases = df[is_purchase].copy()
    sales = df[is_sale].copy()
    unknown = df[~(is_purchase | is_sale)].copy()
    return purchases, sales, unknown

def _prepare_purchases(df: pd.DataFrame) -> pd.DataFrame:
    """Map unified columns -> purchases table columns."""
    df2 = df.copy()
    # Rename shared columns
    rename_map = {
        "txn_date": "purchase_date",
        "unit_price": "purchase_price",
    }
    df2.rename(columns={c: rename_map.get(c, c) for c in df2.columns}, inplace=True)
    # Keep only columns present in file; the handler will subset to actual table columns
    return df2

def _prepare_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Map unified columns -> sales table columns."""
    df2 = df.copy()
    rename_map = {
        "txn_date": "sale_date",
        "unit_price": "sale_price",
    }
    df2.rename(columns={c: rename_map.get(c, c) for c in df2.columns}, inplace=True)
    return df2

# ---------- Section ----------
def _section_unified():
    st.subheader("Unified Purchases & Sales Upload")

    c1, c2 = st.columns([1, 3], vertical_alignment="center")
    with c1:
        st.download_button(
            "📄 Excel template (unified)",
            data=_make_template(UNIFIED_TEMPLATE_COLS),
            file_name="unified_sales_purchases_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="tmpl_unified",
        )
    with c2:
        st.caption(
            "Use one sheet with these headers: "
            "`bill_type`, `txn_date`, `item_id` (optional), `item_name`, `item_barcode`, `quantity`, `unit_price`.\n"
            "Rows are routed by `bill_type`.\n\n"
            "**Sales:** `sales invoice`, `sales return invoice`\n"
            "**Purchases:** `purchase invoice direct`, `purchasing return invoice`"
        )

    file = st.file_uploader(
        "Choose CSV/Excel for **Unified Purchases & Sales**",
        key="upl_unified",
        type=["csv", "xlsx", "xls"],
    )
    if file is None:
        st.info("No file selected yet.")
        st.divider()
        return

    # Before counts
    try:
        before_p = get_row_count("purchases")
        before_s = get_row_count("sales")
        st.info(f"🔎 Before insert → `purchases`: {before_p:,} rows, `sales`: {before_s:,} rows")
    except Exception as e:
        st.warning(f"Pre-insert counts failed: {e}")
        before_p = before_s = None

    # Read & preview
    df = _read_file(file)
    st.write(f"Rows: **{df.shape[0]:,}**, Columns: **{df.shape[1]}**")
    st.dataframe(df.head(200), use_container_width=True, height=320)
    st.caption("Preview shows up to 200 rows. Upload will insert all rows.")

    # Split
    purchases_df, sales_df, unknown_df = _split_routes(df)
    st.info(
        f"Routing result → Purchases: **{len(purchases_df):,}** rows, "
        f"Sales: **{len(sales_df):,}** rows, Unknown: **{len(unknown_df):,}** rows"
    )
    if not unknown_df.empty:
        with st.expander("⚠️ Unknown bill_type rows (won't be inserted)"):
            st.dataframe(unknown_df.head(100), use_container_width=True, height=240)
            st.caption("Tip: check spelling/case/extra spaces in `bill_type`.")

    do_upload = st.button("✅ Bulk insert routed rows", type="primary")
    if not do_upload:
        st.divider()
        return

    with st.status("Running bulk inserts…", expanded=True) as status:
        try:
            total_rows = 0
            details = {}

            # Purchases
            if not purchases_df.empty:
                t0 = time.perf_counter()
                prepared = _prepare_purchases(purchases_df)
                res_p = bulk_insert_purchases(df=prepared)
                total_ms_p = (time.perf_counter() - t0) * 1000
                total_rows += res_p["rows"]
                details["purchases"] = {
                    **res_p,
                    "end_to_end_ms": round(total_ms_p, 1),
                }
                st.success(f"Purchases: inserted {res_p['rows']:,} rows (COPY used: {res_p['used_copy']})")
            else:
                st.info("Purchases: no rows to insert.")

            # Sales
            if not sales_df.empty:
                t0 = time.perf_counter()
                prepared = _prepare_sales(sales_df)
                res_s = bulk_insert_sales(df=prepared)
                total_ms_s = (time.perf_counter() - t0) * 1000
                total_rows += res_s["rows"]
                details["sales"] = {
                    **res_s,
                    "end_to_end_ms": round(total_ms_s, 1),
                }
                st.success(f"Sales: inserted {res_s['rows']:,} rows (COPY used: {res_s['used_copy']})")
            else:
                st.info("Sales: no rows to insert.")

            status.update(label="Commit(s) successful ✅", state="complete")

            # Timings summary
            st.write("Timings / columns:", details)

            # After counts
            try:
                after_p = get_row_count("purchases")
                after_s = get_row_count("sales")
                if before_p is not None and before_s is not None:
                    st.info(
                        f"📊 After insert → `purchases`: {after_p:,} "
                        f"(+{after_p - before_p:,}), "
                        f"`sales`: {after_s:,} "
                        f"(+{after_s - before_s:,})"
                    )
                else:
                    st.info(f"📊 After insert → `purchases`: {after_p:,}, `sales`: {after_s:,}")
            except Exception as e:
                st.warning(f"Post-insert counts failed: {e}")

        except Exception as exc:
            status.update(label="Commit failed ❌", state="error")
            st.error(f"Upload failed → {exc}")

    st.divider()

# ---------- Page ----------
def page():
    st.title("🧾 Unified Purchases & Sales Upload (FAST)")
    st.caption(
        "One template for both tables. We route rows by `bill_type`, map `txn_date`/`unit_price` "
        "to the correct per-table columns, and bulk-insert with COPY."
    )
    _section_unified()

if __name__ == "__main__":
    st.set_page_config(page_title="Unified Sales & Purchases Upload", page_icon="🧾", layout="wide")
    page()
