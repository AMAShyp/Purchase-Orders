# sale_upload.py – FAST uploads for Purchases & Sales (headers-subset, numeric coercion)
import time
from io import StringIO, BytesIO
from typing import List

import pandas as pd
import streamlit as st

# Prefer the explicit helpers; fallback to generic if needed
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


# ───────────────────────── file I/O ───────────────────────── #

def _read_file(file) -> pd.DataFrame:
    """Read CSV or Excel without value checks (fast path)."""
    t0 = time.perf_counter()
    if file.type == "text/csv":
        # CSV is fastest; try pandas 2.x pyarrow backend if available
        try:
            df = pd.read_csv(StringIO(file.getvalue().decode("utf-8")), dtype_backend="pyarrow")
        except TypeError:
            df = pd.read_csv(StringIO(file.getvalue().decode("utf-8")))
    else:
        df = pd.read_excel(BytesIO(file.getvalue()))
    st.caption(f"📥 File read in {(time.perf_counter() - t0)*1000:,.0f} ms")
    return df


def _make_template(columns: List[str]) -> bytes:
    """Small helper to export an Excel template for convenience."""
    buf = BytesIO()
    pd.DataFrame(columns=columns).to_excel(buf, index=False, engine="openpyxl")
    buf.seek(0)
    return buf.read()


# ───────────────────────── templates (omit IDs/timestamps) ───────────────────────── #

PURCHASES_TEMPLATE_COLS = [
    # Subset allowed; DB will fill any omitted defaults (IDs/timestamps)
    "bill_type",
    "purchase_date",
    "item_id",
    "item_name",
    "item_barcode",
    "quantity",
    "purchase_price",
]

SALES_TEMPLATE_COLS = [
    "bill_type",
    "sale_date",
    "item_id",
    "item_name",
    "item_barcode",
    "quantity",
    "sale_price",
]


# ───────────────────────── shared UI section ───────────────────────── #

def _upload_section(
    *,
    label: str,
    table: str,                # "purchases" | "sales"
    template_cols: List[str],
    call_fn,                   # bulk_insert_purchases or bulk_insert_sales
    key_prefix: str,
):
    st.subheader(label)

    c1, c2 = st.columns([1, 3], vertical_alignment="center")
    with c1:
        st.download_button(
            "📄 Excel template (optional)",
            data=_make_template(template_cols),
            file_name=f"{table}_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"tmpl_{key_prefix}",
        )
    with c2:
        st.caption(
            "CSV is fastest. We insert **as-is** when your headers match table columns (subset allowed). "
            "Numeric fields (e.g., `quantity`, `*_price`) are auto-cleaned: commas removed; integers rounded. "
            "Leave out IDs/timestamps if your DB fills them by default."
        )

    file = st.file_uploader(
        f"Choose CSV/Excel for **{table}**",
        key=f"upl_{key_prefix}",
        type=["csv", "xlsx", "xls"],
    )
    if file is None:
        st.info("No file selected yet.")
        st.divider()
        return

    # Before count
    try:
        before = get_row_count(table)
        st.info(f"🔎 Before insert → `{table}` rows: {before:,}")
    except Exception as e:
        st.warning(f"Pre-insert row count failed: {e}")
        before = None

    # Load & preview
    df = _read_file(file)
    st.write(f"Rows: **{df.shape[0]:,}**, Columns: **{df.shape[1]}**")
    st.dataframe(df.head(200), use_container_width=True, height=320)
    st.caption("Preview shows up to 200 rows. Upload will insert all rows.")

    if st.button(f"✅ Bulk insert into `{table}`", key=f"btn_{key_prefix}", type="primary"):
        with st.status("Running bulk insert…", expanded=True) as status:
            try:
                t0 = time.perf_counter()
                # The handler decorator injects `conn`; we pass keyword args only
                result = call_fn(df=df)
                total_ms = (time.perf_counter() - t0) * 1000

                status.update(label="Commit successful ✅", state="complete")
                st.success(
                    f"Inserted **{result['rows']:,}** rows into `{table}` "
                    f"(COPY used: **{result['used_copy']}**)"
                )

                timings = result["timings"]
                st.write({
                    "used_columns": result.get("used_columns"),
                    "fetch_columns_ms": round(timings.get("fetch_columns_ms", 0), 1),
                    "align_columns_ms": round(timings.get("align_columns_ms", 0), 1),
                    "numeric_coercion_ms": round(timings.get("numeric_coercion_ms", 0), 1),
                    "copy_or_insert_ms": round(timings.get("copy_or_insert_ms", 0), 1),
                    "executemany_ms": round(timings.get("executemany_ms", 0), 1),
                    "total_ms (handler)": round(timings.get("total_ms", 0), 1),
                    "total_ms (end-to-end)": round(total_ms, 1),
                })

                try:
                    after = get_row_count(table)
                    if before is not None:
                        st.info(f"📊 After insert → `{table}` rows: {after:,} "
                                f"(+{after - before:,} new)")
                    else:
                        st.info(f"📊 After insert → `{table}` rows: {after:,}")
                except Exception as e:
                    st.warning(f"Post-insert row count failed: {e}")

            except Exception as exc:
                status.update(label="Commit failed ❌", state="error")
                st.error(f"Upload failed → {exc}")

    st.divider()


# ───────────────────────── page entry ───────────────────────── #

def page():
    st.title("🧾 Bulk Upload — Purchases & Sales (FAST)")
    st.caption("Provide only the columns you want to insert. IDs/timestamps can be omitted and will be filled by the DB.")

    _upload_section(
        label="Daily Purchases",
        table="purchases",
        template_cols=PURCHASES_TEMPLATE_COLS,
        call_fn=bulk_insert_purchases,
        key_prefix="purchases",
    )

    _upload_section(
        label="Daily Sales",
        table="sales",
        template_cols=SALES_TEMPLATE_COLS,
        call_fn=bulk_insert_sales,
        key_prefix="sales",
    )


if __name__ == "__main__":
    st.set_page_config(page_title="Sales & Purchases Upload (FAST)", page_icon="🧾", layout="wide")
    page()
