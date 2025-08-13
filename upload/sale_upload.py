# sale_upload.py – Purchases & Sales bulk uploads (FAST)
import time
from io import StringIO, BytesIO
from typing import List

import pandas as pd
import streamlit as st

# relative imports with fallback
try:
    from .sales_upload_handler import bulk_upload_sales_like
    from ..db_handler import fetch_dataframe
except Exception:
    from sales_upload_handler import bulk_upload_sales_like
    from db_handler import fetch_dataframe


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

def _row_count(table: str) -> int:
    df = fetch_dataframe(f"SELECT COUNT(*) AS c FROM {table};")
    return int(df["c"].iloc[0])


def _section(title: str, table: str, template_cols: List[str]):
    st.subheader(title)

    c1, c2 = st.columns([1, 3], vertical_alignment="center")
    with c1:
        st.download_button(
            "📄 Excel template (optional)",
            data=_make_template(template_cols),
            file_name=f"{table}_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"tmpl_{table}",
        )
    with c2:
        st.caption(
            "Headers can be a subset (must exist in table). "
            "If `item_id` is missing, we resolve it from `item_name`/`item_barcode`. "
            "New items are auto-created for **Purchase Invoice** only."
        )

    file = st.file_uploader(
        f"Choose CSV/Excel for **{table}**",
        key=f"upl_{table}",
        type=["csv", "xlsx", "xls"],
    )
    if file is None:
        st.info("No file selected yet.")
        st.divider()
        return

    # Before count
    try:
        before = _row_count(table)
        st.info(f"🔎 Before insert → `{table}` rows: {before:,}")
    except Exception as e:
        st.warning(f"Pre-insert row count failed: {e}")
        before = None

    df = _read_file(file)
    st.write(f"Rows: **{df.shape[0]:,}**, Columns: **{df.shape[1]}**")
    st.dataframe(df.head(200), use_container_width=True, height=320)
    st.caption("Preview shows up to 200 rows. Upload will insert all rows.")

    if st.button(f"✅ Bulk upload `{table}`", key=f"btn_{table}", type="primary"):
        with st.status("Processing…", expanded=True) as status:
            try:
                t0 = time.perf_counter()
                result = bulk_upload_sales_like(df=df, table=table)
                total_ms = (time.perf_counter() - t0) * 1000

                status.update(label="Commit successful ✅", state="complete")
                st.success(
                    f"Inserted **{result['rows']:,}** rows into `{table}` "
                    f"(COPY used: **{result['used_copy']}**)"
                )
                st.write({
                    "used_columns": result.get("used_columns"),
                    "created_items": result.get("created_items"),
                    "timings_ms": {k: round(v, 1) for k, v in result["timings"].items()},
                    "total_ms (end-to-end)": round(total_ms, 1),
                })

                try:
                    after = _row_count(table)
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


def page():
    st.title("🧾 Sales & Purchases Uploads (FAST)")
    st.caption("Bulk upload daily purchases and sales; item_id resolved automatically.")

    # Purchases expect: bill_type, purchase_date, item_id or (item_name/barcode), quantity, purchase_price
    _section(
        "Daily Purchases",
        "purchases",
        ["bill_type","purchase_date","item_id","item_name","item_barcode","quantity","purchase_price","category","unit"],
    )

    # Sales expect: bill_type, sale_date, item_id or (item_name/barcode), quantity, sale_price
    _section(
        "Daily Sales",
        "sales",
        ["bill_type","sale_date","item_id","item_name","item_barcode","quantity","sale_price"],
    )


if __name__ == "__main__":
    st.set_page_config(page_title="Sales/Purchases Upload", page_icon="🧾", layout="wide")
    page()
