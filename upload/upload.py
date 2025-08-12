# upload.py – FAST v1.2 (subset headers + numeric coercion)
import time
from io import StringIO, BytesIO
from typing import List

import pandas as pd
import streamlit as st

try:
    from .upload_handler import bulk_insert_exact_headers, get_row_count
except Exception:
    from upload_handler import bulk_insert_exact_headers, get_row_count


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

TABLES = {
    "Inventory": "inventory",
    "Purchases": "purchases",
    "Sales":     "sales",
}

def _section(label: str, table: str, template_cols: List[str]):
    st.subheader(label)

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
            "CSV is fastest. We insert **as-is** when headers match table columns (subset allowed). "
            "Numeric columns are auto-cleaned (commas removed; integers rounded). "
            "Do **not** include auto/default columns like `item_id`, `created_at`, `updated_at`."
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
        before = get_row_count(table)
        st.info(f"🔎 Before insert → `{table}` rows: {before:,}")
    except Exception as e:
        st.warning(f"Pre-insert row count failed: {e}")
        before = None

    # Load
    df = _read_file(file)
    st.write(f"Rows: **{df.shape[0]:,}**, Columns: **{df.shape[1]}**")
    st.dataframe(df.head(200), use_container_width=True, height=320)
    st.caption("Preview shows up to 200 rows. Upload will insert all rows.")

    if st.button(f"✅ Bulk insert into `{table}`", key=f"btn_{table}", type="primary"):
        with st.status("Running bulk insert…", expanded=True) as status:
            try:
                t0 = time.perf_counter()
                result = bulk_insert_exact_headers(df=df, table=table)
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

def page():
    st.title("⚡ Bulk Upload (Headers‑Match, Ultra‑Fast)")
    st.caption("Provide only the columns you want to insert. We skip auto/default columns.")

    # Inventory: exclude item_id, created_at, updated_at
    _section(
        "Inventory Items",
        TABLES["Inventory"],
        ["item_name","item_barcode","category","unit","initial_stock","current_stock"],
    )

    # (Keep these if you also bulk load purchases/sales; otherwise remove)
    _section(
        "Daily Purchases",
        TABLES["Purchases"],
        ["bill_type","purchase_date","item_id","item_name","item_barcode","quantity","purchase_price"],
    )
    _section(
        "Daily Sales",
        TABLES["Sales"],
        ["bill_type","sale_date","item_id","item_name","item_barcode","quantity","sale_price"],
    )

if __name__ == "__main__":
    st.set_page_config(page_title="Upload (FAST)", page_icon="⚡", layout="wide")
    page()
