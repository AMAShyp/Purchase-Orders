# sale_upload.py – Unified Purchases & Sales upload (input/output quantity + repairs summary)
import time
from io import StringIO, BytesIO
from typing import List

import pandas as pd
import streamlit as st

try:
    from .upload_handler import bulk_insert_unified_txns, get_row_count
except Exception:
    from upload_handler import bulk_insert_unified_txns, get_row_count


UNIFIED_TEMPLATE_COLS = [
    "bill_type",      # routes the row
    "txn_date",       # → purchase_date / sale_date
    "item_name",
    "item_barcode",
    "input_quantity",   # used when stock flows IN
    "output_quantity",  # used when stock flows OUT
    "unit_price",       # → purchase_price / sale_price (optional)
    # Optional for first-time creation on positive purchases:
    # "category", "unit"
]

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
            "Headers: `bill_type, txn_date, item_name, item_barcode, input_quantity, output_quantity, unit_price`.\n"
            "Input qty used for **purchase invoice direct / purchase invoice / sales return invoice**; "
            "Output qty for **sales invoice / purchasing return invoice / purchase return invoice**.\n"
            "We strictly pair `item_name`+`item_barcode` and will **repair** missing/mismatched pairs when safe. "
            "Unknown items are auto-created only for positive purchases."
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
        before_i = get_row_count("inventory")
        st.info(f"🔎 Before → purchases: {before_p:,}, sales: {before_s:,}, inventory rows: {before_i:,}")
    except Exception as e:
        st.warning(f"Pre-insert counts failed: {e}")
        before_p = before_s = before_i = None

    # Read & preview
    df = _read_file(file)
    st.write(f"Rows: **{df.shape[0]:,}**, Columns: **{df.shape[1]}**")
    st.dataframe(df.head(200), use_container_width=True, height=320)
    st.caption("Preview shows up to 200 rows. Upload will insert all rows.")

    if st.button("✅ Bulk insert & update inventory", type="primary"):
        with st.status("Processing unified upload…", expanded=True) as status:
            try:
                t0 = time.perf_counter()
                result = bulk_insert_unified_txns(df=df)
                total_ms = (time.perf_counter() - t0) * 1000

                status.update(label="Commit successful ✅", state="complete")
                st.success("Inserted into purchases/sales, updated inventory, and applied safe repairs where needed.")

                repairs = result.get("repairs") or {}
                st.info(
                    f"🛠️ Repairs applied → purchases: {repairs.get('purchases', 0)}, "
                    f"sales: {repairs.get('sales', 0)}, total: **{repairs.get('total', 0)}**"
                )

                st.write({
                    "purchases": result.get("purchases"),
                    "sales": result.get("sales"),
                    "inventory_update": result.get("inventory_update"),
                    "total_ms (end-to-end)": round(total_ms, 1),
                })

                # After counts
                try:
                    after_p = get_row_count("purchases")
                    after_s = get_row_count("sales")
                    after_i = get_row_count("inventory")
                    if None not in (before_p, before_s, before_i):
                        st.info(
                            f"📊 After → purchases: {after_p:,} (+{after_p - before_p:,}), "
                            f"sales: {after_s:,} (+{after_s - before_s:,}), "
                            f"inventory rows: {after_i:,} (+{after_i - before_i:,})"
                        )
                    else:
                        st.info(f"📊 After → purchases: {after_p:,}, sales: {after_s:,}, inventory rows: {after_i:,}")
                except Exception as e:
                    st.warning(f"Post-insert counts failed: {e}")

            except Exception as exc:
                status.update(label="Commit failed ❌", state="error")
                st.error(f"Upload failed → {exc}")

    st.divider()

def page():
    st.title("🧾 Unified Purchases & Sales Upload (input/output quantity + repairs)")
    st.caption("Strict (name+barcode) pairing with safe auto-repair. No item_id in Excel required.")
    _section_unified()

if __name__ == "__main__":
    st.set_page_config(page_title="Unified Sales & Purchases Upload", page_icon="🧾", layout="wide")
    page()
