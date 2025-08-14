# upload.py – Inventory uploader (fast COPY, stage -> merge, skip duplicates)
# - Stages into a TEMP table (all columns nullable), then merges into inventory:
#   * rows that have NULL in required (NOT NULL) destination columns are skipped and counted
#   * conflicts on any UNIQUE are skipped via ON CONFLICT DO NOTHING
# - Subset headers allowed; DB-managed columns (item_id/created_at/updated_at) are not required

from __future__ import annotations

import pandas as pd
import streamlit as st
from io import StringIO, BytesIO
from typing import List

# Robust imports (package vs flat)
try:
    from .upload_handler import bulk_insert_inventory_skip_conflicts
    try:
        from ..db_handler import fetch_dataframe
    except Exception:
        from db_handler import fetch_dataframe
except Exception:
    from upload_handler import bulk_insert_inventory_skip_conflicts
    from db_handler import fetch_dataframe


INVENTORY_TEMPLATE_COLS: List[str] = [
    "item_name",
    "item_barcode",
    "category",
    "unit",
    "initial_stock",
    "current_stock",
]

# ---------- file I/O ----------
def _read_file(file) -> pd.DataFrame:
    if file.type == "text/csv":
        return pd.read_csv(StringIO(file.getvalue().decode("utf-8")))
    return pd.read_excel(BytesIO(file.getvalue()))

def _make_template(columns: List[str]) -> bytes:
    buf = BytesIO()
    pd.DataFrame(columns=columns).to_excel(buf, index=False, engine="openpyxl")
    buf.seek(0)
    return buf.read()

def _arrow_preview(df: pd.DataFrame) -> pd.DataFrame:
    prev = df.copy()
    for col in ["item_name", "item_barcode", "category", "unit"]:
        if col in prev.columns:
            prev[col] = prev[col].astype(str).where(prev[col].notna(), "")
    return prev


# ---------- PAGE ----------
def page():
    st.title("⬆️ Inventory Upload (stage → merge, skip duplicates)")
    st.caption(
        "Rows with NULLs in required columns (e.g., item_name, unit, initial_stock, current_stock) "
        "are skipped automatically. Duplicate (item_name/item_barcode) rows are also skipped."
    )

    st.download_button(
        "📄 Download Excel template",
        data=_make_template(INVENTORY_TEMPLATE_COLS),
        file_name="inventory_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="secondary",
    )

    file = st.file_uploader(
        "Choose CSV or Excel for inventory",
        key="inventory_uploader",
        type=["csv", "xlsx", "xls"],
        help="Headers must match real columns (subset allowed). Do not include item_id/created_at/updated_at.",
    )
    if file is None:
        st.info("No file selected yet.")
        st.divider()
        return

    df = _read_file(file)
    st.info(f"Loaded file with **{df.shape[0]}** rows and **{df.shape[1]}** columns.")
    st.dataframe(_arrow_preview(df.head(250)), use_container_width=True, height=300)

    if st.button("✅ Commit to DB", key="commit_inventory", type="primary"):
        with st.spinner("Staging and merging (skipping invalid & duplicates)…"):
            try:
                result = bulk_insert_inventory_skip_conflicts(df=df)
                st.success("✅ Inventory upload finished.")
                st.write(
                    f"Staged: **{result['staged']}**, "
                    f"Valid for insert: **{result['valid_rows']}**, "
                    f"Inserted: **{result['inserted']}**, "
                    f"Skipped (NULL required): **{result['skipped_null_required']}**, "
                    f"Skipped (duplicates): **{result['skipped_duplicates']}** "
                    f"(via {'COPY' if result.get('used_copy') else 'executemany'})"
                )
                if "used_columns" in result:
                    st.write("Used columns:", ", ".join(result["used_columns"]))
                with st.expander("Timing (ms)"):
                    st.json({k: round(v, 2) for k, v in result["timings"].items()})

                # Post-insert debug preview
                try:
                    count_df = fetch_dataframe('SELECT COUNT(*) AS total FROM "public"."inventory";')
                    st.info(f"📊 Table `inventory` now has **{int(count_df['total'].iloc[0])}** total rows.")
                    st.write("Last 5 rows by item_id:")
                    last_rows = fetch_dataframe('SELECT * FROM "public"."inventory" ORDER BY item_id DESC LIMIT 5;')
                    st.dataframe(last_rows, use_container_width=True, height=220)
                except Exception as dbg_err:
                    st.warning(f"Post-insert debug query failed: {dbg_err}")

            except Exception as exc:
                st.error(f"❌ Upload failed → {exc}")

    st.divider()


if __name__ == "__main__":
    st.set_page_config(page_title="Inventory Upload", page_icon="⬆️", layout="wide")
    page()
