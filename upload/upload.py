# upload.py – Inventory uploader (fast COPY, NULL-tolerant staging, skip duplicates)
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


def page():
    st.title("⬆️ Inventory Upload (NULL-tolerant, skip duplicates)")
    st.caption("We stage to a TEMP table (all columns nullable), insert only valid rows into `inventory`, and skip duplicates.")

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
        help="Headers must be real column names (subset allowed). Do not include item_id/created_at/updated_at.",
    )
    if file is None:
        st.info("No file selected yet.")
        st.divider()
        return

    df = _read_file(file)
    st.info(f"Loaded file with **{df.shape[0]}** rows and **{df.shape[1]}** columns.")
    st.dataframe(_arrow_preview(df.head(250)), use_container_width=True, height=300)

    if st.button("✅ Commit to DB", key="commit_inventory", type="primary"):
        with st.spinner("Staging and inserting (NULL-tolerant, skipping duplicates)…"):
            try:
                result = bulk_insert_inventory_skip_conflicts(df=df)
                st.success("✅ Inventory upload finished.")
                msg = (
                    f"Staged: **{result['staged']}** · "
                    f"Inserted: **{result['inserted']}** · "
                    f"Skipped (NULL required): **{result.get('skipped_null_required', 0)}** · "
                    f"Skipped (duplicates): **{result.get('skipped_duplicates', 0)}**"
                )
                st.write(msg)
                if "used_columns" in result:
                    st.write("Used columns:", ", ".join(result["used_columns"]))
                with st.expander("Timing (ms)"):
                    st.json({k: round(v, 2) for k, v in result["timings"].items()})

                # Post-insert debug preview
                try:
                    count_df = fetch_dataframe('SELECT COUNT(*) AS total FROM "public"."inventory";')
                    st.info(f"📊 Table `inventory` now has **{int(count_df['total'].iloc[0])}** total rows.")
                    st.write("Last 5 rows by item_id:")
                    last_rows = fet_
