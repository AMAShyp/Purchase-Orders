# sale_upload.py – Unified Purchases & Sales uploader
# - One template: bill_type + txn_date + (item_name, item_barcode) + input_quantity/output_quantity + unit_price
# - Mode switch: Insert & Update Inventory  OR  Insert Only (no stock changes)
# - Minimal validation; relies on upload_handler for fast COPY + repair mode

from __future__ import annotations

import io
from io import StringIO, BytesIO
from typing import List

import pandas as pd
import streamlit as st

# Robust imports (package vs flat)
try:
    from .upload_handler import bulk_insert_unified_txns
except Exception:
    from upload_handler import bulk_insert_unified_txns


TEMPLATE_COLUMNS: List[str] = [
    "bill_type",         # e.g. "sales invoice", "sales return invoice",
                         #      "purchase invoice direct", "purchasing return invoice"
    "txn_date",          # will be mapped to sale_date / purchase_date
    "item_name",
    "item_barcode",
    "input_quantity",    # used for: purchase invoice (direct) & sales return invoice
    "output_quantity",   # used for: sales invoice & purchasing return invoice
    "unit_price",        # mapped to sale_price / purchase_price
    # Optional helpers when creating new items (only for positive purchases):
    "category",
    "unit",
]

# ---------- File I/O ----------
def _read_file(file) -> pd.DataFrame:
    if file.type == "text/csv":
        return pd.read_csv(StringIO(file.getvalue().decode("utf-8")))
    return pd.read_excel(BytesIO(file.getvalue()))

def _make_template() -> bytes:
    buf = BytesIO()
    pd.DataFrame(columns=TEMPLATE_COLUMNS).to_excel(buf, index=False, engine="openpyxl")
    buf.seek(0)
    return buf.read()

def _arrow_preview(df: pd.DataFrame) -> pd.DataFrame:
    prev = df.copy()
    for col in ["bill_type", "item_name", "item_barcode"]:
        if col in prev.columns:
            prev[col] = prev[col].astype(str).where(prev[col].notna(), "")
    return prev


# ---------- PAGE ----------
def page():
    st.title("⬆️ Sales & Purchases Upload (Unified)")
    st.caption("Upload a single file with both sales **and** purchases. Choose whether to update inventory now or insert only for planning.")

    # Mode switch
    mode = st.radio(
        "Behavior after inserting rows:",
        ["Insert & Update Inventory (recommended)", "Insert Only (no stock changes)"],
        index=0,
        horizontal=False,
    )
    update_inventory = (mode.startswith("Insert & Update"))

    # Pair-repair policy
    policy = st.selectbox(
        "If item_name and item_barcode disagree for the same row, prefer:",
        ["Barcode", "Name"],
        index=0,
        help="This is used only when the name points to one item and the barcode points to a different item.",
    )
    mismatch_policy = "prefer_barcode" if policy == "Barcode" else "prefer_name"

    # Template download
    st.download_button(
        "📄 Download Excel template",
        data=_make_template(),
        file_name="unified_transactions_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="secondary",
    )

    file = st.file_uploader(
        "Choose CSV or Excel",
        type=["csv", "xlsx", "xls"],
        key="unified_uploader",
        help="Headers must match the template. Large files are supported.",
    )
    if not file:
        st.info("No file selected yet.")
        st.divider()
        return

    df = _read_file(file)
    st.info(f"Loaded file: **{df.shape[0]} rows**, **{df.shape[1]} columns**.")
    st.dataframe(_arrow_preview(df.head(200)), use_container_width=True, height=280)

    if st.button("✅ Commit to DB", type="primary"):
        with st.spinner("Processing and inserting rows…"):
            try:
                result = bulk_insert_unified_txns(
                    df=df,
                    mismatch_policy=mismatch_policy,
                    update_inventory=update_inventory,
                )
                # Summary
                st.success("✅ Upload completed.")
                st.write(f"**Mode:** {result.get('mode')}")
                st.write(f"**Unknown bill_type rows skipped:** {result.get('unknown_bill_type_rows', 0)}")

                # Purchases block
                p = result.get("purchases")
                s = result.get("sales")
                if p:
                    st.info(
                        f"🧾 Purchases inserted: **{p.get('rows', 0)}** "
                        f"(via {'COPY' if p.get('used_copy') else 'executemany'})"
                    )
                if s:
                    st.info(
                        f"🧾 Sales inserted: **{s.get('rows', 0)}** "
                        f"(via {'COPY' if s.get('used_copy') else 'executemany'})"
                    )

                # Inventory update / preview
                inv = result.get("inventory_update") or {}
                if inv.get("mode") == "insert_and_update":
                    st.success(
                        f"📦 Inventory updated for **{inv.get('items_updated', 0)}** items "
                        f"(net delta: {inv.get('net_delta', 0)})"
                    )
                else:
                    st.warning(
                        f"📦 Insert-only mode: inventory **not** updated. "
                        f"Preview — items that would change: **{inv.get('items_would_change', 0)}**, "
                        f"net delta preview: **{inv.get('net_delta_preview', 0)}**."
                    )

                # Repairs / Skips
                rep = result.get("repairs_summary") or {}
                skip = result.get("skipped_summary") or {}
                with st.expander("Details: repairs & skipped"):
                    st.write("**Repairs summary**")
                    st.json(rep, expanded=False)
                    st.write("**Skipped summary**")
                    st.json(skip, expanded=False)

                # Timing
                with st.expander("Timing (ms)"):
                    t_total = round(result.get("total_ms", 0.0), 2)
                    st.write(f"Total processing: **{t_total} ms**")
                    if p and "timings" in p:
                        st.write("Purchases timings:")
                        st.json({k: round(v, 2) for k, v in p["timings"].items()})
                    if s and "timings" in s:
                        st.write("Sales timings:")
                        st.json({k: round(v, 2) for k, v in s["timings"].items()})

            except Exception as exc:
                st.error(f"❌ Upload failed → {exc}")

    st.divider()


if __name__ == "__main__":
    st.set_page_config(page_title="Sales & Purchases Upload", page_icon="⬆️", layout="wide")
    page()
