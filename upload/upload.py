import streamlit as st
import pandas as pd
from io import StringIO, BytesIO
from .upload_handler import upsert_dataframe, check_columns
from db_handler import fetch_dataframe  # for DB duplicate checks


# ---------- helpers ----------
def _read_file(file):
    if file.type == "text/csv":
        return pd.read_csv(StringIO(file.getvalue().decode("utf-8")))
    return pd.read_excel(BytesIO(file.getvalue()))

def _make_template(columns):
    buf = BytesIO()
    pd.DataFrame(columns=columns).to_excel(buf, index=False, engine="openpyxl")
    buf.seek(0)
    return buf.read()

def _normalize_name(series: pd.Series) -> pd.Series:
    # lowercased, trimmed
    return series.fillna("").astype(str).str.strip().str.casefold()

def _normalize_barcode(series: pd.Series) -> pd.Series:
    # text, trimmed, drop trailing ".0" (Excel artifact)
    s = series.fillna("").astype(str).str.strip()
    return s.str.replace(r"\.0$", "", regex=True)

def _filter_inventory_conflicts(df: pd.DataFrame):
    """
    Skip rows ONLY when BOTH item_name AND item_barcode are duplicates
    (within file OR vs DB). Empty barcode never blocks the row.
    Returns (filtered_df, skipped_df).
    """
    dfc = df.copy()

    name_norm = _normalize_name(dfc["item_name"])
    code_norm = _normalize_barcode(dfc.get("item_barcode", ""))

    # Dups within file
    dup_name_file = name_norm.map(name_norm.value_counts()).gt(1) & name_norm.ne("")
    dup_code_file = code_norm.map(code_norm.value_counts()).gt(1) & code_norm.ne("")

    # Dups vs DB
    try:
        existing = fetch_dataframe("SELECT item_name, item_barcode FROM inventory;")
        db_names = set(_normalize_name(existing["item_name"]))
        db_codes = set(_normalize_barcode(existing["item_barcode"]))
    except Exception:
        db_names, db_codes = set(), set()

    dup_name_db = name_norm.isin(db_names) & name_norm.ne("")
    dup_code_db = code_norm.isin(db_codes) & code_norm.ne("")

    name_dup_any = dup_name_file | dup_name_db
    code_dup_any = dup_code_file | dup_code_db

    # If barcode is empty, do NOT consider it a duplicate blocker
    barcode_nonempty = code_norm.ne("")
    both_dup_mask = name_dup_any & code_dup_any & barcode_nonempty

    skipped = dfc[both_dup_mask]
    filtered = dfc[~both_dup_mask].copy()
    return filtered, skipped

def _section(label, table, required_cols):
    st.subheader(label)

    st.download_button(
        "📄 Download Excel template",
        data=_make_template(required_cols),
        file_name=f"{table}_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=f"tmpl_{table}",
    )

    if table in ("purchases", "sales"):
        st.caption("Allowed bill types (case-insensitive): "
                   "Sales Invoice, Sales Return Invoice, "
                   "Purchase Invoice, Purchase Return Invoice.")

    file = st.file_uploader(
        f"Choose CSV or Excel for **{label}**",
        key=f"{table}_uploader",
        type=["csv", "xlsx", "xls"],
    )
    if file is None:
        st.info("No file selected yet.")
        st.divider()
        return

    df = _read_file(file)
    st.write("Preview:")
    st.dataframe(df, use_container_width=True, height=300)

    # Schema validation (columns must exist; values can be blank)
    try:
        check_columns(df, table)
        valid = True
    except ValueError as e:
        st.error(str(e))
        valid = False

    # Inventory-specific duplicate policy
    df_to_commit = df
    if valid and table == "inventory":
        # Require item_name (barcode optional)
        if df["item_name"].isna().any() | (df["item_name"].astype(str).str.strip() == "").any():
            st.error("All inventory rows must have a non-empty item_name. Barcode can be blank.")
            valid = False
        else:
            filtered, skipped = _filter_inventory_conflicts(df)
            if len(skipped) > 0:
                st.warning(
                    f"Skipping {len(skipped)} row(s) where BOTH item_name and item_barcode "
                    "duplicate existing data. Rows with empty barcode or a single-field "
                    "duplicate are allowed."
                )
                with st.expander("Show skipped rows"):
                    st.dataframe(skipped, use_container_width=True, height=220)
            if filtered.empty:
                st.info("After skipping conflicting rows, there is nothing to insert.")
                valid = False
            else:
                df_to_commit = filtered

    if st.button("✅ Commit to DB", key=f"commit_{table}", disabled=not valid):
        with st.spinner("Inserting rows …"):
            try:
                upsert_dataframe(df=df_to_commit, table=table)
            except Exception as exc:
                st.error(f"Upload failed → {exc}")
            else:
                st.success(f"Inserted {len(df_to_commit)} rows into **{table}**.")
    st.divider()

# ---------- PAGE ENTRY POINT ----------
def page() -> None:
    st.title("⬆️ Bulk Uploads")

    # Inventory: item_name, item_barcode (optional), category, unit, initial_stock, current_stock
    _section(
        "Inventory Items",
        "inventory",
        ["item_name", "item_barcode", "category", "unit", "initial_stock", "current_stock"],
    )

    # Purchases: bill_type, purchase_date, item_name, item_barcode, quantity, purchase_price
    _section(
        "Daily Purchases",
        "purchases",
        ["bill_type", "purchase_date", "item_name", "item_barcode", "quantity", "purchase_price"],
    )

    # Sales: bill_type, sale_date, item_name, item_barcode, quantity, sale_price
    _section(
        "Daily Sales",
        "sales",
        ["bill_type", "sale_date", "item_name", "item_barcode", "quantity", "sale_price"],
    )

if __name__ == "__main__":
    st.set_page_config(page_title="Upload", page_icon="⬆️", layout="wide")
    page()
