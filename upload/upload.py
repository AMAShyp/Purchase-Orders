import streamlit as st
import pandas as pd
from io import StringIO, BytesIO
from .upload_handler import upsert_dataframe, check_columns
from db_handler import fetch_dataframe  # ← to check against existing inventory


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
    # lowercased, trimmed; empty -> ""
    return series.fillna("").astype(str).str.strip().str.casefold()

def _normalize_barcode(series: pd.Series) -> pd.Series:
    # keep as text, strip spaces, drop trailing ".0" if Excel coerced numbers
    s = series.fillna("").astype(str).str.strip()
    return s.str.replace(r"\.0$", "", regex=True)

def _validate_inventory_uniques(df: pd.DataFrame) -> bool:
    """
    Validate inventory upload:
      - no missing item_name / item_barcode
      - no duplicates within the file
      - no duplicates vs existing DB inventory
    Returns True if OK, else False (and renders errors in Streamlit).
    """
    ok = True
    df_local = df.copy()

    # Basic presence
    name_norm = _normalize_name(df_local["item_name"])
    code_norm = _normalize_barcode(df_local["item_barcode"])

    missing_name_mask = name_norm.eq("")
    missing_code_mask = code_norm.eq("")

    if missing_name_mask.any():
        st.error(f"Missing `item_name` in rows: {list(df_local.index[missing_name_mask])}")
        st.dataframe(df_local[missing_name_mask], use_container_width=True, height=150)
        ok = False

    if missing_code_mask.any():
        st.error(f"Missing `item_barcode` in rows: {list(df_local.index[missing_code_mask])}")
        st.dataframe(df_local[missing_code_mask], use_container_width=True, height=150)
        ok = False

    # Duplicates within the file
    dup_name_mask = name_norm.duplicated(keep=False) & name_norm.ne("")
    if dup_name_mask.any():
        st.error("Duplicate `item_name` found within the uploaded file.")
        st.dataframe(df_local[dup_name_mask], use_container_width=True, height=200)
        ok = False

    dup_code_mask = code_norm.duplicated(keep=False) & code_norm.ne("")
    if dup_code_mask.any():
        st.error("Duplicate `item_barcode` found within the uploaded file.")
        st.dataframe(df_local[dup_code_mask], use_container_width=True, height=200)
        ok = False

    # Conflicts against existing DB inventory
    try:
        existing = fetch_dataframe("SELECT item_name, item_barcode FROM inventory;")
        db_names = _normalize_name(existing["item_name"])
        db_codes = _normalize_barcode(existing["item_barcode"])
        name_conflict_mask = name_norm.isin(set(db_names)) & name_norm.ne("")
        code_conflict_mask = code_norm.isin(set(db_codes)) & code_norm.ne("")

        if name_conflict_mask.any():
            st.error("These rows conflict with existing `item_name` values in inventory.")
            st.dataframe(df_local[name_conflict_mask], use_container_width=True, height=200)
            ok = False

        if code_conflict_mask.any():
            st.error("These rows conflict with existing `item_barcode` values in inventory.")
            st.dataframe(df_local[code_conflict_mask], use_container_width=True, height=200)
            ok = False

    except Exception as e:
        # If DB is unreachable, don't hard-fail the page—just warn.
        st.warning(f"Could not validate against existing inventory: {e}")

    return ok

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

    # Schema validation
    try:
        check_columns(df, table)
        valid = True
    except ValueError as e:
        st.error(str(e))
        valid = False

    # Extra duplicate checks for INVENTORY uploads (in this page layer)
    if valid and table == "inventory":
        if not _validate_inventory_uniques(df):
            valid = False

    if st.button("✅ Commit to DB", key=f"commit_{table}", disabled=not valid):
        with st.spinner("Inserting rows …"):
            try:
                upsert_dataframe(df=df, table=table)
            except Exception as exc:
                st.error(f"Upload failed → {exc}")
            else:
                st.success(f"Inserted {len(df)} rows into **{table}**.")
    st.divider()

# ---------- PAGE ENTRY POINT ----------
def page() -> None:
    st.title("⬆️ Bulk Uploads")

    # Inventory: item_name, item_barcode, category, unit, initial_stock, current_stock
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
