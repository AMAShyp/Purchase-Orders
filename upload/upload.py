import streamlit as st
import pandas as pd
from io import StringIO, BytesIO
from .upload_handler import upsert_dataframe, check_columns

# ---------- helpers ----------
def _read_file(file):
    # Read as text for CSV; for Excel read as strings to preserve barcodes
    if file.type == "text/csv":
        return pd.read_csv(StringIO(file.getvalue().decode("utf-8")))
    return pd.read_excel(BytesIO(file.getvalue()), dtype=str, engine="openpyxl")

def _make_template(columns):
    buf = BytesIO()
    pd.DataFrame(columns=columns).to_excel(buf, index=False, engine="openpyxl")
    buf.seek(0)
    return buf.read()

def _preview_fix(df: pd.DataFrame, table: str) -> pd.DataFrame:
    """Coerce text cols to strings & reorder for a clean preview."""
    df = df.copy()
    if table == "inventory":
        for c in ["item_name", "item_barcode", "category", "unit"]:
            if c in df.columns:
                df[c] = df[c].astype(str)
        order = ["item_name", "item_barcode", "category", "unit", "initial_stock", "current_stock"]
    elif table == "purchases":
        for c in ["bill_type", "purchase_date", "item_name", "item_barcode"]:
            if c in df.columns:
                df[c] = df[c].astype(str)
        order = ["bill_type", "purchase_date", "item_name", "item_barcode", "quantity", "purchase_price"]
    else:
        for c in ["bill_type", "sale_date", "item_name", "item_barcode"]:
            if c in df.columns:
                df[c] = df[c].astype(str)
        order = ["bill_type", "sale_date", "item_name", "item_barcode", "quantity", "sale_price"]
    present = [c for c in order if c in df.columns]
    return df[present + [c for c in df.columns if c not in present]]

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
                   "Sales Invoice / Sales Return Invoice, "
                   "Purchase Invoice / Purchase Return Invoice.")

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
    st.dataframe(_preview_fix(df, table), use_container_width=True, height=300)

    try:
        check_columns(df, table)
        valid = True
    except ValueError as e:
        st.error(str(e))
        valid = False

    if st.button("✅ Commit to DB", key=f"commit_{table}", disabled=not valid):
        with st.spinner("Inserting rows …"):
            try:
                result = upsert_dataframe(df=df, table=table)
            except Exception as exc:
                st.error(f"Upload failed → {exc}")
            else:
                if table == "inventory":
                    msg = (
                        f"✅ Done.\n\n"
                        f"- Written (inserted/updated): **{result['written']}** rows\n"
                        f"- Skipped due to duplicate barcodes: **{result['skipped_barcode']}**"
                    )
                    st.success(msg)
                    if result.get("skipped_rows"):
                        buf = BytesIO()
                        pd.DataFrame(result["skipped_rows"]).to_csv(buf, index=False)
                        st.download_button(
                            "⬇️ Download skipped rows (duplicate barcodes)",
                            data=buf.getvalue(),
                            file_name="skipped_duplicate_barcodes.csv",
                            mime="text/csv",
                            key=f"dl_skipped_{table}",
                        )
                else:
                    st.success(f"Inserted/updated **{result.get('written', len(df))}** rows into **{table}**.")
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

# standalone test
if __name__ == "__main__":
    st.set_page_config(page_title="Upload", page_icon="⬆️", layout="wide")
    page()
