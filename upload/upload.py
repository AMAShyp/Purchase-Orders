@@ -1,112 +1,69 @@
import streamlit as st
import pandas as pd
from io import StringIO, BytesIO
from .upload_handler import upsert_dataframe, check_columns

# --------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------- #
def read_file(file) -> pd.DataFrame:
    if file.type == "text/csv":
        return pd.read_csv(StringIO(file.getvalue().decode("utf-8")))
    return pd.read_excel(BytesIO(file.getvalue()))


def make_template(columns: list[str]) -> bytes:
    buf = BytesIO()
    pd.DataFrame(columns=columns).to_excel(buf, index=False, engine="openpyxl")
    buf.seek(0)
    return buf.read()


def upload_section(label: str, table: str, required_cols: list[str]) -> None:
    st.subheader(label)

    st.download_button(
        "📄 Download Excel template",
        data=make_template(required_cols),
        file_name=f"{table}_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=f"tmpl_{table}",

    )

    file = st.file_uploader(
        f"Choose CSV or Excel for **{label}**",
        key=f"{table}_uploader",
        type=["csv", "xlsx", "xls"],
    )
    if file is None:
        st.info("No file selected yet.")
        st.divider()
        return

    df = read_file(file)
    st.write("Preview:")
    st.dataframe(df, use_container_width=True, height=300)

    # Validate
    try:
        check_columns(df, table)
        valid = True
    except ValueError as e:
        st.error(str(e))
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


# --------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------- #
def page() -> None:
    st.title("⬆️ Bulk Uploads")

    upload_section(
        "Inventory Items",
        table="inventory",
        required_cols=[
            "item_name",
            "item_barcode",
            "category",
            "initial_stock",
            "current_stock",
            "unit",
        ],
    )

    upload_section(
        "Daily Purchases",
        table="purchases",
        required_cols=[
            "item_name",      # or item_barcode / item_id
            "item_barcode",
            "quantity",
            "purchase_price",
            "purchase_date",  # optional – defaults to today
        ],
    )

    upload_section(
        "Daily Sales",
        table="sales",
        required_cols=[
            "item_name",      # or item_barcode / item_id
            "item_barcode",
            "quantity",
            "sale_price",
            "sale_date",      # optional – defaults to today
        ],
    )


if __name__ == "__main__":
    st.set_page_config(page_title="Upload", page_icon="⬆️", layout="wide")
    page()
