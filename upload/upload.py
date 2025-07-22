import streamlit as st
import pandas as pd
from io import StringIO, BytesIO
from .upload_handler import upsert_dataframe, check_columns

# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #
def read_file(file) -> pd.DataFrame:
    """Return uploaded CSV/XLS(X) as DataFrame."""
    if file.type == "text/csv":
        return pd.read_csv(StringIO(file.getvalue().decode("utf-8")))
    return pd.read_excel(BytesIO(file.getvalue()))


def make_template(columns: list[str]) -> bytes:
    """Return an in-memory Excel file containing only headers."""
    buf = BytesIO()
    pd.DataFrame(columns=columns).to_excel(buf, index=False, engine="openpyxl")
    buf.seek(0)
    return buf.read()


# --------------------------------------------------------------------- #
# Re-usable upload section
# --------------------------------------------------------------------- #
def upload_section(label: str, target_table: str, required_cols: list[str]) -> None:
    st.subheader(label)

    # --- Template download button ------------------------------------ #
    st.download_button(
        label="📄 Download Excel template",
        data=make_template(required_cols),
        file_name=f"{target_table}_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=f"tmpl_{target_table}",
    )

    # --- File uploader ------------------------------------------------ #
    file = st.file_uploader(
        f"Choose CSV or Excel for **{label}**",
        key=f"{target_table}_uploader",
        type=["csv", "xlsx", "xls"],
    )

    if file is None:
        st.info("No file selected yet.")
        st.divider()
        return

    df = read_file(file)
    st.write("Preview:")
    st.dataframe(df, use_container_width=True, height=300)

    # --- Validation --------------------------------------------------- #
    try:
        check_columns(df, target_table)
        valid = True
    except ValueError as e:
        st.error(str(e))
        valid = False

    # --- Commit button ------------------------------------------------ #
    if st.button("✅ Commit to DB", key=f"commit_{target_table}", disabled=not valid):
        with st.spinner("Inserting rows …"):
            try:
                upsert_dataframe(df=df, table=target_table)
            except Exception as exc:
                st.error(f"Upload failed → {exc}")
            else:
                st.success(f"Inserted {len(df)} rows into **{target_table}**.")

    st.divider()


# --------------------------------------------------------------------- #
# Page entry point
# --------------------------------------------------------------------- #
def page() -> None:
    st.title("⬆️ Bulk Uploads")

    upload_section(
        label="Inventory Items",
        target_table="inventory",
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
        label="Daily Purchases",
        target_table="purchases",
        required_cols=["item_id", "quantity", "purchase_price"],
    )

    upload_section(
        label="Daily Sales",
        target_table="sales",
        required_cols=["item_id", "quantity", "sale_price"],
    )


# Stand-alone test capability
if __name__ == "__main__":
    st.set_page_config(page_title="Upload", page_icon="⬆️", layout="wide")
    page()
