import streamlit as st
import pandas as pd
from io import StringIO, BytesIO
from .upload_handler import upsert_dataframe, check_columns

# ───────────────────────── Helpers ───────────────────────────────────── #
def read_file(file) -> pd.DataFrame:
    if file.type == "text/csv":
        return pd.read_csv(StringIO(file.getvalue().decode("utf-8")))
    return pd.read_excel(BytesIO(file.getvalue()))  # xls/xlsx


def upload_section(label: str, target_table: str) -> None:
    st.subheader(label)

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

    # Basic column validation
    try:
        check_columns(df, target_table)
        valid = True
    except ValueError as e:
        st.error(str(e))
        valid = False

    if st.button("✅ Commit to DB", key=f"commit_{target_table}", disabled=not valid):
        with st.spinner("Inserting rows …"):
            try:
                upsert_dataframe(df=df, table=target_table)
            except Exception as exc:
                st.error(f"Upload failed → {exc}")
            else:
                st.success(f"Inserted {len(df)} rows into **{target_table}**.")
    st.divider()


# ───────────────────────── Page entry point ──────────────────────────── #
def page() -> None:
    st.title("⬆️ Bulk Uploads")

    # Three independent sections
    upload_section("Inventory Items", target_table="inventory")
    upload_section("Daily Purchases", target_table="purchases")
    upload_section("Daily Sales",     target_table="sales")


# Optional: standalone run for quick local testing
if __name__ == "__main__":
    st.set_page_config(page_title="Upload", page_icon="⬆️", layout="wide")
    page()
