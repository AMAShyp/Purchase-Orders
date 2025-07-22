import streamlit as st
import pandas as pd
from io import StringIO, BytesIO
from .upload_handler import upsert_dataframe, check_columns

# ───────────────────────── Page function ─────────────────────────────── #
def page() -> None:
    """Render the Upload page (called from app.py)."""
    st.header("⬆️ Bulk Upload")

    # Sidebar controls
    st.sidebar.subheader("Upload Options")
    data_type = st.sidebar.selectbox(
        "File represents …",
        ("initial inventory", "daily purchases", "daily sales"),
    )
    table_map = {
        "initial inventory": "inventory",
        "daily purchases": "purchases",
        "daily sales": "sales",
    }
    target_table = table_map[data_type]

    file = st.sidebar.file_uploader(
        "Choose CSV or Excel",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=False,
    )

    if not file:
        st.info("⬅️ Select a file to begin.")
        return

    # Read file
    if file.type == "text/csv":
        df = pd.read_csv(StringIO(file.getvalue().decode("utf-8")))
    else:
        df = pd.read_excel(BytesIO(file.getvalue()))

    st.subheader("Preview")
    st.dataframe(df, height=400, use_container_width=True)

    # Validate
    try:
        check_columns(df, target_table)
        valid = True
    except ValueError as e:
        st.error(str(e))
        valid = False

    # Commit button
    if st.button("✅ Commit to DB", disabled=not valid):
        with st.spinner("Inserting rows…"):
            try:
                upsert_dataframe(df=df, table=target_table)
            except Exception as exc:
                st.error(f"Upload failed → {exc}")
            else:
                st.success(f"Inserted {len(df)} rows into **{target_table}**.")

# Optional: allow standalone execution for quick testing
if __name__ == "__main__":
    st.set_page_config(page_title="Upload", page_icon="⬆️", layout="wide")
    page()
