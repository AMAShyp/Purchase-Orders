import streamlit as st
import pandas as pd
from io import StringIO, BytesIO
from .upload_handler import upsert_dataframe, check_columns

st.set_page_config(page_title="Upload Data", page_icon="⬆️", layout="wide")
st.title("⬆️ Bulk Upload")

# ───────────────────────── Sidebar controls ───────────────────────────── #
st.sidebar.header("Upload Options")

data_type = st.sidebar.selectbox(
    "What does this file contain?",
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

# ───────────────────────── Main area ───────────────────────────────────── #
if not file:
    st.info("⬅️ Select a file to begin.")
    st.stop()

# Read file into a DataFrame
if file.type == "text/csv":
    df = pd.read_csv(StringIO(file.getvalue().decode("utf-8")))
else:
    df = pd.read_excel(BytesIO(file.getvalue()))

st.subheader("Preview")
st.dataframe(df, height=400, use_container_width=True)

# Validate columns
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
            upsert_dataframe(df=df, table=target_table)  # uses db_handler under the hood
        except Exception as exc:
            st.error(f"Upload failed → {exc}")
        else:
            st.success(f"Inserted {len(df)} rows into **{target_table}**.")
