import importlib
import streamlit as st
import pandas as pd
from db_handler import fetch_dataframe

# ───────────────────────── Dashboard helper ─────────────────────────── #
@st.cache_data(ttl=60)
def load_inventory() -> pd.DataFrame:
    sql = """
        SELECT item_id,
               item_name,
               item_barcode,
               category,
               current_stock,
               unit,
               updated_at
        FROM inventory
        ORDER BY item_name;
    """
    return fetch_dataframe(sql)


def dashboard() -> None:
    st.title("📦 Current Inventory Dashboard")

    df = load_inventory()

    col1, col2 = st.columns(2)
    col1.metric("Distinct Items", len(df))
    col2.metric("Total Units in Stock", int(df["current_stock"].sum()))

    st.dataframe(df, use_container_width=True, hide_index=True)


# ───────────────────────── App layout & router ──────────────────────── #
st.set_page_config(page_title="Hypermarket App", page_icon="📦", layout="wide")

st.sidebar.title("Navigation")
choice = st.sidebar.radio(
    "Go to",
    ("Dashboard", "Upload", "Reorder"),
    index=0,
)

if choice == "Dashboard":
    dashboard()

elif choice == "Upload":
    # Lazy-import the page and surface any import errors
    try:
        upload_module = importlib.import_module("upload.upload")
        upload_module.page()
    except Exception as e:
        st.error("❌ Failed to load *Upload* page:")
        st.exception(e)

else:  # "Reorder"
    try:
        order_module = importlib.import_module("order.order")
        order_module.page()
    except Exception as e:
        st.error("❌ Failed to load *Reorder* page:")
        st.exception(e)
