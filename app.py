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
    # import only when needed → avoids circular-import timing issues
    from upload.upload import page as upload_page
    upload_page()

else:  # "Reorder"
    from order.order import page as order_page
    order_page()
