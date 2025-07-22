import streamlit as st
from db_handler import fetch_dataframe
# (no more SQLAlchemy imports here)

st.set_page_config(page_title="Hypermarket Inventory", page_icon="📦", layout="wide")

@st.cache_data(ttl=60)
def load_inventory():
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

st.title("📦 Current Inventory Dashboard")

df = load_inventory()

left, right = st.columns(2)
left.metric("Distinct Items", len(df))
right.metric("Total Units in Stock", int(df["current_stock"].sum()))

st.dataframe(df, use_container_width=True, hide_index=True)
