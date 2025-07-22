import streamlit as st
import pandas as pd
from sqlalchemy import text
from db_handler import get_engine      # ← single import

st.set_page_config(page_title="Hypermarket Inventory", page_icon="📦", layout="wide")

@st.cache_data(ttl=60)
def load_inventory() -> pd.DataFrame:
    sql = text("""
        SELECT item_id,
               item_name,
               item_barcode,
               category,
               current_stock,
               unit,
               updated_at
        FROM inventory
        ORDER BY item_name;
    """)
    with get_engine().connect() as conn:
        return pd.read_sql(sql, conn)

# ────────── UI ──────────
st.title("📦 Current Inventory Dashboard")

df = load_inventory()

left, right = st.columns(2)
left.metric("Distinct Items", len(df))
right.metric("Total Units in Stock", int(df["current_stock"].sum()))

st.dataframe(df, use_container_width=True, hide_index=True)
