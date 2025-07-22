import streamlit as st
from sqlalchemy import create_engine
import pandas as pd
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

# Set up database connection
engine = create_engine(DATABASE_URL)

# Function to load inventory data
@st.cache_data
def load_inventory():
    query = "SELECT item_id, item_name, item_barcode, category, current_stock, unit, updated_at FROM inventory;"
    return pd.read_sql(query, engine)

# Streamlit UI
def main():
    st.title("📦 Inventory Dashboard")

    # Load data
    inventory_df = load_inventory()

    st.subheader("Current Inventory Stock")

    # Display inventory in a table
    st.dataframe(inventory_df, use_container_width=True)

    # Simple metrics
    total_items = inventory_df.shape[0]
    total_stock = inventory_df['current_stock'].sum()

    col1, col2 = st.columns(2)
    col1.metric("Total Items", total_items)
    col2.metric("Total Stock Units", total_stock)

if __name__ == "__main__":
    main()
