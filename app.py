import streamlit as st
from db_handler import fetch_dataframe
from upload import upload as upload_page  # <- import the module, not the function

st.set_page_config(page_title="Hypermarket App", page_icon="📦", layout="wide")

# ───────────────────────── Sidebar navigation ────────────────────────── #
st.sidebar.title("Navigation")
choice = st.sidebar.radio(
    "Go to",
    ("Dashboard", "Upload", "Reorder"),
    index=0,
)

# ───────────────────────── Dashboard page ────────────────────────────── #
def dashboard() -> None:
    st.title("📦 Current Inventory Dashboard")

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

    df = load_inventory()

    left, right = st.columns(2)
    left.metric("Distinct Items", len(df))
    right.metric("Total Units in Stock", int(df["current_stock"].sum()))

    st.dataframe(df, use_container_width=True, hide_index=True)

# ───────────────────────── Page router ───────────────────────────────── #
if choice == "Dashboard":
    dashboard()
else:  # "Upload"
    upload_page.page()
