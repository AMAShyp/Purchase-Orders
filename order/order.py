import streamlit as st
import pandas as pd
from .order_handler import reorder_suggestions, overstock_items


def page() -> None:
    st.title("📑 Stock Insights")

    # ───────── Sidebar controls ───────── #
    with st.sidebar:
        st.header("Settings")
        desired_days = st.slider("Re‑order coverage (days)", 1, 30, 7)
        overstock_days = st.slider("Over‑stock threshold (days)", 5, 60, 10)
        window_days = st.slider("Sales history window", 7, 60, 28, step=7)

    # ───────── Re‑order section ───────── #
    st.subheader("🛒 Items that need re‑ordering")
    need_df = reorder_suggestions(desired_days, window_days)

    if need_df.empty:
        st.success("🚀 All items are sufficiently stocked for "
                   f"{desired_days} days.")
    else:
        st.dataframe(need_df, use_container_width=True, height=350)
        csv = need_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download re‑order CSV",
            data=csv,
            file_name="reorder_suggestions.csv",
            mime="text/csv",
        )

    st.divider()

    # ───────── Over‑stock section ───────── #
    st.subheader("📦 Over‑stocked items")
    over_df = overstock_items(overstock_days, window_days)

    if over_df.empty:
        st.success("🎉 No over‑stock detected for "
                   f"the next {overstock_days} days.")
    else:
        st.dataframe(over_df, use_container_width=True, height=350)
        csv_over = over_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download over‑stock CSV",
            data=csv_over,
            file_name="overstock_items.csv",
            mime="text/csv",
        )


# standalone test
if __name__ == "__main__":
    st.set_page_config(page_title="Stock Insights", page_icon="📑", layout="wide")
    page()
