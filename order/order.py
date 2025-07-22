import streamlit as st
import pandas as pd
from .order_handler import reorder_suggestions


def page() -> None:
    st.title("📑 Reorder Suggestions")

    with st.sidebar:
        st.header("Settings")
        desired_days = st.slider("Target coverage (days)", 1, 30, 7)
        window_days = st.slider("Sales history window", 7, 60, 28, step=7)

    df = reorder_suggestions(desired_days=desired_days, window_days=window_days)

    st.write(
        f"Items that need reordering to cover **{desired_days}** days "
        f"(based on last **{window_days}** days of sales):"
    )

    if df.empty:
        st.success("🚀 All items are sufficiently stocked!")
        return

    st.dataframe(df, use_container_width=True, height=500)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download CSV",
        data=csv,
        file_name="reorder_suggestions.csv",
        mime="text/csv",
    )


# Optional standalone execution
if __name__ == "__main__":
    st.set_page_config(page_title="Reorder", page_icon="📑", layout="wide")
    page()
