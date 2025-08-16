import streamlit as st
import pandas as pd
from .order_handler import (
    reorder_suggestions,
    overstock_items,
    apply_stock_overrides,
)

def _editable_grid(df: pd.DataFrame, key: str, help_text: str) -> pd.DataFrame:
    """
    Show a limited, editable view with a 'new_current_stock' column.
    Returns the edited DataFrame (not filtered to changed rows).
    """
    if df.empty:
        st.info("No items.")
        return df

    # Create editable copy with a new column
    edit_df = df.copy()
    edit_df["new_current_stock"] = edit_df["current_stock"]

    st.caption(help_text)
    edited = st.data_editor(
        edit_df,
        hide_index=True,
        use_container_width=True,
        height=420,
        key=key,
        column_config={
            "item_id": st.column_config.NumberColumn("item_id", disabled=True),
            "item_name": st.column_config.TextColumn("item_name", disabled=True),
            "item_barcode": st.column_config.TextColumn("item_barcode", disabled=True),
            "current_stock": st.column_config.NumberColumn("current_stock", help="Existing value", disabled=True),
            "avg_daily_sales": st.column_config.NumberColumn("avg_daily_sales", format="%.2f", disabled=True),
            "required_stock": st.column_config.NumberColumn("required_stock", disabled=True) if "required_stock" in df.columns else None,
            "reorder_qty": st.column_config.NumberColumn("reorder_qty", disabled=True) if "reorder_qty" in df.columns else None,
            "stock_for_days": st.column_config.NumberColumn("stock_for_days", disabled=True) if "stock_for_days" in df.columns else None,
            "excess_qty": st.column_config.NumberColumn("excess_qty", disabled=True) if "excess_qty" in df.columns else None,
            "new_current_stock": st.column_config.NumberColumn("new_current_stock", help="Edit this to correct stock"),
        },
    )
    return pd.DataFrame(edited)


def _extract_changes(edited_df: pd.DataFrame) -> pd.DataFrame:
    """
    From an edited grid DF (must include item_id, current_stock, new_current_stock),
    return only the rows that changed with columns: item_id, old_stock, new_stock.
    """
    if edited_df.empty:
        return pd.DataFrame(columns=["item_id", "old_stock", "new_stock"])

    # Ensure required columns exist
    required = {"item_id", "current_stock", "new_current_stock"}
    missing = required - set(edited_df.columns)
    if missing:
        # Defensive fallback (nothing to apply)
        return pd.DataFrame(columns=["item_id", "old_stock", "new_stock"])

    # Coerce numeric
    edited_df["current_stock"] = pd.to_numeric(edited_df["current_stock"], errors="coerce").fillna(0).round().astype(int)
    edited_df["new_current_stock"] = pd.to_numeric(edited_df["new_current_stock"], errors="coerce").fillna(0).round().astype(int)

    changed = edited_df[edited_df["new_current_stock"] != edited_df["current_stock"]].copy()
    if changed.empty:
        return pd.DataFrame(columns=["item_id", "old_stock", "new_stock"])

    # ✅ Rename both columns so final selection exists
    changed = changed.rename(columns={
        "current_stock": "old_stock",
        "new_current_stock": "new_stock",
    })
    changed = changed[["item_id", "old_stock", "new_stock"]]
    return changed


def page() -> None:
    st.title("📑 Stock Insights")

    # ───────── Sidebar controls ───────── #
    with st.sidebar:
        st.header("Settings")
        desired_days = st.slider("Re-order coverage (days)", 1, 30, 7)
        overstock_days = st.slider("Over-stock threshold (days)", 5, 60, 10)
        window_days = st.slider("Sales history window", 7, 60, 28, step=7)
        st.divider()
        apply_reason = st.text_input("Adjustment reason", "manual correction from reorder page")
        log_changes = st.checkbox("Log adjustments (recommended)", value=True)

    # ───────── Re-order section ───────── #
    st.subheader("🛒 Items that need re-ordering")
    need_df = reorder_suggestions(desired_days, window_days)

    if need_df.empty:
        st.success(f"🚀 All items are sufficiently stocked for {desired_days} days.")
    else:
        edited_need = _editable_grid(
            need_df,
            key="need_grid",
            help_text="Tip: Edit **New Stock** for any item you want to correct, then click **Apply corrections** below.",
        )
        changes_need = _extract_changes(edited_need)
        c1, c2 = st.columns([1, 3])
        with c1:
            st.metric("Changed (Re-order)", len(changes_need))
        with c2:
            if st.button("✅ Apply corrections (Re-order section)", key="apply_need", use_container_width=False):
                if changes_need.empty:
                    st.info("No changes to apply.")
                else:
                    res = apply_stock_overrides(rows=changes_need, reason=apply_reason, log_adjustments=log_changes)
                    st.success(
                        f"Applied {res['updated']} updates "
                        f"(Δ total: {res['total_delta']:+d}); "
                        f"logs inserted: {res['inserted_logs']}"
                    )

    st.divider()

    # ───────── Over-stock section ───────── #
    st.subheader("📦 Over-stocked items")
    over_df = overstock_items(overstock_days, window_days)

    if over_df.empty:
        st.success(f"🎉 No over-stock detected for the next {overstock_days} days.")
    else:
        edited_over = _editable_grid(
            over_df,
            key="over_grid",
            help_text="Edit **New Stock** to correct. This is great for quickly dialing down obvious over-stock.",
        )
        changes_over = _extract_changes(edited_over)
        c1, c2 = st.columns([1, 3])
        with c1:
            st.metric("Changed (Over-stock)", len(changes_over))
        with c2:
            if st.button("✅ Apply corrections (Over-stock section)", key="apply_over", use_container_width=False):
                if changes_over.empty:
                    st.info("No changes to apply.")
                else:
                    res = apply_stock_overrides(rows=changes_over, reason=apply_reason, log_adjustments=log_changes)
                    st.success(
                        f"Applied {res['updated']} updates "
                        f"(Δ total: {res['total_delta']:+d}); "
                        f"logs inserted: {res['inserted_logs']}"
                    )


# standalone test
if __name__ == "__main__":
    st.set_page_config(page_title="Stock Insights", page_icon="📑", layout="wide")
    page()
