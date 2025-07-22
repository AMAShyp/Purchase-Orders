"""
Business logic for reorder suggestions.
---------------------------------------

* Uses the last *window_days* of sales to compute average daily demand.
* Suggests a quantity that covers *desired_days* of future sales.
"""

from __future__ import annotations
import pandas as pd
from typing import Literal
from db_handler import fetch_dataframe


def _sales_summary(window_days: int) -> pd.DataFrame:
    sql = f"""
        SELECT item_id, SUM(quantity) AS qty_last_{window_days}d
        FROM sales
        WHERE sale_date >= CURRENT_DATE - INTERVAL '{window_days} days'
        GROUP BY item_id;
    """
    return fetch_dataframe(sql)


def _inventory_snapshot() -> pd.DataFrame:
    sql = """
        SELECT item_id, item_name, item_barcode, current_stock
        FROM inventory;
    """
    return fetch_dataframe(sql)


def reorder_suggestions(
    desired_days: int = 7,
    window_days: int = 28,
) -> pd.DataFrame:
    """
    Return a DataFrame listing items that need reordering.

    Columns: item_id, item_name, item_barcode, current_stock,
             avg_daily_sales, required_stock, reorder_qty
    """
    inv = _inventory_snapshot()
    sales = _sales_summary(window_days)

    df = inv.merge(sales, on="item_id", how="left")
    df[f"qty_last_{window_days}d"].fillna(0, inplace=True)

    df["avg_daily_sales"] = df[f"qty_last_{window_days}d"] / window_days
    df["required_stock"] = (df["avg_daily_sales"] * desired_days).round()
    df["reorder_qty"] = (
        (df["required_stock"] - df["current_stock"]).clip(lower=0).astype(int)
    )

    needed = df[df["reorder_qty"] > 0].copy()
    needed.sort_values("reorder_qty", ascending=False, inplace=True)

    return needed[
        [
            "item_id",
            "item_name",
            "item_barcode",
            "current_stock",
            "avg_daily_sales",
            "required_stock",
            "reorder_qty",
        ]
    ]
