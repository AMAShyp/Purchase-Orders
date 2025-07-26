"""
Business logic for reorder & over‑stock suggestions.
----------------------------------------------------

* Reorder list = items whose current_stock < demand for `desired_days`.
* Over‑stock list = items whose current_stock > demand for `overstock_days`.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Literal
from db_handler import fetch_dataframe


# ───────────────────────── Internal helpers ─────────────────────────── #
def _sales_summary(window_days: int) -> pd.DataFrame:
    sql = f"""
        SELECT item_id,
               SUM(quantity)::numeric AS qty_last_{window_days}d
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


def _base_df(window_days: int) -> pd.DataFrame:
    """Return merged DF with avg_daily_sales already computed."""
    inv = _inventory_snapshot()
    sales = _sales_summary(window_days)

    df = inv.merge(sales, on="item_id", how="left")
    qty_col = f"qty_last_{window_days}d"

    df[qty_col].fillna(0, inplace=True)
    df[qty_col] = pd.to_numeric(df[qty_col], errors="coerce").fillna(0)
    df["current_stock"] = pd.to_numeric(
        df["current_stock"], errors="coerce"
    ).fillna(0)

    df["avg_daily_sales"] = df[qty_col] / window_days
    return df


# ───────────────────────── Public API functions ─────────────────────── #
def reorder_suggestions(
    desired_days: int = 7,
    window_days: int = 28,
) -> pd.DataFrame:
    """Items that need topping‑up to cover `desired_days`."""
    df = _base_df(window_days)

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


def overstock_items(
    overstock_days: int = 10,
    window_days: int = 28,
) -> pd.DataFrame:
    """
    Items holding more stock than needed for `overstock_days`.

    Columns returned:
        item_id, item_name, item_barcode, current_stock,
        avg_daily_sales, stock_for_days, excess_qty
    """
    df = _base_df(window_days)

    df["required_stock"] = (df["avg_daily_sales"] * overstock_days).round()
    df["excess_qty"] = (df["current_stock"] - df["required_stock"]).astype(int)

    # For avg_daily_sales == 0 (no recent sales) treat any positive stock as excess
    no_sales_mask = df["avg_daily_sales"] == 0
    df.loc[no_sales_mask, "excess_qty"] = df.loc[no_sales_mask, "current_stock"]

    over = df[df["excess_qty"] > 0].copy()

    # days of stock on hand (avoid div‑by‑zero)
    over["stock_for_days"] = np.where(
        over["avg_daily_sales"] > 0,
        (over["current_stock"] / over["avg_daily_sales"]).round(1),
        np.inf,
    )

    over.sort_values("excess_qty", ascending=False, inplace=True)

    return over[
        [
            "item_id",
            "item_name",
            "item_barcode",
            "current_stock",
            "avg_daily_sales",
            "stock_for_days",
            "excess_qty",
        ]
    ]
