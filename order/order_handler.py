"""
Business logic for reorder & over-stock suggestions + inline stock corrections.
------------------------------------------------------------------------------

* Reorder list = items whose current_stock < demand for `desired_days`.
* Over-stock list = items whose current_stock > demand for `overstock_days`.
* New: apply inline inventory corrections in bulk + optional adjustment logging.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Literal, List, Dict, Any
from sqlalchemy import text
from db_handler import fetch_dataframe, run_transaction


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
    df["current_stock"] = pd.to_numeric(df["current_stock"], errors="coerce").fillna(0)

    df["avg_daily_sales"] = df[qty_col] / window_days
    return df


# ───────────────────────── Public: insights ─────────────────────────── #
def reorder_suggestions(
    desired_days: int = 7,
    window_days: int = 28,
) -> pd.DataFrame:
    """Items that need topping-up to cover `desired_days`."""
    df = _base_df(window_days)

    df["required_stock"] = (df["avg_daily_sales"] * desired_days).round()
    df["reorder_qty"] = ((df["required_stock"] - df["current_stock"]).clip(lower=0)).astype(int)

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

    # days of stock on hand (avoid div-by-zero)
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


# ───────────────────────── New: stock adjustments ───────────────────── #
@run_transaction
def _ensure_adjustments_table(conn) -> None:
    """
    Create a simple adjustments table if it doesn't exist.
    Tracks manual corrections made from the Reorder page.
    """
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS stock_adjustments (
            id SERIAL PRIMARY KEY,
            item_id INT NOT NULL REFERENCES inventory(item_id),
            old_stock INT NOT NULL,
            new_stock INT NOT NULL,
            adjustment_qty INT NOT NULL,
            reason TEXT,
            source TEXT DEFAULT 'reorder_page',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """))


@run_transaction
def apply_stock_overrides(
    conn,
    *,
    rows: pd.DataFrame,
    reason: str = "manual correction from reorder page",
    log_adjustments: bool = True,
) -> Dict[str, Any]:
    """
    Apply absolute stock overrides (set current_stock = new_stock) for many items at once.

    Expected columns in `rows`:
        - item_id (int)
        - old_stock (int)   # the current_stock the user saw
        - new_stock (int)   # the value the user entered

    Returns a summary dict with counts and totals.
    """
    if rows.empty:
        return {"updated": 0, "inserted_logs": 0, "total_delta": 0}

    # Clean & validate
    df = rows.copy()
    df["item_id"] = pd.to_numeric(df["item_id"], errors="coerce").astype("Int64")
    df["old_stock"] = pd.to_numeric(df["old_stock"], errors="coerce").fillna(0).round().astype(int)
    df["new_stock"] = pd.to_numeric(df["new_stock"], errors="coerce").fillna(0).round().astype(int)

    df = df[df["item_id"].notna()].copy()
    df["item_id"] = df["item_id"].astype(int)

    df["delta"] = df["new_stock"] - df["old_stock"]
    df = df[df["delta"] != 0]  # only rows that actually change

    if df.empty:
        return {"updated": 0, "inserted_logs": 0, "total_delta": 0}

    # Bulk UPDATE using VALUES
    pairs = list(zip(df["item_id"].tolist(), df["new_stock"].tolist()))
    values_clause = ", ".join(f"(:id{i}, :nv{i})" for i in range(len(pairs)))
    params: Dict[str, Any] = {}
    for i, (iid, nv) in enumerate(pairs):
        params[f"id{i}"] = int(iid)
        params[f"nv{i}"] = int(nv)

    conn.execute(
        text(f"""
            UPDATE "public"."inventory" AS i
            SET current_stock = v.new_stock,
                updated_at = CURRENT_TIMESTAMP
            FROM (VALUES {values_clause}) AS v(item_id, new_stock)
            WHERE i.item_id = v.item_id;
        """),
        params,
    )

    # Optional: log adjustments
    inserted_logs = 0
    total_delta = int(df["delta"].sum())
    if log_adjustments:
        _ensure_adjustments_table(conn)  # within same transaction

        # Prepare payload
        log_params = []
        for _, r in df.iterrows():
            log_params.append({
                "iid": int(r["item_id"]),
                "old": int(r["old_stock"]),
                "new": int(r["new_stock"]),
                "adj": int(r["delta"]),
                "reason": reason,
            })

        conn.execute(
            text("""
                INSERT INTO stock_adjustments (item_id, old_stock, new_stock, adjustment_qty, reason, source)
                VALUES (:iid, :old, :new, :adj, :reason, 'reorder_page');
            """),
            log_params,
        )
        inserted_logs = len(log_params)

    return {"updated": len(pairs), "inserted_logs": inserted_logs, "total_delta": total_delta}
