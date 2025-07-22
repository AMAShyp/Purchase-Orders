"""
Centralised database helper for the Hypermarket apps.
-----------------------------------------------------

• Creates one pooled SQLAlchemy engine (cached) pointing to Neon.
• Works both locally (.env / export) and on Streamlit Cloud (st.secrets).
• Provides tiny helpers for common patterns: fetch_dataframe & execute.
"""

from __future__ import annotations

import os
import streamlit as st
from sqlalchemy import create_engine, text
import pandas as pd

# ───────────────────────── Engine (singleton) ────────────────────────── #
@st.cache_resource(show_spinner="🔗 Connecting to database…")
def get_engine():
    """
    Return a singleton SQLAlchemy engine with connection-pooling and SSL
    (Neon requires `sslmode=require`).  Reads DATABASE_URL from:

        1. st.secrets        (production on Streamlit Cloud)
        2. environment var   (local dev / GitHub CI)
    """
    db_url = (
        st.secrets.get("DATABASE_URL")  # type: ignore[attr-defined]
        or os.getenv("DATABASE_URL")
    )
    if not db_url:
        raise RuntimeError(
            "DATABASE_URL not found.  "
            "Add it to Streamlit secrets or export it in your shell."
        )

    # Ensure sslmode=require is present (Neon best-practice)
    if "sslmode" not in db_url:
        db_url += ("&" if "?" in db_url else "?") + "sslmode=require"

    return create_engine(
        db_url,
        pool_pre_ping=True,       # drop stale conns
        pool_size=5,
        max_overflow=10,
    )

# ───────────────────────── Helper wrappers ───────────────────────────── #
def fetch_dataframe(sql: str, params: dict | None = None) -> pd.DataFrame:
    """Return the query as a Pandas DataFrame."""
    engine = get_engine()
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})

def execute(sql: str, params: dict | None = None) -> None:
    """Run INSERT/UPDATE/DELETE (auto-commit)."""
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text(sql), params or {})

# Optional convenience for explicit transactions later on:
def run_transaction(fn):
    """
    Decorator to wrap a function in a single DB transaction.

    Example
    -------
    @run_transaction
    def add_purchase(conn, item_id, qty, price):
        conn.execute(...)
    """
    def _wrapper(*args, **kwargs):
        engine = get_engine()
        with engine.begin() as conn:
            return fn(conn, *args, **kwargs)
    return _wrapper
