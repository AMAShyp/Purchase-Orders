"""
db_handler.py  –  Lightweight Neon connection helper
Place in the same folder as app.py
"""
import os
import streamlit as st
from sqlalchemy import create_engine

@st.cache_resource(show_spinner="🔗 Connecting to database…")
def get_engine():
    db_url = (
        st.secrets.get("DATABASE_URL")  # Streamlit Cloud
        or os.getenv("DATABASE_URL")    # local dev / CI
    )
    if not db_url:
        raise RuntimeError(
            "DATABASE_URL not found – add it to secrets.toml or env vars."
        )

    if "sslmode" not in db_url:
        db_url += ("&" if "?" in db_url else "?") + "sslmode=require"

    return create_engine(db_url, pool_pre_ping=True, pool_size=5, max_overflow=10)
