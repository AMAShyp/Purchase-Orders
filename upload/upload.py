# upload.py – Streamlit bulk upload UI (final)
# Shows granular debug steps before/during/after DB commit

import streamlit as st
import pandas as pd
from io import StringIO, BytesIO
from typing import List, Tuple, Optional

from upload_handler import (
    upsert_dataframe,
    check_columns,
    DebugSink,
)
from db_handler import fetch_dataframe

MAX_NAME_LEN = 255
MAX_BARCODE_LEN = 128

# ---------- file I/O ----------
def _read_file(file) -> pd.DataFrame:
    if file.type == "text/csv":
        return pd.read_csv(StringIO(file.getvalue().decode("utf-8")))
    return pd.read_excel(BytesIO(file.getvalue()))

def _make_template(columns: List[str]) -> bytes:
    buf = BytesIO()
    pd.DataFrame(columns=columns).to_excel(buf, index=False, engine="openpyxl")
    buf.seek(0)
    return buf.read()

# ---------- normalizers ----------
def _normalize_name(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.casefold()

def _normalize_barcode(series: pd.Series) -> pd.Series:
    s = series.fillna("").astype(str).str.strip()
    return s.str.replace(r"\.0$", "", regex=True)

# ---------- preview: make Arrow-friendly ----------
def _arrow_friendly_preview(df: pd.DataFrame, table: str) -> pd.DataFrame:
    preview = df.copy()
    text_cols = {
        "inventory": ["item_name", "item_barcode", "category", "unit"],
        "purchases": ["bill_type", "item_name", "item_barcode"],
        "sales":     ["bill_type", "item_name", "item_barcode"],
    }.get(table, [])
    for col in text_cols:
        if col in preview.columns:
            preview[col] = preview[col].astype(str).where(preview[col].notna(), "")
            preview[col] = preview[col].replace({"nan": ""})
    return preview

# ---------- inventory duplicate policy ----------
def _filter_inventory_conflicts(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    name_norm = _normalize_name(df["item_name"])
    code_norm = _normalize_barcode(df.get("item_barcode", ""))

    dup_name_file = name_norm.map(name_norm.value_counts()).gt(1) & name_norm.ne("")
    dup_code_file = code_norm.map(code_norm.value_counts()).gt(1) & code_norm.ne("")

    try:
        existing = fetch_dataframe("SELECT item_name, item_barcode FROM inventory;")
        db_names = set(_normalize_name(existing["item_name"]))
        db_codes = set(_normalize_barcode(existing["item_barcode"]))
    except Exception as e:
        st.warning(f"Could not fetch existing inventory for duplicate check: {e}")
        db_names, db_codes = set(), set()

    dup_name_db = name_norm.isin(db_names) & name_norm.ne("")
    dup_code_db = code_norm.isin(db_codes) & code_norm.ne("")

    name_dup_any = dup_name_file | dup_name_db
    code_dup_any = dup_code_file | dup_code_db

    barcode_nonempty = code_norm.ne("")
    both_dup_mask = name_dup_any & code_dup_any & barcode_nonempty

    skipped = df[both_dup_mask]
    filtered = df[~both_dup_mask].copy()
    return filtered, skipped

# ---------- length guard ----------
def _validate_inventory_lengths(df: pd.DataFrame) -> Tuple[bool, pd.DataFrame]:
    name = df["item_name"].fillna("").astype(str)
    code = _normalize_barcode(df.get("item_barcode", pd.Series([], dtype="object")))

    too_long = (name.str.len() > MAX_NAME_LEN) | (code.str.len() > MAX_BARCODE_LEN)
    if not too_long.any():
        return True, df

    st.error(
        f"Some rows exceed length limits (item_name ≤ {MAX_NAME_LEN}, item_barcode ≤ {MAX_BARCODE_LEN})."
    )
    show = df.loc[too_long].copy()
    show["item_name_len"] = name[too_long].str.len().values
    show["item_barcode_len"] = code[too_long].str.len().values
    st.dataframe(_arrow_friendly_preview(show, "inventory"), use_container_width=True, height=220)

    if st.checkbox("✂️ Auto-truncate long values and continue (not recommended)"):
        df2 = df.copy()
        df2["item_name"] = name.str.slice(0, MAX_NAME_LEN)
        code2 = code.str.slice(0, MAX_BARCODE_LEN)
        code2 = code2.mask(code2.str.strip() == "", pd.NA)
        df2["item_barcode"] = code2
        st.info("Values truncated to fit limits.")
        return True, df2

    return False, df

# ---------- Debug UI helpers ----------
def _show_debug(title: str, logs: List[str]):
    if not logs:
        return
    with st.expander(title, expanded=False):
        for ln in logs:
            st.text(ln)

def _pre_snapshot(table: str) -> Optional[pd.DataFrame]:
    try:
        return fetch_dataframe(f"SELECT COUNT(*) AS c FROM {table};")
    except Exception:
        return None

def _post_snapshot(table: str) -> Optional[pd.DataFrame]:
    try:
        return fetch_dataframe(f"SELECT COUNT(*) AS c FROM {table};")
    except Exception:
        return None

# ---------- section ----------
def _section(label: str, table: str, required_cols: List[str]):
    st.subheader(label)

    c1, c2 = st.columns([1, 2])
    with c1:
        st.download_button(
            "📄 Download Excel template",
            data=_make_template(required_cols),
            file_name=f"{table}_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"tmpl_{table}",
        )

    with c2:
        st.caption("Template includes required columns only. You can add optional columns if your DB has them.")

    file = st.file_uploader(
        f"Choose CSV or Excel for **{label}**",
        key=f"{table}_uploader",
        type=["csv", "xlsx", "xls"],
    )
    if file is None:
        st.info("No file selected yet.")
        st.divider()
        return

    # BEFORE: snapshot table count
    before = _pre_snapshot(table)
    if before is not None:
        st.info(f"🔎 Before insert → `{table}` rows: {int(before['c'].iloc[0])}")

    # Load & preview
    df = _read_file(file)
    st.info(f"📥 Loaded file with {df.shape[0]} rows and {df.shape[1]} columns.")
    st.dataframe(_arrow_friendly_preview(df, table), use_container_width=True, height=300)

    # Validate columns
    try:
        check_columns(df, table)  # raises on error
        valid = True
        st.success("✅ Column check passed.")
    except ValueError as e:
        st.error(f"❌ Column check failed: {e}")
        valid = False

    # Inventory-only extra checks
    df_to_commit = df
    if valid and table == "inventory":
        if (df["item_name"].astype(str).str.strip() == "").any() or df["item_name"].isna().any():
            st.error("❌ All inventory rows must have a non-empty item_name. Barcode can be blank.")
            valid = False
        else:
            ok_len, df_lenfixed = _validate_inventory_lengths(df)
            st.info(f"After length check: {df_lenfixed.shape[0]} rows remain.")
            if ok_len:
                filtered, skipped = _filter_inventory_conflicts(df_lenfixed)
                st.info(f"After duplicate filtering: {filtered.shape[0]} rows to insert, {skipped.shape[0]} skipped.")
                if not skipped.empty:
                    st.warning("⚠️ Skipped rows due to both name & barcode duplicate.")
                    st.dataframe(_arrow_friendly_preview(skipped, "inventory"),
                                 use_container_width=True, height=200)
                if filtered.empty:
                    st.error("❌ No rows left to insert after filtering.")
                    valid = False
                else:
                    filtered = filtered.copy()
                    if "item_barcode" in filtered.columns:
                        bc = _normalize_barcode(filtered["item_barcode"])
                        filtered["item_barcode"] = bc.mask(bc.str.strip() == "", pd.NA)
                    st.info("Final data to commit:")
                    st.dataframe(_arrow_friendly_preview(filtered, "inventory"),
                                 use_container_width=True, height=300)
                    df_to_commit = filtered

    # Commit
    dbg = DebugSink()
    commit_btn = st.button("✅ Commit to DB", key=f"commit_{table}", disabled=not valid, type="primary")
    if commit_btn:
        with st.status("Running bulk insert…", expanded=True) as status:
            st.write("• Validating and normalising data…")
            try:
                # Let the handler run everything inside one DB transaction
                upsert_dataframe(df=df_to_commit, table=table, debug=dbg)
                st.write("• Insert statements executed.")
                status.update(label="Commit successful ✅", state="complete")

                # AFTER: snapshot table count
                after = _post_snapshot(table)
                if after is not None:
                    st.info(f"📊 After insert → `{table}` rows: {int(after['c'].iloc[0])}")
                # Show last 5 rows for quick sanity
                try:
                    st.write("Here are the last 5 rows in the table:")
                    last_rows = fetch_dataframe(
                        f"SELECT * FROM {table} ORDER BY item_id DESC LIMIT 5;" if table == "inventory"
                        else f"SELECT * FROM {table} ORDER BY 1 DESC LIMIT 5;"
                    )
                    st.dataframe(last_rows, use_container_width=True)
                except Exception as dbg_err:
                    st.warning(f"Post-insert debug query failed: {dbg_err}")

            except Exception as exc:
                status.update(label="Commit failed ❌", state="error")
                st.error(f"❌ Upload failed → {exc}")

        # Always show detailed internal logs
        _show_debug("🪵 Detailed debug log (from transaction)", dbg.dump())

    st.divider()

# ---------- PAGE ENTRY POINT ----------
def page():
    st.title("⬆️ Bulk Uploads (with Full Debug)")
    st.caption("Upload Excel/CSV to insert rows into the database. Every step is logged for visibility.")

    _section(
        "Inventory Items",
        "inventory",
        ["item_name", "item_barcode", "category", "unit", "initial_stock", "current_stock"],
    )
    _section(
        "Daily Purchases",
        "purchases",
        ["bill_type", "purchase_date", "item_name", "item_barcode", "quantity", "purchase_price"],
    )
    _section(
        "Daily Sales",
        "sales",
        ["bill_type", "sale_date", "item_name", "item_barcode", "quantity", "sale_price"],
    )

if __name__ == "__main__":
    st.set_page_config(page_title="Upload", page_icon="⬆️", layout="wide")
    page()
