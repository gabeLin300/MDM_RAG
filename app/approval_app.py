"""
MDM_RAG Product Attribute Extraction & Review Dashboard
--------------------------------------------------------
Streamlit app for data stewards to run the full RAG extraction pipeline
on a product CSV and review / approve / reject extracted attributes.

Run from MDM_RAG/app/:
    streamlit run approval_app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
APP_DIR = Path(__file__).parent
REPO_ROOT = APP_DIR.parent
DATA_DIR = APP_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ── File paths ─────────────────────────────────────────────────────────────────
UPLOADED_CSV_PATH = DATA_DIR / "uploaded_data.csv"
EXTRACTED_JSON_PATH = DATA_DIR / "extracted_output.json"
REVIEWED_CSV_PATH = DATA_DIR / "reviewed_output.csv"
APPROVED_CSV_PATH = DATA_DIR / "approved_pim_export.csv"
FAISS_DIR = DATA_DIR / "processed"
DEFAULT_CSV_PATH = REPO_ROOT / "data" / "raw" / "100_sample_advanced_rag.csv"

# ── Attribute list (mirrors UnifiedAgent.ATTRIBUTES) ──────────────────────────
ATTRIBUTES = [
    "certifications", "standards_compliance", "regulatory_approvals",
    "safety_certifications", "environmental_certifications", "industry_certifications",
    "communication_protocols", "wired_interfaces", "ports",
    "network_capabilities", "data_rate", "bus_type",
    "voltage_rating", "current_rating", "power_consumption", "power_supply", "frequency",
    "operating_temperature", "storage_temperature", "humidity_range",
    "ingress_protection", "shock_resistance", "vibration_resistance", "altitude_rating",
    "dimensions", "weight", "material", "housing", "finish", "mounting_type", "enclosure_type",
]

STATUS_ICON = {
    "Approved": "🟢",
    "Edited":   "🔵",
    "Rejected": "🔴",
    "Pending":  "⚪",
}

# ── Session state ──────────────────────────────────────────────────────────────
_SS_DEFAULTS: dict = {
    "product_ids":      [],   # ordered list of all product IDs
    "current_idx":      0,    # index into product_ids
    "extracted_full":   {},   # {pid: full orchestrator result}
    "review_records":   {},   # {pid: {"attributes": {...}, "review_status": ..., "reviewer_comments": ...}}
    "pipeline_running": False,
    "pipeline_log":     [],
    "csv_loaded":       False,
    "uploaded_filename": "",
    "auto_loaded":      False,
}


def _init_ss() -> None:
    for key, val in _SS_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ── Data helpers ───────────────────────────────────────────────────────────────

def flatten_attributes(full_result: dict) -> dict:
    """Convert {attr: {"value": X, "confidence": N}} → {attr: X or ""}."""
    attrs = full_result.get("attributes", {})
    flat: dict = {}
    for attr_name in ATTRIBUTES:
        entry = attrs.get(attr_name, {})
        if isinstance(entry, dict):
            flat[attr_name] = entry.get("value") or ""
        else:
            flat[attr_name] = str(entry) if entry else ""
    return flat


def _build_review_record(pid: str, full_result: dict) -> dict:
    return {
        "attributes":        flatten_attributes(full_result),
        "review_status":     "Pending",
        "reviewer_comments": "",
    }


def load_extracted_json() -> dict:
    """Load extracted_output.json.  Handles both list-of-dicts and nested-dict formats."""
    if not EXTRACTED_JSON_PATH.exists():
        return {}
    try:
        with open(EXTRACTED_JSON_PATH, encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return {}

    # Convert the saved [{attr: val}, ...] format back to full result format
    result: dict = {}
    for pid, content in raw.items():
        if isinstance(content, list):
            attrs = {}
            for item in content:
                if isinstance(item, dict):
                    for k, v in item.items():
                        attrs[k] = {"value": v, "confidence": 0, "source_excerpt": None}
            result[pid] = {
                "product_id":      pid,
                "attributes":      attrs,
                "quality_flags":   [],
                "review_required": True,
            }
        elif isinstance(content, dict):
            result[pid] = content
    return result


def save_extracted_json(extracted_full: dict) -> None:
    """Persist extracted data as {pid: [{attr: val}, ...]}."""
    output: dict = {}
    for pid, full_result in extracted_full.items():
        attrs = full_result.get("attributes", {})
        output[pid] = [
            {attr_name: entry.get("value") if isinstance(entry, dict) else entry}
            for attr_name, entry in attrs.items()
            if (isinstance(entry, dict) and entry.get("value") is not None)
            or (not isinstance(entry, dict) and entry)
        ]
    with open(EXTRACTED_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def load_reviewed_csv() -> dict:
    """Load reviewed_output.csv → review_records dict."""
    if not REVIEWED_CSV_PATH.exists():
        return {}
    try:
        df = pd.read_csv(REVIEWED_CSV_PATH, dtype=str).fillna("")
        records: dict = {}
        skip_cols = {"product_id", "review_status", "reviewer_comments"}
        for _, row in df.iterrows():
            pid = row.get("product_id", "")
            if not pid:
                continue
            attrs = {col: row.get(col, "") for col in df.columns if col not in skip_cols}
            records[pid] = {
                "attributes":        attrs,
                "review_status":     row.get("review_status", "Pending"),
                "reviewer_comments": row.get("reviewer_comments", ""),
            }
        return records
    except Exception:
        return {}


def save_reviewed_csv(review_records: dict) -> None:
    """Write review_records → reviewed_output.csv."""
    rows = []
    for pid, rec in review_records.items():
        row: dict = {"product_id": pid}
        row.update(rec.get("attributes", {}))
        row["review_status"]     = rec.get("review_status", "Pending")
        row["reviewer_comments"] = rec.get("reviewer_comments", "")
        rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(REVIEWED_CSV_PATH, index=False)


def export_approved_csv(review_records: dict) -> pd.DataFrame:
    """Write Approved + Edited records to approved_pim_export.csv and return the DataFrame."""
    rows = []
    for pid, rec in review_records.items():
        if rec.get("review_status") in ("Approved", "Edited"):
            row: dict = {"product_id": pid}
            row.update(rec.get("attributes", {}))
            row["review_status"] = rec.get("review_status", "")
            rows.append(row)
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(APPROVED_CSV_PATH, index=False)
        return df
    return pd.DataFrame()


def prepare_csv_for_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Add any missing required columns so the pipeline schema check passes."""
    df = df.copy()
    # Map 'metadata' → 'document_type' if present
    if "metadata" in df.columns and "document_type" not in df.columns:
        df["document_type"] = df["metadata"]
    # Add missing required columns with safe defaults
    if "id" not in df.columns:
        df["id"] = [f"row-{i}" for i in range(len(df))]
    if "file_name" not in df.columns:
        df["file_name"] = df.get("title", pd.Series(["unknown"] * len(df)))
    if "document_type" not in df.columns:
        df["document_type"] = "specification"
    if "file_content" not in df.columns and "text" in df.columns:
        df["file_content"] = df["text"]
    return df


def compute_metrics(review_records: dict) -> dict:
    statuses = [r.get("review_status", "Pending") for r in review_records.values()]
    return {
        "Total":    len(statuses),
        "Approved": statuses.count("Approved"),
        "Edited":   statuses.count("Edited"),
        "Rejected": statuses.count("Rejected"),
        "Pending":  statuses.count("Pending"),
    }


def _populate_review_records(extracted_full: dict) -> None:
    """Create review_records entries for products not already reviewed."""
    existing = st.session_state.review_records
    for pid, full_result in extracted_full.items():
        if pid not in existing:
            existing[pid] = _build_review_record(pid, full_result)
    st.session_state.review_records = existing


def _load_extracted_into_state(extracted_full: dict) -> None:
    st.session_state.extracted_full = extracted_full
    st.session_state.product_ids    = sorted(extracted_full.keys())
    st.session_state.current_idx    = 0
    _populate_review_records(extracted_full)


# ── Pipeline execution ─────────────────────────────────────────────────────────

def run_extraction_pipeline(csv_path: Path) -> tuple[bool, list[str]]:
    """
    Run the full RAG extraction pipeline.
    All heavy imports are deferred to this function so Streamlit starts quickly.
    Returns (success, log_messages).
    """
    messages: list[str] = []
    try:
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / ".env")
        messages.append("✅ Environment loaded")

        # Lazy imports — keeps Streamlit startup fast
        from pipeline.run_pipeline import (   # noqa: PLC0415
            parse_documents,
            build_index_artifacts,
            run_batch_orchestrator,
        )
        from agents.orchestrator import Orchestrator  # noqa: PLC0415
        messages.append("✅ Pipeline modules imported")

        # Parse
        parsed_docs, ingestion_report = parse_documents(csv_path)
        n_parsed = len(parsed_docs)
        schema_ok = ingestion_report.get("schema_ok", False)
        messages.append(
            f"✅ Parsed {n_parsed} documents "
            f"(schema_ok={schema_ok}, rows_read={ingestion_report.get('rows_read', '?')})"
        )
        if not parsed_docs:
            messages.append("❌ No valid documents found. Check that the CSV has 'file_content' and 'product_id' columns.")
            return False, messages

        # Build FAISS index
        messages.append("⏳ Building embeddings + FAISS index (this may take a minute)…")
        FAISS_DIR.mkdir(parents=True, exist_ok=True)
        artifacts = build_index_artifacts(parsed_docs, output_dir=str(FAISS_DIR))
        if artifacts["index"] is None:
            messages.append("❌ FAISS index not built — no valid text chunks produced.")
            return False, messages
        n_chunks = artifacts["manifest"].get("chunks_count", 0)
        messages.append(f"✅ Indexed {n_chunks} chunks across {n_parsed} documents")

        # Batch extraction
        n_docs = len({c.get("doc_id") for c in artifacts["metadata_store"]})
        messages.append(
            f"⏳ Running LLM extraction on {n_docs} documents "
            "(1 call per document, ~1 s rate-limit delay each)…"
        )
        orchestrator = Orchestrator(
            index=artifacts["index"],
            metadata=artifacts["metadata_store"],
        )
        batch_result = run_batch_orchestrator(
            orchestrator,
            artifacts["metadata_store"],
            rate_limit_delay=1.0,
        )
        extracted_full = batch_result.get("extracted_attributes", {})
        n_ok   = batch_result.get("products_processed", 0)
        n_fail = batch_result.get("products_failed", 0)
        messages.append(f"✅ Extraction done: {n_ok} documents processed, {n_fail} failed")

        if not extracted_full:
            messages.append("⚠️ No attributes were extracted. Check GROQ_API_KEY in .env.")
            return False, messages

        # Persist results and update session state
        _load_extracted_into_state(extracted_full)
        save_extracted_json(extracted_full)
        n_pids = len(extracted_full)
        messages.append(f"✅ Saved {n_pids} products → {EXTRACTED_JSON_PATH.name}")
        return True, messages

    except Exception as exc:
        import traceback
        messages.append(f"❌ Unexpected error: {exc}")
        messages.append(traceback.format_exc())
        return False, messages


# ── Sidebar ────────────────────────────────────────────────────────────────────

def render_sidebar() -> None:
    with st.sidebar:
        st.title("MDM_RAG")
        st.caption("Product Attribute Extraction · Review Dashboard")
        st.divider()

        # ── 1. Load Product Data ─────────────────────────────────────────────
        st.subheader("1. Load Product Data")

        if DEFAULT_CSV_PATH.exists():
            if st.button("Load Default CSV", use_container_width=True):
                try:
                    df = pd.read_csv(DEFAULT_CSV_PATH, dtype=str)
                    df = prepare_csv_for_pipeline(df)
                    df.to_csv(UPLOADED_CSV_PATH, index=False)
                    st.session_state.csv_loaded       = True
                    st.session_state.uploaded_filename = DEFAULT_CSV_PATH.name
                    st.success(f"Loaded {DEFAULT_CSV_PATH.name} ({len(df)} rows)")
                except Exception as exc:
                    st.error(f"Failed to load default CSV: {exc}")

        uploaded_file = st.file_uploader(
            "Upload CSV",
            type=["csv"],
            help="Minimum required columns: product_id, file_content. "
                 "Optional: id, title, file_name, document_type, metadata.",
        )
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file, dtype=str)
                df = prepare_csv_for_pipeline(df)
                missing = [c for c in ("product_id", "file_content") if c not in df.columns]
                if missing:
                    st.error(f"Missing required column(s): {', '.join(missing)}")
                else:
                    df.to_csv(UPLOADED_CSV_PATH, index=False)
                    st.session_state.csv_loaded       = True
                    st.session_state.uploaded_filename = uploaded_file.name
                    st.success(f"Loaded {uploaded_file.name} ({len(df)} rows)")
            except Exception as exc:
                st.error(f"Failed to read CSV: {exc}")

        if st.session_state.csv_loaded:
            st.caption(f"Active: **{st.session_state.uploaded_filename}**")

        st.divider()

        # ── 2. Run Extraction ────────────────────────────────────────────────
        st.subheader("2. Run RAG Extraction")

        csv_ready = st.session_state.csv_loaded and UPLOADED_CSV_PATH.exists()
        if st.button(
            "▶ Run Extraction",
            use_container_width=True,
            disabled=not csv_ready or st.session_state.pipeline_running,
            help="Runs full pipeline: parse → chunk → embed → FAISS → LLM extraction",
        ):
            st.session_state.pipeline_running = True
            with st.spinner("Running extraction pipeline… (may take several minutes)"):
                success, log_msgs = run_extraction_pipeline(UPLOADED_CSV_PATH)
            st.session_state.pipeline_running = False
            st.session_state.pipeline_log     = log_msgs
            if success:
                st.rerun()

        if st.session_state.pipeline_log:
            with st.expander("Pipeline log", expanded=not st.session_state.product_ids):
                for msg in st.session_state.pipeline_log:
                    st.markdown(msg)

        st.divider()

        # ── 3. Load Existing Results ─────────────────────────────────────────
        st.subheader("3. Load Existing Results")

        if EXTRACTED_JSON_PATH.exists():
            if st.button(
                f"Load {EXTRACTED_JSON_PATH.name}",
                use_container_width=True,
                key="btn_load_extracted",
            ):
                data = load_extracted_json()
                if data:
                    _load_extracted_into_state(data)
                    st.success(f"Loaded {len(data)} products.")
                    st.rerun()
                else:
                    st.warning("File exists but could not be parsed.")

        json_upload = st.file_uploader(
            "Upload JSON results",
            type=["json"],
            key="json_uploader",
        )
        if json_upload is not None:
            try:
                raw = json.load(json_upload)
                extracted_full: dict = {}
                for pid, content in raw.items():
                    if isinstance(content, list):
                        attrs = {
                            k: {"value": v, "confidence": 0, "source_excerpt": None}
                            for item in content
                            for k, v in item.items()
                        }
                        extracted_full[pid] = {
                            "product_id": pid, "attributes": attrs,
                            "quality_flags": [], "review_required": True,
                        }
                    elif isinstance(content, dict):
                        extracted_full[pid] = content
                if extracted_full:
                    _load_extracted_into_state(extracted_full)
                    st.success(f"Loaded {len(extracted_full)} products.")
                    st.rerun()
                else:
                    st.warning("No valid product data found in uploaded JSON.")
            except Exception as exc:
                st.error(f"Failed to parse JSON: {exc}")

        if REVIEWED_CSV_PATH.exists():
            if st.button(
                f"Load {REVIEWED_CSV_PATH.name}",
                use_container_width=True,
                key="btn_load_reviewed",
            ):
                records = load_reviewed_csv()
                if records:
                    st.session_state.review_records.update(records)
                    if not st.session_state.product_ids:
                        st.session_state.product_ids = sorted(records.keys())
                    st.success(f"Loaded {len(records)} reviewed records.")
                    st.rerun()
                else:
                    st.warning("Could not load reviewed CSV.")

        st.divider()

        # ── 4. Navigate ──────────────────────────────────────────────────────
        product_ids = st.session_state.product_ids
        if product_ids:
            st.subheader("4. Navigate")
            metrics      = compute_metrics(st.session_state.review_records)
            reviewed_cnt = metrics["Approved"] + metrics["Edited"] + metrics["Rejected"]
            st.caption(f"**{reviewed_cnt} / {len(product_ids)}** products reviewed")

            def _on_selectbox_change():
                selected_pid = st.session_state["nav_selectbox"]
                try:
                    st.session_state.current_idx = product_ids.index(selected_pid)
                except ValueError:
                    pass

            st.selectbox(
                "Jump to product",
                options=product_ids,
                index=min(st.session_state.current_idx, len(product_ids) - 1),
                key="nav_selectbox",
                on_change=_on_selectbox_change,
            )


# ── Metrics bar ────────────────────────────────────────────────────────────────

def render_metrics() -> None:
    product_ids = st.session_state.product_ids
    if not product_ids:
        return
    metrics = compute_metrics(st.session_state.review_records)
    cols = st.columns(5)
    for col, (label, value) in zip(cols, metrics.items()):
        col.metric(label, value)
    st.divider()


# ── Product card ───────────────────────────────────────────────────────────────

def render_product_card() -> None:
    product_ids = st.session_state.product_ids
    if not product_ids:
        st.info(
            "No extraction data loaded.  "
            "Use the sidebar to upload a CSV and run extraction, "
            "or load an existing extracted_output.json."
        )
        return

    idx = min(st.session_state.current_idx, len(product_ids) - 1)
    pid = product_ids[idx]

    # Ensure a review record exists for this product
    if pid not in st.session_state.review_records:
        full_result = st.session_state.extracted_full.get(pid, {})
        st.session_state.review_records[pid] = _build_review_record(pid, full_result)

    record  = st.session_state.review_records[pid]
    status  = record.get("review_status", "Pending")
    badge   = STATUS_ICON.get(status, "⚪")

    # ── Header ───────────────────────────────────────────────────────────────
    hc1, hc2, hc3 = st.columns([5, 1, 1])
    with hc1:
        st.subheader(f"{badge} {pid}")
        st.caption(f"Product **{idx + 1}** of **{len(product_ids)}**  ·  Status: **{status}**")
    with hc2:
        if st.button("← Prev", use_container_width=True, disabled=idx == 0, key="btn_prev"):
            st.session_state.current_idx = idx - 1
            st.rerun()
    with hc3:
        if st.button("Next →", use_container_width=True,
                     disabled=idx >= len(product_ids) - 1, key="btn_next"):
            st.session_state.current_idx = idx + 1
            st.rerun()

    # ── Quality flags & confidence (collapsed by default) ────────────────────
    full_result   = st.session_state.extracted_full.get(pid, {})
    quality_flags = full_result.get("quality_flags", [])
    if quality_flags:
        with st.expander(f"⚠️ Quality flags ({len(quality_flags)})", expanded=False):
            for flag in quality_flags:
                st.warning(flag)

    attrs_with_conf = full_result.get("attributes", {})
    if attrs_with_conf:
        with st.expander("📊 Extraction confidence scores", expanded=False):
            conf_rows = [
                {
                    "Attribute": attr_name,
                    "Extracted Value": entry.get("value", "") if isinstance(entry, dict) else entry,
                    "Confidence %": entry.get("confidence", 0) if isinstance(entry, dict) else 0,
                    "Source excerpt": (entry.get("source_excerpt") or "") if isinstance(entry, dict) else "",
                }
                for attr_name, entry in attrs_with_conf.items()
            ]
            st.dataframe(
                pd.DataFrame(conf_rows),
                use_container_width=True,
                hide_index=True,
            )

    # ── Editable attribute table ──────────────────────────────────────────────
    attrs = record.get("attributes", {})
    if not attrs:
        attrs = {a: "" for a in ATTRIBUTES}
        st.session_state.review_records[pid]["attributes"] = attrs

    attrs_df = pd.DataFrame(
        [{"Attribute": k, "Value": str(v) if v is not None else ""} for k, v in attrs.items()]
    )

    st.markdown("##### Product Attributes")
    edited_df: pd.DataFrame = st.data_editor(
        attrs_df,
        key=f"editor_{pid}",
        use_container_width=True,
        hide_index=True,
        column_config={
            "Attribute": st.column_config.TextColumn("Attribute", disabled=True, width="medium"),
            "Value":     st.column_config.TextColumn("Value", width="large"),
        },
        num_rows="fixed",
    )

    # ── Reviewer comments ─────────────────────────────────────────────────────
    comments_key = f"comments_{pid}"
    # Pre-populate session state on first encounter so we don't fight with value=
    if comments_key not in st.session_state:
        st.session_state[comments_key] = record.get("reviewer_comments", "")

    comments = st.text_area(
        "Reviewer comments",
        key=comments_key,
        height=80,
        placeholder="Optional notes for this product…",
    )

    # ── Action buttons ────────────────────────────────────────────────────────
    st.markdown("**Review actions**")
    bc1, bc2, bc3, bc4 = st.columns(4)

    def _commit(new_status: str | None = None) -> None:
        """Write edited_df + comments into review_records and persist to CSV."""
        updated_attrs = {row["Attribute"]: row["Value"] for _, row in edited_df.iterrows()}
        rec = st.session_state.review_records[pid]
        rec["attributes"]        = updated_attrs
        rec["reviewer_comments"] = comments
        if new_status is not None:
            rec["review_status"] = new_status
        save_reviewed_csv(st.session_state.review_records)

    with bc1:
        if st.button("💾 Save", use_container_width=True, key=f"btn_save_{pid}"):
            _commit()
            st.toast(f"Saved {pid}")

    with bc2:
        if st.button("✅ Approve", use_container_width=True, key=f"btn_approve_{pid}"):
            _commit(new_status="Approved")
            # Auto-advance
            if idx < len(product_ids) - 1:
                st.session_state.current_idx = idx + 1
            st.rerun()

    with bc3:
        if st.button("✏️ Mark Edited", use_container_width=True, key=f"btn_edited_{pid}"):
            _commit(new_status="Edited")
            st.toast(f"Marked {pid} as Edited")
            st.rerun()

    with bc4:
        if st.button("❌ Reject", use_container_width=True, key=f"btn_reject_{pid}"):
            _commit(new_status="Rejected")
            # Auto-advance
            if idx < len(product_ids) - 1:
                st.session_state.current_idx = idx + 1
            st.rerun()


# ── Export bar ─────────────────────────────────────────────────────────────────

def render_export_bar() -> None:
    if not st.session_state.product_ids:
        return

    st.divider()
    st.markdown("#### Export")
    ec1, ec2 = st.columns(2)

    metrics      = compute_metrics(st.session_state.review_records)
    n_exportable = metrics["Approved"] + metrics["Edited"]

    with ec1:
        if st.button(
            f"Export {n_exportable} Approved / Edited records",
            use_container_width=True,
            key="btn_export_approved",
        ):
            df = export_approved_csv(st.session_state.review_records)
            if not df.empty:
                st.success(f"Exported {len(df)} records → {APPROVED_CSV_PATH.name}")
            else:
                st.warning("No Approved or Edited records to export yet.")

    with ec2:
        if REVIEWED_CSV_PATH.exists():
            with open(REVIEWED_CSV_PATH, "rb") as fh:
                st.download_button(
                    "⬇ Download reviewed_output.csv",
                    data=fh.read(),
                    file_name="reviewed_output.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="btn_download_reviewed",
                )


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="MDM_RAG · Review",
        page_icon="🔍",
        layout="wide",
    )

    _init_ss()

    # Auto-load reviewed_output.csv on the very first run
    if not st.session_state.auto_loaded:
        st.session_state.auto_loaded = True
        if REVIEWED_CSV_PATH.exists():
            records = load_reviewed_csv()
            if records:
                st.session_state.review_records.update(records)
                if not st.session_state.product_ids:
                    st.session_state.product_ids = sorted(records.keys())

    st.title("🔍 Product Attribute Extraction Review")

    render_sidebar()
    render_metrics()
    render_product_card()
    render_export_bar()


main()
