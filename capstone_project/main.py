"""
main.py
--------
Streamlit entry point for the Compliance Advisory & Triage Agent.

Pages:
  🏠 Advisor     — Main Q&A interface
  📋 Audit Log   — Full audit history with download
  🧪 Evaluation  — Built-in PASS/FAIL test suite

Run with:
    streamlit run main.py

Upgrade notes (v3.0):
  * Debug Mode sidebar toggle — shows retrieval chunks, scores, query
    expansions, and final LLM context block after each query.
  * Embedding model auto-rebuild — if the DB was built with a different
    model, it is deleted and rebuilt automatically on startup.
  * retrieve() now returns (chunks, debug_info) — debug_info flows through
    the agent into ComplianceAnswer.retrieval_debug for the debug panel.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on PYTHONPATH when running via `streamlit run`
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

# ── Page config — must be the very first Streamlit call ────────
st.set_page_config(
    page_title="Compliance Advisory & Triage Agent",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Application imports (after page config) ────────────────────
from app.agents.compliance_agent import run_compliance_agent
from app.audit.logger import get_audit_stats, get_json_for_download, load_audit_log, log_answer
from app.rag.chunker import split_documents
from app.rag.embeddings import _DEFAULT_MODEL as _EMBEDDING_MODEL
from app.rag.loader import LoadedDocumentInfo, load_policies
from app.rag.retriever import (
    build_vectorstore,
    load_vectorstore,
    rebuild_vectorstore,
    vectorstore_exists,
)
from app.ui.components import (
    EXAMPLE_PROMPTS,
    inject_theme_css,
    render_answer_card,
    render_audit_table,
    render_sidebar,
    render_stats_dashboard,
)
from app.ui.evaluation import render_evaluation_page
from app.utils.config import get_settings


# ──────────────────────────────────────────────────────────────
# Session-state initialisation
# ──────────────────────────────────────────────────────────────

def _init_session_state() -> None:
    defaults: dict = {
        "theme": "dark",
        "vectorstore": None,
        "doc_infos": None,      # List[LoadedDocumentInfo] after first build
        "last_answer": None,
        "question_input": "",
        "debug_mode": False,    # Show retrieval debug panel
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ──────────────────────────────────────────────────────────────
# Vector store bootstrap  (cached as a resource)
# ──────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _cached_load_or_build():
    """
    Load an existing ChromaDB or build one from the policy PDFs.

    Auto-rebuilds when the embedding model used to build the existing DB
    differs from the currently active model. This is detected via a
    ``.model`` sentinel file written alongside the DB.

    Returns
    -------
    (vectorstore, doc_infos)
        ``vectorstore`` — Chroma instance (or None if no PDFs found)
        ``doc_infos``   — list of LoadedDocumentInfo for the sidebar table
    """
    import shutil
    from app.rag.loader import get_policy_infos

    cfg = get_settings()
    sentinel_path = cfg.chroma_persist_dir / ".model"
    current_model = _EMBEDDING_MODEL

    # Detect model mismatch with existing DB
    needs_rebuild = False
    if vectorstore_exists():
        if sentinel_path.exists():
            stored_model = sentinel_path.read_text().strip()
            if stored_model != current_model:
                st.toast(
                    f"⚙️ Embedding model changed ({stored_model} → {current_model}). "
                    "Rebuilding knowledge base…",
                    icon="🔄",
                )
                needs_rebuild = True
        else:
            # No sentinel — DB pre-dates this feature; rebuild to be safe
            needs_rebuild = True

    # Wipe stale DB if needed
    if needs_rebuild and cfg.chroma_persist_dir.exists():
        shutil.rmtree(cfg.chroma_persist_dir, ignore_errors=True)

    # If DB exists and model matches, just load it
    if vectorstore_exists() and not needs_rebuild:
        vs = load_vectorstore()
        infos = get_policy_infos()   # now returns real page counts via pypdf
        for info in infos:
            info.embedding_status = "✓ Done"

        # Populate chunk counts from db_stats.json written during build
        import json as _json
        stats_path = cfg.chroma_persist_dir / "db_stats.json"
        if stats_path.exists():
            try:
                db_stats = _json.loads(stats_path.read_text())
                chunk_counts = db_stats.get("chunk_counts", {})
                for info in infos:
                    info.chunks = chunk_counts.get(info.filename, 0)
            except Exception:
                pass   # non-fatal — sidebar will show 0 for chunks only

        # Ensure sentinel exists so future restarts don't trigger a rebuild
        if not sentinel_path.exists():
            cfg.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
            sentinel_path.write_text(current_model)

        return vs, infos

    # Build from scratch
    docs, infos = load_policies()
    if not docs:
        return None, infos

    chunks, chunk_counts = split_documents(docs)
    for info in infos:
        info.chunks = chunk_counts.get(info.filename, 0)

    vs = build_vectorstore(chunks)

    # Write sentinel so future loads know which model built this DB
    cfg.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
    sentinel_path.write_text(current_model)

    return vs, infos


def _force_rebuild() -> tuple:
    """
    Clear the Streamlit cache and rebuild the vector store from scratch.
    Called when the user clicks "Rebuild Knowledge Base".
    """
    _cached_load_or_build.clear()   # bust the @cache_resource

    with st.spinner("🔄 Rebuilding knowledge base — this may take a minute …"):
        vs, infos = rebuild_vectorstore()

    # Re-write sentinel after manual rebuild
    if vs is not None:
        cfg = get_settings()
        sentinel_path = cfg.chroma_persist_dir / ".model"
        cfg.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        sentinel_path.write_text(_EMBEDDING_MODEL)

    return vs, infos


# ──────────────────────────────────────────────────────────────
# Debug panel
# ──────────────────────────────────────────────────────────────

def _render_debug_panel(answer) -> None:
    """
    Show a detailed RAG retrieval debug panel beneath the answer card.
    Only rendered when Debug Mode is enabled in the sidebar.

    Displays:
      - Original query + any expansion queries used
      - Retrieval threshold value
      - All raw retrieved chunks with scores (before OOC filter)
      - Relevant chunks that passed the OOC threshold (sent to LLM)
      - Rejected chunks that were below the OOC threshold
      - Why refusal happened (if applicable)
      - Why escalation happened / didn't happen
      - Why this risk level was assigned
      - Full LLM context block
      - Vector database diagnostics
    """
    from app.utils.models import ComplianceAnswer

    if not isinstance(answer, ComplianceAnswer):
        return

    with st.expander("🔬 Debug Mode — Retrieval & Agent Details", expanded=True):

        # ── Query expansion ────────────────────────────────────
        debug = getattr(answer, "retrieval_debug", None)
        agent_debug = getattr(answer, "agent_debug", None) or {}

        ooc_threshold = agent_debug.get("ooc_threshold", 0.35)

        if debug is not None:
            st.markdown("#### Query Expansion")
            st.markdown(f"**Original query:** `{debug.original_query}`")
            if len(debug.expanded_queries) > 1:
                st.markdown(
                    f"**Expanded to {len(debug.expanded_queries)} queries:**"
                )
                for i, q in enumerate(debug.expanded_queries, 1):
                    st.markdown(f"  {i}. _{q}_")
            else:
                st.caption("No query expansion triggered for this question.")

            st.markdown("---")

            if debug.failure_reason:
                st.error(f"⚠️ **Retrieval failure:** {debug.failure_reason}")

        # ── Retrieval threshold ────────────────────────────────
        st.markdown("#### Retrieval Threshold")
        col_thresh1, col_thresh2 = st.columns(2)
        with col_thresh1:
            st.metric(
                "OOC Threshold",
                f"{ooc_threshold:.2f}",
                help=(
                    "Minimum cosine similarity for a chunk to be considered in-corpus. "
                    "Chunks below this score are rejected and not sent to the LLM."
                ),
            )
        with col_thresh2:
            if debug is not None:
                st.metric(
                    "Passed Threshold",
                    f"{debug.filtered_count} / {len(debug.raw_results)}",
                    help="Number of chunks that scored above the OOC threshold.",
                )

        st.markdown("---")

        # ── Raw retrieved chunks ───────────────────────────────
        if debug is not None:
            st.markdown("#### Raw Retrieved Chunks (before OOC filter)")
            st.caption(
                f"{len(debug.raw_results)} raw results · "
                f"{debug.filtered_count} passed OOC threshold ({ooc_threshold:.2f})"
            )

            if debug.raw_results:
                import pandas as pd
                rows = []
                for rank, (content, score) in enumerate(debug.raw_results, 1):
                    passed = "✅" if score >= ooc_threshold else "❌"
                    bar = "█" * int(score * 20)
                    rows.append({
                        "#": rank,
                        "Score": f"{score:.4f}",
                        "Passed?": passed,
                        "Bar": bar,
                        "Preview (300 chars)": content.replace("\n", " ")[:300],
                    })
                st.dataframe(
                    pd.DataFrame(rows),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.warning("No raw results were returned by the retriever.")

        st.markdown("---")

        # ── Relevant chunks (passed OOC threshold) ────────────
        relevant_chunks = agent_debug.get("relevant_chunks") or answer.sources or []
        st.markdown(f"#### ✅ Relevant Chunks Passed to LLM ({len(relevant_chunks)})")
        if relevant_chunks:
            for i, chunk in enumerate(relevant_chunks, 1):
                score_pct = int(chunk.relevance_score * 100)
                st.markdown(
                    f"**[{i}] {chunk.source}** — "
                    f"Page {chunk.page if chunk.page is not None else '?'} · "
                    f"Score: `{chunk.relevance_score:.4f}` ({score_pct}%) ✅"
                )
                st.progress(chunk.relevance_score)
                st.code(
                    chunk.content[:500] + ("…" if len(chunk.content) > 500 else ""),
                    language=None,
                )
        else:
            st.info(
                "No chunks passed the OOC threshold — refusal was triggered or "
                "the question is outside the loaded corpus."
            )

        st.markdown("---")

        # ── Rejected chunks (below OOC threshold) ─────────────
        rejected_chunks = agent_debug.get("rejected_chunks", [])
        st.markdown(f"#### ❌ Rejected Chunks (below OOC threshold = {ooc_threshold:.2f}) — {len(rejected_chunks)}")
        if rejected_chunks:
            for i, chunk in enumerate(rejected_chunks, 1):
                score_pct = int(chunk.relevance_score * 100)
                st.markdown(
                    f"**[{i}] {chunk.source}** — "
                    f"Page {chunk.page if chunk.page is not None else '?'} · "
                    f"Score: `{chunk.relevance_score:.4f}` ({score_pct}%) ❌"
                )
                st.progress(min(chunk.relevance_score, 1.0))
                st.caption(chunk.content[:200].replace("\n", " ") + "…")
        else:
            st.caption(
                "No chunks were rejected — all retrieved chunks passed the threshold, "
                "or the question was refused (all chunks are treated as rejected)."
            )

        st.markdown("---")

        # ── Refusal reason ─────────────────────────────────────
        st.markdown("#### Why Was the Answer Refused / Allowed?")
        refusal_reason = agent_debug.get("refusal_reason")
        if answer.refused:
            reason_labels = {
                "EMPTY_RETRIEVAL": "🚫 EMPTY_RETRIEVAL — the knowledge base appears to be empty or no chunks were indexed.",
                "OUT_OF_CORPUS": (
                    f"🚫 OUT_OF_CORPUS — the top retrieved chunk scored below the OOC threshold "
                    f"({ooc_threshold:.2f}). The question topic is not covered by the loaded documents."
                ),
                "LOW_SIMILARITY": "🚫 LOW_SIMILARITY — all retrieved chunks scored below the similarity floor.",
            }
            reason_display = reason_labels.get(
                refusal_reason,
                f"🚫 Refused — reason: {refusal_reason or 'unknown'}"
            )
            st.error(reason_display)
        else:
            st.success(
                f"✅ **Not refused** — top chunk scored above OOC threshold ({ooc_threshold:.2f}). "
                f"{len(relevant_chunks)} relevant chunk(s) passed into LLM context."
            )

        st.markdown("---")

        # ── Escalation reason ──────────────────────────────────
        st.markdown("#### Why Was / Wasn't This Escalated?")
        escalation_reason = agent_debug.get("escalation_reason", "")
        if answer.escalated:
            st.warning(f"⚠️ **Escalated** — {escalation_reason}")
        else:
            st.success(f"✅ **Not escalated** — {escalation_reason}")

        st.markdown("---")

        # ── Risk level reasoning ───────────────────────────────
        st.markdown("#### Why This Risk Level?")
        risk_level = agent_debug.get("risk_level") or answer.risk_level
        routing_reasoning = agent_debug.get("routing_reasoning", "No routing reasoning available.")
        bypass_intent = agent_debug.get("bypass_intent", False)
        risk_colors = {"HIGH": "#F44336", "MEDIUM": "#FF9800", "LOW": "#00C853"}
        risk_color = risk_colors.get(risk_level.value if hasattr(risk_level, 'value') else str(risk_level), "#888")
        st.markdown(
            f"**Risk Level:** "
            f"<span style='color:{risk_color}; font-weight:700;'>"
            f"{risk_level.value if hasattr(risk_level, 'value') else risk_level}</span>",
            unsafe_allow_html=True,
        )
        if bypass_intent:
            st.warning("🔴 Bypass intent override: question asked to ignore/circumvent a compliance control → forced to HIGH.")
        st.caption(f"Routing agent reasoning: {routing_reasoning}")

        st.markdown("---")

        # ── Final context sent to LLM ──────────────────────────
        st.markdown("#### Final Context Sent to LLM")
        if relevant_chunks:
            context_parts = []
            for i, chunk in enumerate(relevant_chunks[:5], 1):
                context_parts.append(
                    f"[Source {i}: {chunk.source}, Page {chunk.page}]\n"
                    f"{chunk.content}"
                )
            full_context = "\n\n---\n\n".join(context_parts)
            st.text_area(
                label="LLM context block",
                value=full_context,
                height=300,
                disabled=True,
                label_visibility="collapsed",
            )
        else:
            st.caption(
                "No context was sent — agent refused or no relevant chunks retrieved."
            )

        st.markdown("---")

        # ── Vector Database Diagnostics ────────────────────────
        st.markdown("#### Vector Database Diagnostics")
        from app.rag.retriever import get_db_stats
        db = get_db_stats()

        diag_rows = {
            "Persist Path":      db["persist_path"],
            "Collection Name":   db["collection_name"],
            "Collection Exists": "Yes" if db["exists"] else "No",
            "Embedding Count":   f"{db['embedding_count']:,}",
            "Embedding Model":   db["embedding_model"],
            "Database Status":   db["status"].upper(),
        }
        if db.get("all_collections"):
            diag_rows["All Collections"] = ", ".join(
                f"{c['name']} ({c['count']})" for c in db["all_collections"]
            )

        import pandas as pd
        diag_df = pd.DataFrame(
            [{"Setting": k, "Value": v} for k, v in diag_rows.items()]
        )
        st.dataframe(diag_df, use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────────────────────
# Page: Advisor
# ──────────────────────────────────────────────────────────────

def page_advisor(vectorstore) -> None:
    """Main compliance Q&A page."""

    st.markdown(
        """
        <div style="margin-bottom: 1.5rem;">
            <h1 style="margin-bottom: 0.2rem;">⚖️ Compliance Advisory & Triage Agent</h1>
            <p style="opacity: 0.65; font-size: 1rem;">
                Ask any compliance question. Answers are sourced exclusively from
                your organisation's policy documents.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Stats dashboard (only shown after the first query)
    stats = get_audit_stats()
    if stats["total"] > 0:
        render_stats_dashboard(stats)
        st.divider()

    # No documents guard
    if vectorstore is None:
        st.error(
            "⚠️ **No policy documents found or vector store could not be built.**\n\n"
            "Add PDF files to `data/policies/` then click **Rebuild Knowledge Base** "
            "in the sidebar."
        )
        return

    # Example prompts
    with st.expander("💡 Example Questions", expanded=False):
        cols = st.columns(3)
        for i, prompt in enumerate(EXAMPLE_PROMPTS):
            with cols[i % 3]:
                if st.button(prompt, key=f"ex_{i}", use_container_width=True):
                    st.session_state["question_input"] = prompt
                    st.rerun()

    st.divider()

    # Query input
    question = st.text_area(
        label="Your compliance question",
        value=st.session_state.get("question_input", ""),
        placeholder="e.g. What are the data retention requirements under GDPR?",
        height=100,
        key="query_area",
        label_visibility="collapsed",
    )

    col_btn, col_clear = st.columns([2, 1])
    with col_btn:
        ask_clicked = st.button(
            "🔍 Ask Compliance Agent",
            type="primary",
            use_container_width=True,
            disabled=(not question or not question.strip()),
        )
    with col_clear:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state["question_input"] = ""
            st.session_state["last_answer"] = None
            st.rerun()

    # Process query
    if ask_clicked and question.strip():
        with st.spinner("🔎 Searching policy documents …"):
            try:
                answer = run_compliance_agent(question.strip(), vectorstore)
                log_answer(answer)
                st.session_state["last_answer"] = answer
                st.session_state["question_input"] = question
            except Exception as exc:
                st.error(f"Agent error: {exc}")
                st.stop()

    # Render answer + optional debug panel
    if st.session_state.get("last_answer") is not None:
        st.markdown("---")
        render_answer_card(st.session_state["last_answer"])
        if st.session_state.get("debug_mode", False):
            _render_debug_panel(st.session_state["last_answer"])

    # Recent history preview
    recent = load_audit_log(limit=5)
    if recent and len(recent) > 1:
        with st.expander("🕐 Recent Queries", expanded=False):
            for rec in recent[1:4]:
                col_q, col_r, col_c = st.columns([4, 1, 1])
                with col_q:
                    st.caption(rec.question[:100])
                with col_r:
                    badge_color = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(
                        rec.risk_level, "⚪"
                    )
                    st.caption(f"{badge_color} {rec.risk_level}")
                with col_c:
                    st.caption(f"{rec.confidence * 100:.0f}%")


# ──────────────────────────────────────────────────────────────
# Page: Audit Log
# ──────────────────────────────────────────────────────────────

def page_audit_log() -> None:
    """Full audit history page."""

    st.markdown("## 📋 Audit History")
    st.markdown(
        "Complete record of all compliance queries. "
        "Records are stored in both JSON and SQLite format."
    )

    stats = get_audit_stats()
    render_stats_dashboard(stats)
    st.divider()

    col_filter, col_download = st.columns([3, 1])
    with col_filter:
        filter_risk = st.selectbox(
            "Filter by Risk Level",
            options=["All", "HIGH", "MEDIUM", "LOW"],
            index=0,
        )
    with col_download:
        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
        json_data = get_json_for_download()
        st.download_button(
            label="⬇️ Download Audit Log",
            data=json_data,
            file_name="compliance_audit_log.json",
            mime="application/json",
            use_container_width=True,
        )

    st.divider()

    records = load_audit_log(limit=200)
    if filter_risk != "All":
        records = [r for r in records if r.risk_level == filter_risk]

    st.markdown(f"**{len(records)} records**")
    render_audit_table(records, max_rows=50)


# ──────────────────────────────────────────────────────────────
# Main application
# ──────────────────────────────────────────────────────────────

def main() -> None:
    _init_session_state()

    # Theme CSS (injected before any visible elements)
    inject_theme_css()

    # ── Bootstrap vector store ─────────────────────────────────
    if st.session_state["vectorstore"] is None:
        with st.spinner("⚙️ Loading policy documents …"):
            vs, infos = _cached_load_or_build()
            st.session_state["vectorstore"] = vs
            st.session_state["doc_infos"] = infos

    vectorstore = st.session_state["vectorstore"]
    doc_infos: list[LoadedDocumentInfo] = st.session_state.get("doc_infos") or []

    # ── Sidebar (returns True if Rebuild was clicked) ──────────
    rebuild_clicked = render_sidebar(doc_infos=doc_infos)

    if rebuild_clicked:
        vs, infos = _force_rebuild()
        st.session_state["vectorstore"] = vs
        st.session_state["doc_infos"] = infos
        st.session_state["last_answer"] = None
        if vs is not None:
            st.success(
                f"✅ Knowledge base rebuilt — {len(infos)} document(s) indexed."
            )
        else:
            st.warning(
                "⚠️ No PDF files found in `data/policies/`. Nothing was indexed."
            )
        st.rerun()

    # ── Navigation + Developer Tools ──────────────────────────
    with st.sidebar:
        st.markdown("---")
        st.markdown("**Navigation**")
        page = st.radio(
            label="page",
            options=["🏠 Advisor", "📋 Audit Log", "🧪 Evaluation"],
            index=0,
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.markdown("**🔬 Developer Tools**")
        debug_on = st.toggle(
            "Debug Mode",
            value=st.session_state.get("debug_mode", False),
            help=(
                "After each query, show: retrieved chunks, similarity scores, "
                "source PDF, page number, query expansions, and the final "
                "context block sent to the LLM."
            ),
        )
        if debug_on != st.session_state.get("debug_mode", False):
            st.session_state["debug_mode"] = debug_on
            st.rerun()

    # ── Route to selected page ─────────────────────────────────
    if page == "🏠 Advisor":
        page_advisor(vectorstore)
    elif page == "📋 Audit Log":
        page_audit_log()
    elif page == "🧪 Evaluation":
        render_evaluation_page(vectorstore)


if __name__ == "__main__":
    main()
