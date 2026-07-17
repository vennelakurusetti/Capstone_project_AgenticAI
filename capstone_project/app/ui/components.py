"""
app/ui/components.py
---------------------
Reusable Streamlit UI components for the Compliance Advisory & Triage Agent.

Components:
  * render_sidebar() — document status table, vector DB status, session stats
  * render_answer_card() — formats a ComplianceAnswer for display
  * render_audit_table() — paginated audit log
  * render_stats_dashboard() — metric tiles
  * Dark / Light theme toggle

Theme:
  Dark  — background #0D1117, secondary #161B22, text #E6EDF3
  Light — background #FFFFFF, secondary #F6F8FA, text #1F2328
  Accent: teal (#00BCD4), blue (#1565C0)
"""

from __future__ import annotations

from typing import List, Optional

import streamlit as st

from app.audit.logger import get_audit_stats, load_audit_log
from app.governance.confidence import confidence_color, confidence_label
from app.governance.escalation import escalation_banner_text
from app.rag.loader import LoadedDocumentInfo, get_policy_infos
from app.rag.retriever import get_collection_count, vectorstore_exists
from app.utils.models import AuditRecord, ComplianceAnswer, RiskLevel


# ──────────────────────────────────────────────────────────────
# Theme management
# ──────────────────────────────────────────────────────────────

def _dark_css() -> str:
    return """
    <style>
    :root {
        --bg-primary:    #0D1117;
        --bg-secondary:  #161B22;
        --bg-card:       #1C2128;
        --border:        #30363D;
        --text-primary:  #E6EDF3;
        --text-muted:    #8B949E;
        --accent-teal:   #00BCD4;
        --accent-blue:   #388BFD;
        --risk-low:      #00C853;
        --risk-med:      #FF9800;
        --risk-high:     #F44336;
    }
    .stApp { background-color: var(--bg-primary) !important; color: var(--text-primary); }
    section[data-testid="stSidebar"] { background-color: var(--bg-secondary) !important; }
    .stTextInput > div > div > input,
    .stTextArea textarea {
        background-color: var(--bg-card) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px;
    }
    .stButton > button {
        background: linear-gradient(135deg, #00BCD4, #1565C0);
        color: white; border: none; border-radius: 8px;
        padding: 0.5rem 1.5rem; font-weight: 600; transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }
    .compliance-card {
        background: var(--bg-card); border: 1px solid var(--border);
        border-radius: 12px; padding: 1.5rem; margin: 1rem 0;
    }
    .answer-text { font-size: 1.05rem; line-height: 1.7; color: var(--text-primary); }
    .badge {
        display: inline-block; padding: 0.25rem 0.75rem; border-radius: 20px;
        font-size: 0.78rem; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase;
    }
    .badge-low    { background: #00C85322; color: #00C853; border: 1px solid #00C853; }
    .badge-medium { background: #FF980022; color: #FF9800; border: 1px solid #FF9800; }
    .badge-high   { background: #F4433622; color: #F44336; border: 1px solid #F44336; }
    .escalation-banner {
        background: #F4433615; border: 1px solid #F44336;
        border-left: 4px solid #F44336; border-radius: 8px;
        padding: 1rem 1.2rem; margin: 0.75rem 0; color: #F44336;
    }
    .metric-tile {
        background: var(--bg-card); border: 1px solid var(--border);
        border-radius: 10px; padding: 1rem; text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: var(--accent-teal); }
    .metric-label { font-size: 0.8rem; color: var(--text-muted); text-transform: uppercase; }
    .status-dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; }
    .dot-green  { background: #00C853; }
    .dot-red    { background: #F44336; }
    .dot-orange { background: #FF9800; }
    /* Doc table in sidebar */
    .doc-table { width: 100%; border-collapse: collapse; font-size: 0.75rem; }
    .doc-table th {
        background: #0D1117; color: #8B949E; text-transform: uppercase;
        font-size: 0.65rem; letter-spacing: 0.06em; padding: 0.35rem 0.4rem;
        border-bottom: 1px solid #30363D; text-align: left;
    }
    .doc-table td {
        padding: 0.3rem 0.4rem; border-bottom: 1px solid #21262D;
        color: #E6EDF3; vertical-align: top; word-break: break-all;
    }
    .doc-table tr:hover td { background: #21262D; }
    .tag { display: inline-block; padding: 0.1rem 0.4rem; border-radius: 4px;
           font-size: 0.65rem; font-weight: 600; background: #00BCD422;
           color: #00BCD4; border: 1px solid #00BCD466; }
    .stProgress > div > div > div { background: var(--accent-teal) !important; }
    .source-chip {
        display: inline-block; background: var(--bg-secondary);
        border: 1px solid var(--border); border-radius: 6px;
        padding: 0.2rem 0.6rem; margin: 0.2rem; font-size: 0.78rem;
        color: var(--accent-teal);
    }
    hr { border-color: var(--border) !important; }
    .stDataFrame { background: var(--bg-card) !important; }
    </style>
    """


def _light_css() -> str:
    return """
    <style>
    :root {
        --bg-primary:   #FFFFFF; --bg-secondary: #F6F8FA;
        --bg-card:      #F0F4F8; --border: #D0D7DE;
        --text-primary: #1F2328; --text-muted: #656D76;
        --accent-teal:  #0969DA; --accent-blue: #1565C0;
    }
    .stApp { background-color: var(--bg-primary) !important; color: var(--text-primary); }
    section[data-testid="stSidebar"] { background-color: var(--bg-secondary) !important; }
    .stTextInput > div > div > input,
    .stTextArea textarea {
        background-color: var(--bg-card) !important; color: var(--text-primary) !important;
        border: 1px solid var(--border) !important; border-radius: 8px;
    }
    .stButton > button {
        background: linear-gradient(135deg, #0969DA, #1565C0);
        color: white; border: none; border-radius: 8px; padding: 0.5rem 1.5rem; font-weight: 600;
    }
    .compliance-card {
        background: var(--bg-card); border: 1px solid var(--border);
        border-radius: 12px; padding: 1.5rem; margin: 1rem 0;
    }
    .answer-text { font-size: 1.05rem; line-height: 1.7; color: var(--text-primary); }
    .badge { display: inline-block; padding: 0.25rem 0.75rem; border-radius: 20px;
             font-size: 0.78rem; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase; }
    .badge-low    { background: #E6F9ED; color: #1B7A3A; border: 1px solid #1B7A3A; }
    .badge-medium { background: #FFF3E0; color: #B45309; border: 1px solid #B45309; }
    .badge-high   { background: #FEECEC; color: #C62828; border: 1px solid #C62828; }
    .escalation-banner {
        background: #FEECEC; border: 1px solid #C62828; border-left: 4px solid #C62828;
        border-radius: 8px; padding: 1rem 1.2rem; margin: 0.75rem 0; color: #C62828;
    }
    .metric-tile { background: var(--bg-card); border: 1px solid var(--border);
                   border-radius: 10px; padding: 1rem; text-align: center; }
    .metric-value { font-size: 2rem; font-weight: 700; color: var(--accent-blue); }
    .metric-label { font-size: 0.8rem; color: var(--text-muted); text-transform: uppercase; }
    .status-dot { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; }
    .dot-green { background:#1B7A3A; } .dot-red { background:#C62828; } .dot-orange { background:#B45309; }
    .doc-table { width:100%; border-collapse:collapse; font-size:0.75rem; }
    .doc-table th { background:#F6F8FA; color:#656D76; text-transform:uppercase; font-size:0.65rem;
                    letter-spacing:0.06em; padding:0.35rem 0.4rem; border-bottom:1px solid #D0D7DE; text-align:left; }
    .doc-table td { padding:0.3rem 0.4rem; border-bottom:1px solid #E1E4E8;
                    color:#1F2328; vertical-align:top; word-break:break-all; }
    .tag { display:inline-block; padding:0.1rem 0.4rem; border-radius:4px; font-size:0.65rem;
           font-weight:600; background:#0969DA22; color:#0969DA; border:1px solid #0969DA66; }
    .source-chip { display:inline-block; background:var(--bg-secondary); border:1px solid var(--border);
                   border-radius:6px; padding:0.2rem 0.6rem; margin:0.2rem; font-size:0.78rem; color:var(--accent-blue); }
    </style>
    """


def inject_theme_css() -> None:
    """Inject dark or light CSS based on session state."""
    if st.session_state.get("theme", "dark") == "dark":
        st.markdown(_dark_css(), unsafe_allow_html=True)
    else:
        st.markdown(_light_css(), unsafe_allow_html=True)


def render_theme_toggle() -> None:
    """Sidebar dark/light mode toggle button."""
    current = st.session_state.get("theme", "dark")
    icon = "☀️" if current == "dark" else "🌙"
    label = f"{icon} {'Light Mode' if current == 'dark' else 'Dark Mode'}"
    if st.sidebar.button(label, key="theme_toggle", use_container_width=True):
        st.session_state["theme"] = "light" if current == "dark" else "dark"
        st.rerun()


# ──────────────────────────────────────────────────────────────
# Document status table (sidebar)
# ──────────────────────────────────────────────────────────────

def _doc_table_html(infos: List[LoadedDocumentInfo]) -> str:
    """Build an HTML table of loaded documents for the sidebar."""
    rows_html = ""
    for info in infos:
        status_icon = (
            "✓" if info.embedding_status == "✓ Done"
            else ("✗" if info.embedding_status == "✗ Failed" else "⏳")
        )
        status_color = (
            "#00C853" if status_icon == "✓"
            else ("#F44336" if status_icon == "✗" else "#FF9800")
        )
        # Truncate long filenames
        name = info.filename
        if len(name) > 30:
            name = name[:27] + "…"

        pages_str = str(info.pages) if info.pages > 0 else "—"
        chunks_str = str(info.chunks) if info.chunks > 0 else "—"

        rows_html += f"""
        <tr>
          <td title="{info.filename}">{name}</td>
          <td><span class="tag">{info.document_type}</span></td>
          <td style="text-align:center">{pages_str}</td>
          <td style="text-align:center">{chunks_str}</td>
          <td style="color:{status_color}; font-weight:700; text-align:center">{status_icon}</td>
        </tr>"""

    return f"""
    <table class="doc-table">
      <thead>
        <tr>
          <th>Filename</th>
          <th>Type</th>
          <th>Pg</th>
          <th>Ch</th>
          <th>✓</th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
    """


# ──────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────

def render_sidebar(
    doc_infos: Optional[List[LoadedDocumentInfo]] = None,
) -> None:
    """
    Render the full sidebar.

    Parameters
    ----------
    doc_infos:
        If provided (after a full ingest), shows live page/chunk counts.
        If None, fetches lightweight info (filename + type only).
    """
    with st.sidebar:
        # ── Logo ───────────────────────────────────────────────
        st.markdown(
            """
            <div style="text-align:center; padding: 1rem 0 0.5rem;">
                <div style="font-size:2rem;">⚖️</div>
                <div style="font-size:1.1rem; font-weight:700; letter-spacing:0.03em;">
                    Compliance Agent
                </div>
                <div style="font-size:0.75rem; opacity:0.6; margin-top:0.2rem;">
                    Advisory & Triage System
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()

        # ── Theme toggle ───────────────────────────────────────
        render_theme_toggle()
        st.divider()

        # ── Document status table ──────────────────────────────
        st.markdown("**📄 Documents Loaded**")

        infos = doc_infos if doc_infos is not None else get_policy_infos()

        if infos:
            st.markdown(_doc_table_html(infos), unsafe_allow_html=True)
            st.caption(f"{len(infos)} document(s) discovered")
        else:
            st.markdown(
                '<span class="status-dot dot-red"></span>'
                'No PDFs found — add files to `data/policies/`',
                unsafe_allow_html=True,
            )

        st.divider()

        # ── Vector DB status ───────────────────────────────────
        st.markdown("**🗄️ Vector Database**")
        if vectorstore_exists():
            from app.rag.retriever import get_db_stats
            stats = get_db_stats()
            emb_count = stats["embedding_count"]
            status = stats["status"]

            if status == "ready":
                # Compute aggregate doc / page / chunk counts from infos
                total_pages  = sum(i.pages  for i in (infos or []))
                total_chunks = sum(i.chunks for i in (infos or []))

                st.markdown(
                    '<span class="status-dot dot-green"></span>**Ready**',
                    unsafe_allow_html=True,
                )
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Documents", len(infos) if infos else "—")
                    st.metric("Pages", total_pages if total_pages else "—")
                with col_b:
                    st.metric("Chunks", total_chunks if total_chunks else "—")
                    st.metric("Embeddings", f"{emb_count:,}")
            elif status == "empty":
                st.markdown(
                    '<span class="status-dot dot-orange"></span>'
                    'DB exists but collection is empty — click Rebuild.',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<span class="status-dot dot-red"></span>'
                    f'DB error — status: {status}',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<span class="status-dot dot-orange"></span>'
                'Not built — run `python ingest.py` first',
                unsafe_allow_html=True,
            )

        st.divider()

        # ── Audit statistics ───────────────────────────────────
        st.markdown("**📊 Session Stats**")
        stats = get_audit_stats()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Queries", stats["total"])
            st.metric("Escalated", stats["escalated"])
        with col2:
            st.metric("High Risk", stats["high_risk"])
            st.metric("Refused", stats["refused"])

        if stats["total"] > 0:
            st.metric("Avg Confidence", f"{stats['avg_confidence'] * 100:.0f}%")

        st.divider()
        st.caption("v2.0 · Compliance Advisory Agent")


# ──────────────────────────────────────────────────────────────
# Answer card
# ──────────────────────────────────────────────────────────────

def _risk_badge(risk: RiskLevel) -> str:
    css_class = {
        RiskLevel.LOW: "badge-low",
        RiskLevel.MEDIUM: "badge-medium",
        RiskLevel.HIGH: "badge-high",
    }.get(risk, "badge-low")
    return f'<span class="badge {css_class}">{risk.value}</span>'


def render_answer_card(answer: ComplianceAnswer) -> None:
    """Render the full compliance answer card with all metadata.

    Section order:
      1. Escalation banner (if applicable)
      2. Topic + Risk badge header
      3. Answer text
      4. Sources (source chips + expander)
      5. Confidence bar
      6. Metadata: Topic · Owner · Risk · Escalation
    """

    if answer.escalated:
        st.markdown(
            f'<div class="escalation-banner">'
            f"{escalation_banner_text(answer.owner.value)}"
            f"</div>",
            unsafe_allow_html=True,
        )

    with st.container():
        st.markdown('<div class="compliance-card">', unsafe_allow_html=True)

        # ── Header: topic + risk badge ─────────────────────────
        col_topic, col_risk = st.columns([3, 1])
        with col_topic:
            st.markdown(f"#### 📋 {answer.topic.value}")
        with col_risk:
            st.markdown(_risk_badge(answer.risk_level), unsafe_allow_html=True)

        st.divider()

        # ── Section 1: Answer ──────────────────────────────────
        st.markdown("**📝 Answer**")
        st.markdown(
            f'<div class="answer-text">{answer.answer}</div>',
            unsafe_allow_html=True,
        )

        st.divider()

        # ── Section 2: Sources ─────────────────────────────────
        if answer.sources:
            st.markdown("**📄 Sources**")
            sources_html = ""
            unique_sources: set[str] = set()
            for chunk in answer.sources:
                key = f"{chunk.source} · p{chunk.page or '?'}"
                if key not in unique_sources:
                    unique_sources.add(key)
                    sources_html += f'<span class="source-chip">📄 {key} ({chunk.relevance_score:.2f})</span>'
            st.markdown(sources_html, unsafe_allow_html=True)

            with st.expander("View source excerpts", expanded=False):
                for i, chunk in enumerate(answer.sources[:5], 1):
                    st.markdown(
                        f"**[{i}] {chunk.source}** — Page {chunk.page or '?'} "
                        f"(score: {chunk.relevance_score:.3f})"
                    )
                    st.code(chunk.content[:400] + ("…" if len(chunk.content) > 400 else ""))

        elif answer.refused:
            st.markdown("**📄 Sources**")
            st.caption("No sources — question was outside the available corpus.")

        st.divider()

        # ── Section 3: Confidence ──────────────────────────────
        conf_pct = answer.confidence
        label = confidence_label(conf_pct)
        color = confidence_color(conf_pct)

        st.markdown("**📊 Confidence**")
        col_conf_label, col_conf_val = st.columns([4, 1])
        with col_conf_label:
            st.markdown(f"_{label}_")
            st.progress(conf_pct)
        with col_conf_val:
            st.markdown(
                f"<div style='color:{color}; font-size:1.4rem; font-weight:700; "
                f"text-align:right; padding-top:0.3rem;'>{conf_pct*100:.0f}%</div>",
                unsafe_allow_html=True,
            )

        if answer.requires_human_review and not answer.refused:
            st.info(
                "⚠️ Confidence is below threshold. Human review is recommended "
                f"before acting on this guidance. Contact: **{answer.owner.value}**"
            )

        st.divider()

        # ── Section 4: Metadata ────────────────────────────────
        meta_col1, meta_col2, meta_col3, meta_col4 = st.columns(4)
        with meta_col1:
            st.markdown(f"**🎯 Topic**  \n{answer.topic.value}")
        with meta_col2:
            st.markdown(f"**👤 Owner**  \n{answer.owner.value}")
        with meta_col3:
            risk_icons = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
            risk_icon = risk_icons.get(answer.risk_level.value, "⚪")
            st.markdown(f"**⚡ Risk**  \n{risk_icon} {answer.risk_level.value}")
        with meta_col4:
            escalated_text = "🔴 Yes — Escalated" if answer.escalated else "🟢 No"
            st.markdown(f"**⚠️ Escalation**  \n{escalated_text}")

        st.markdown("</div>", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# Example prompts
# ──────────────────────────────────────────────────────────────

EXAMPLE_PROMPTS: List[str] = [
    "What is the retention period for customer records?",
    "Can we store EU customer data in the US?",
    "What are customer due diligence requirements?",
    "Can we ignore GDPR for one client?",
    "What password length is required?",
    "What are the incident response steps for a data breach?",
    "What are the requirements for onboarding a new vendor?",
]


def render_example_prompts(on_select_callback) -> None:
    """Render clickable example prompt pills."""
    st.markdown("**💡 Example Questions**")
    cols = st.columns(min(len(EXAMPLE_PROMPTS), 3))
    for i, prompt in enumerate(EXAMPLE_PROMPTS):
        with cols[i % 3]:
            if st.button(prompt, key=f"example_{i}", use_container_width=True, help=prompt):
                on_select_callback(prompt)


# ──────────────────────────────────────────────────────────────
# Audit history table
# ──────────────────────────────────────────────────────────────

def render_audit_table(records: List[AuditRecord], max_rows: int = 20) -> None:
    """Render a styled audit history table."""
    if not records:
        st.info("No audit records yet. Ask a compliance question to get started.")
        return

    import pandas as pd

    rows = []
    for rec in records[:max_rows]:
        rows.append({
            "Timestamp": rec.timestamp[:19].replace("T", " "),
            "Question": rec.question[:80] + ("…" if len(rec.question) > 80 else ""),
            "Topic": rec.topic,
            "Risk": rec.risk_level,
            "Confidence": f"{rec.confidence * 100:.0f}%",
            "Owner": rec.owner,
            "Escalated": "⚠️ Yes" if rec.escalated else "✅ No",
            "Refused": "🚫 Yes" if rec.refused else "✅ No",
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────────────────────
# Stats dashboard
# ──────────────────────────────────────────────────────────────

def render_stats_dashboard(stats: dict) -> None:
    """Render four metric tiles for the dashboard header."""
    col1, col2, col3, col4 = st.columns(4)
    tiles = [
        ("Total Queries", stats["total"], "🔍"),
        ("Escalated", stats["escalated"], "⚠️"),
        ("High Risk", stats["high_risk"], "🔴"),
        ("Avg Confidence", f"{stats['avg_confidence'] * 100:.0f}%", "📊"),
    ]
    for col, (label, value, icon) in zip([col1, col2, col3, col4], tiles):
        with col:
            st.markdown(
                f"""
                <div class="metric-tile">
                    <div style="font-size:1.5rem;">{icon}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-label">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
