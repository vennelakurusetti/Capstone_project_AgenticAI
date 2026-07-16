"""
app/agents/compliance_agent.py
-------------------------------
Main compliance Q&A agent implemented as a LangGraph StateGraph.

Graph nodes:
  retrieve   → pull relevant chunks from ChromaDB
  route      → classify topic / risk / owner
  govern     → dedup chunks, detect bypass intent, check refusal, score confidence
  answer     → synthesize a grounded answer from semantically relevant context
  escalate   → apply escalation rules for HIGH-risk answers
  output     → package the final ComplianceAnswer

Changes in this version:
  * Chunk deduplication by SHA-256 fingerprint before context is built.
  * Rewritten system prompt — instructs the LLM to SYNTHESIZE from
    semantically relevant content rather than requiring keyword match.
  * Bypass-intent detection in govern node — questions that ask to ignore,
    bypass, or circumvent compliance controls are forced to HIGH risk with
    human review required, and the LLM is told to explain the rule firmly.
  * filter_relevant_chunks() applied in answer node — only OOC-cleared
    chunks are sent to the LLM as context.
  * sources in ComplianceAnswer contain only OOC-cleared chunks —
    no unrelated document citations are shown.
  * OOC_REFUSAL_MESSAGE used for REASON_OOC refusal case.
  * refusal_reason and relevant_chunks tracked in state for debug panel.
"""

from __future__ import annotations

import hashlib
from typing import Any, List, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from loguru import logger

from app.agents.routing_agent import classify_question, is_bypass_question
from app.governance.confidence import score_confidence
from app.governance.escalation import should_escalate
from app.governance.refusal import (
    should_refuse,
    filter_relevant_chunks,
    is_unsupported_standard,
    OOC_REFUSAL_MESSAGE,
    OOC_THRESHOLD,
    REFUSAL_MESSAGE,
    REFUSAL_TEXT,
    REASON_EMPTY,
    REASON_LOW_SCORE,
    REASON_OOC,
    REASON_KEYWORD,
)
from app.rag.retriever import retrieve
from app.utils.config import get_settings
from app.utils.models import (
    ComplianceAnswer,
    ComplianceTopic,
    ComplianceOwner,
    DocumentChunk,
    RiskLevel,
    RoutingDecision,
)


# ──────────────────────────────────────────────────────────────
# LangGraph state definition
# ──────────────────────────────────────────────────────────────

class AgentState(TypedDict, total=False):
    question: str
    chunks: List[DocumentChunk]          # deduplicated, post-retrieval
    relevant_chunks: List[DocumentChunk] # OOC-filtered chunks (score >= OOC_THRESHOLD)
    rejected_chunks: List[DocumentChunk] # chunks below OOC threshold (for debug)
    retrieval_debug: Any                 # RetrievalDebugInfo from retriever
    routing: RoutingDecision
    bypass_intent: bool                  # True when question asks to bypass controls
    answer_text: str
    confidence: float
    refused: bool
    refusal_reason: Optional[str]        # REASON_EMPTY | REASON_OOC | REASON_LOW_SCORE
    escalated: bool
    escalation_reason: str               # human-readable reason for escalation
    requires_human_review: bool
    final_answer: ComplianceAnswer
    vectorstore: Any                     # Chroma instance (passed in at runtime)


# ──────────────────────────────────────────────────────────────
# Deduplication helper
# ──────────────────────────────────────────────────────────────

def _dedup_chunks(chunks: List[DocumentChunk]) -> List[DocumentChunk]:
    """
    Remove duplicate chunks by content fingerprint (SHA-256 of first 512 chars).

    When query expansion fires multiple variant queries, the same document
    passage can appear multiple times. Deduplication keeps only the first
    occurrence (highest-score copy, since chunks arrive sorted descending).

    Also merges chunks from the same source+page that differ only in
    chunk_index — keeping the highest-scoring copy.

    Parameters
    ----------
    chunks:
        Retrieved chunks, sorted descending by relevance_score.

    Returns
    -------
    List[DocumentChunk]
        Deduplicated list, still sorted descending by relevance_score.
    """
    seen_fingerprints: set[str] = set()
    seen_page_keys: set[str] = set()
    result: List[DocumentChunk] = []

    for chunk in chunks:
        # Fingerprint by content (first 512 chars covers most duplicates)
        fp = hashlib.sha256(chunk.content[:512].encode()).hexdigest()

        # Also deduplicate by source + page (catches near-identical overlap chunks)
        page_key = f"{chunk.source}::{chunk.page}"

        if fp in seen_fingerprints:
            logger.debug(
                f"[dedup] Dropped duplicate content chunk from "
                f"{chunk.source} p.{chunk.page} (score={chunk.relevance_score:.4f})"
            )
            continue

        if page_key in seen_page_keys:
            logger.debug(
                f"[dedup] Dropped same-page duplicate: {chunk.source} p.{chunk.page}"
            )
            continue

        seen_fingerprints.add(fp)
        seen_page_keys.add(page_key)
        result.append(chunk)

    dropped = len(chunks) - len(result)
    if dropped:
        logger.info(f"[dedup] Removed {dropped} duplicate chunk(s). {len(result)} unique chunks remain.")

    return result


# ──────────────────────────────────────────────────────────────
# System prompts
# ──────────────────────────────────────────────────────────────

# Standard synthesis prompt — instructs the LLM to reason semantically
# from the context rather than requiring an exact keyword match.
_ANSWER_SYSTEM_PROMPT = """\
You are an expert compliance advisor. Your task is to answer the user's \
compliance question by synthesizing information from the retrieved policy \
document excerpts provided below.

CRITICAL RULES:
1. Synthesize your answer from the retrieved context. The context contains \
semantically relevant information even if the exact words of the question do \
not appear verbatim. Reason from the content — do NOT say "the context does \
not contain" simply because a specific phrase is absent.
2. If the context discusses sanctions, obligations, penalties, or \
non-compliance consequences that are relevant to the question, incorporate \
that information into your answer.
3. Cite the source document name and article/section when referencing a \
specific rule (e.g., "Per GDPR Article 44…" or "As stated in \
FATF_AML_Guidelines.pdf…").
4. Be precise, professional, and direct. Do not hedge with phrases like \
"I'm not sure" when the context provides clear guidance.
5. Do NOT fabricate information not present in the context.
6. Do NOT give legal advice — provide compliance guidance only.
7. Keep your answer under 400 words.
"""

# Bypass-intent prompt — used when the user asks whether a compliance
# control can be ignored, skipped, or circumvented.
_BYPASS_SYSTEM_PROMPT = """\
You are an expert compliance advisor. The user's question asks whether a \
compliance control, regulation, or policy can be ignored, bypassed, \
circumvented, or disabled.

Your response MUST:
1. Clearly state that compliance controls CANNOT be bypassed or ignored.
2. Explain the relevant rule or obligation from the retrieved context, \
citing the source document and article/section.
3. Describe the consequences of non-compliance (sanctions, penalties, \
regulatory action) as found in the retrieved context.
4. Conclude with a clear statement that this matter requires immediate \
review by the responsible compliance officer.

Do NOT say "it depends" or "in some cases" — the answer is always that \
mandatory compliance controls cannot be waived unilaterally. \
Be firm, precise, and professional. Under 400 words.
"""


# ──────────────────────────────────────────────────────────────
# Graph node implementations
# ──────────────────────────────────────────────────────────────

def _node_retrieve(state: AgentState) -> AgentState:
    """
    Retrieve top-K relevant chunks from the vector store.

    Layer 1 keyword guard runs FIRST — if the question mentions a known
    unsupported standard (DPDP, HIPAA, ISO 9001, SOC 2, PCI DSS, CCPA, …)
    the question is refused immediately without touching the retriever or
    the LLM.  This prevents GDPR chunks from being returned for DPDP/HIPAA
    questions due to shared vocabulary, which would cause the LLM to produce
    a misleading "GDPR is different from DPDP" answer.
    """
    question = state["question"]

    # ── Layer 1: keyword guard ─────────────────────────────────
    is_unsupported, standard_name = is_unsupported_standard(question)
    if is_unsupported:
        logger.warning(
            f"[retrieve:keyword] Pre-retrieval refusal — "
            f"question asks about unsupported standard: '{standard_name}'. "
            f"Skipping retrieval and LLM call entirely."
        )
        state["chunks"] = []
        state["retrieval_debug"] = None
        state["refused"] = True
        state["refusal_reason"] = REASON_KEYWORD
        state["answer_text"] = REFUSAL_TEXT
        return state

    # ── Normal retrieval ───────────────────────────────────────
    vectorstore = state.get("vectorstore")

    if vectorstore is None:
        logger.error("No vectorstore in agent state — cannot retrieve.")
        state["chunks"] = []
        state["retrieval_debug"] = None
        return state

    chunks, debug_info = retrieve(question, vectorstore)
    state["chunks"] = chunks
    state["retrieval_debug"] = debug_info
    logger.debug(f"[retrieve] Got {len(chunks)} chunks before dedup.")
    return state


def _node_route(state: AgentState) -> AgentState:
    """Classify topic, risk level, and owner."""
    routing = classify_question(state["question"])
    state["routing"] = routing
    logger.debug(f"[route] topic={routing.topic}, risk={routing.risk_level}")
    return state


def _node_govern(state: AgentState) -> AgentState:
    """
    Governance node — runs checks in order:

    1. Early-exit: if Layer 1 (keyword guard) already refused the question
       in _node_retrieve, apply canonical refusal metadata and return.
    2. Bypass-intent detection: if the question asks to ignore/bypass a
       compliance control, force risk=HIGH and requires_human_review=True.
       IMPORTANT: bypass questions skip the OOC score gate entirely —
       they must reach the LLM so it can cite the policy and explain why
       the action is not permitted.
    3. Chunk deduplication: remove duplicate content/page results.
    4. Refusal gate (Layer 2): refuse if no chunks pass the OOC score
       threshold.  Skipped for bypass questions (see step 2).
    5. Confidence scoring: flag for human review if below threshold.

    Refusal canonical metadata (applied for ALL refusal reasons):
      topic    = GENERAL
      owner    = COMPLIANCE_OFFICER
      risk     = LOW
      escalated = False
      confidence = 0.0
    """
    chunks = state.get("chunks", [])
    question = state["question"]
    routing: RoutingDecision = state.get("routing", RoutingDecision())

    # ── 0. Early-exit: keyword guard already refused ───────────
    if state.get("refused") is True:
        _apply_refusal_metadata(state)
        return state

    # ── 1. Bypass-intent detection ─────────────────────────────
    # Run BEFORE the refusal gate so bypass questions are never refused.
    # "Confirm we can ignore GDPR for this client" must be answered by the
    # LLM (with a firm refusal citing the policy), not by the OOC gate.
    bypass = is_bypass_question(question)
    state["bypass_intent"] = bypass

    if bypass:
        logger.warning(
            f"[govern] ⚠️ Bypass intent detected: '{question[:80]}' — "
            "forcing HIGH risk + skipping score gate so LLM answers."
        )
        state["routing"] = RoutingDecision(
            topic=routing.topic,
            risk_level=RiskLevel.HIGH,
            owner=routing.owner,
            reasoning=routing.reasoning + " [OVERRIDE: bypass intent detected]",
        )

    # ── 2. Deduplicate chunks ──────────────────────────────────
    deduped = _dedup_chunks(chunks)
    state["chunks"] = deduped

    # ── 3. Refusal gate (Layer 2 — score) ─────────────────────
    # Bypass questions skip this gate.  The retriever may return low-scoring
    # GDPR chunks for "ignore GDPR" questions (vocab overlap, not exact match),
    # but those chunks are still relevant enough for the LLM to cite the rule.
    if bypass:
        logger.info(
            "[govern] Bypass question — skipping OOC score gate. "
            "Sending retrieved chunks to LLM regardless of scores."
        )
        state["refused"] = False
        state["refusal_reason"] = None
        # Use all chunks as relevant (bypass system prompt will guide the LLM)
        relevant = deduped
        rejected = []
    else:
        refused, refusal_reason = should_refuse(deduped)
        state["refused"] = refused
        state["refusal_reason"] = refusal_reason

        if refused:
            if refusal_reason == REASON_EMPTY:
                state["answer_text"] = (
                    "I was unable to search the policy documents because the "
                    "knowledge base appears to be empty. Please click "
                    "**Rebuild Knowledge Base** in the sidebar."
                )
                logger.warning("[govern] REFUSE — empty retrieval (no chunks indexed).")
            else:
                # REASON_OOC or REASON_LOW_SCORE
                state["answer_text"] = REFUSAL_TEXT
                logger.warning(
                    f"[govern] REFUSE ({refusal_reason}) — "
                    "top chunk score below OOC threshold."
                )

            _apply_refusal_metadata(state)
            state["relevant_chunks"] = []
            state["rejected_chunks"] = deduped
            return state

        relevant = filter_relevant_chunks(deduped)
        rejected = [c for c in deduped if c.relevance_score < OOC_THRESHOLD]

    state["relevant_chunks"] = relevant
    state["rejected_chunks"] = rejected

    # ── 4. Confidence scoring ──────────────────────────────────
    confidence = score_confidence(deduped)
    state["confidence"] = confidence

    cfg = get_settings()
    if bypass:
        state["requires_human_review"] = True
    else:
        state["requires_human_review"] = confidence < cfg.confidence_threshold

    logger.debug(
        f"[govern] confidence={confidence:.3f} | "
        f"bypass={bypass} | review={state['requires_human_review']}"
    )
    return state


def _apply_refusal_metadata(state: AgentState) -> None:
    """
    Set canonical refusal metadata on the state.

    For ALL refusal reasons the answer must have:
      topic     = GENERAL
      owner     = Compliance Officer
      risk      = LOW
      escalated = False
      confidence = 0.0

    This function mutates state in-place and is called from both the
    keyword-guard early-exit path (Layer 1) and the score-gate path (Layer 2).
    """
    state["routing"] = RoutingDecision(
        topic=ComplianceTopic.GENERAL,
        risk_level=RiskLevel.LOW,
        owner=ComplianceOwner.COMPLIANCE_OFFICER,
        reasoning="Refusal: question is outside the available compliance corpus.",
    )
    state["confidence"] = 0.0
    state["escalated"] = False
    state["requires_human_review"] = False
    state["bypass_intent"] = state.get("bypass_intent", False)
    # Ensure relevant/rejected are always populated for debug
    if "relevant_chunks" not in state:
        state["relevant_chunks"] = []
    if "rejected_chunks" not in state:
        state["rejected_chunks"] = state.get("chunks", [])


def _node_answer(state: AgentState) -> AgentState:
    """
    Generate the compliance answer grounded in retrieved context.
    Only runs if the question was NOT refused.

    Uses only OOC-threshold-cleared chunks (relevant_chunks) as context.
    This prevents unrelated documents from being sent to the LLM and
    appearing in source citations.

    Uses the bypass-specific system prompt when bypass intent was detected,
    which instructs the LLM to firmly explain the rule and consequences.
    """
    if state.get("refused", False):
        return state  # answer_text already set by govern node

    # Use only OOC-filtered chunks — these cleared the relevance threshold.
    # Falls back to raw deduped chunks if relevant_chunks not populated.
    relevant_chunks = state.get("relevant_chunks") or state.get("chunks", [])
    question = state["question"]
    bypass = state.get("bypass_intent", False)

    # Build context block from the top 5 relevant (OOC-filtered) chunks only
    context_parts: List[str] = []
    for i, chunk in enumerate(relevant_chunks[:5], start=1):
        context_parts.append(
            f"[Source {i}: {chunk.source}, Page {chunk.page}]\n{chunk.content}"
        )
    context_text = "\n\n---\n\n".join(context_parts)

    user_message = (
        f"Context:\n{context_text}\n\n"
        f"Question: {question}"
    )

    # Select system prompt based on intent
    system_prompt = _BYPASS_SYSTEM_PROMPT if bypass else _ANSWER_SYSTEM_PROMPT

    try:
        cfg = get_settings()
        llm = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=cfg.openrouter_api_key,
            model=cfg.openrouter_model,
            temperature=0.1,
            max_tokens=600,
        )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]
        response = llm.invoke(messages)
        state["answer_text"] = response.content.strip()
        logger.debug(
            f"[answer] LLM answer generated "
            f"({'bypass' if bypass else 'standard'} prompt). "
            f"Context: {len(relevant_chunks)} relevant chunks."
        )

    except Exception as exc:  # noqa: BLE001
        logger.error(f"[answer] LLM call failed: {exc}")
        # Graceful degradation — show raw chunks so the user gets something useful
        state["answer_text"] = (
            "The LLM is temporarily unavailable. "
            "Here are the most relevant policy excerpts:\n\n"
            + "\n\n".join(
                f"[{c.source}, p.{c.page}]: {c.content[:400]}"
                for c in relevant_chunks[:3]
            )
        )

    return state


def _node_escalate(state: AgentState) -> AgentState:
    """
    Determine whether the answer requires escalation.

    Refused questions are NEVER escalated — they already carry the canonical
    refusal metadata (risk=LOW, escalated=False) set by _apply_refusal_metadata.

    Bypass-intent questions are always escalated regardless of what
    should_escalate() returns.

    Populates escalation_reason for the debug panel.
    """
    # Refused questions must not be escalated — the refusal is the final answer.
    if state.get("refused", False):
        state["escalated"] = False
        state["escalation_reason"] = "No escalation — question was refused (out-of-corpus)."
        return state

    routing: RoutingDecision = state.get("routing", RoutingDecision())
    question = state["question"]
    answer_text = state.get("answer_text", "")
    bypass = state.get("bypass_intent", False)

    if bypass:
        escalated = True
        escalation_reason = "Bypass intent detected — user asked to ignore/circumvent a compliance control."
    elif routing.risk_level == RiskLevel.HIGH:
        escalated = True
        escalation_reason = f"Risk level is HIGH (classified by routing agent: {routing.reasoning})."
    else:
        escalated = should_escalate(
            question=question,
            answer_text=answer_text,
            risk_level=routing.risk_level,
        )
        if escalated:
            escalation_reason = "Dangerous intent phrase detected in question or answer implies risky approval."
        else:
            escalation_reason = "No escalation trigger — risk is LOW/MEDIUM and no dangerous phrases detected."

    state["escalated"] = escalated
    state["escalation_reason"] = escalation_reason
    logger.debug(f"[escalate] escalated={escalated} reason={escalation_reason}")
    return state


def _node_output(state: AgentState) -> AgentState:
    """Package the final ComplianceAnswer from accumulated state.

    Uses relevant_chunks (OOC-filtered) as sources — only documents that
    actually contributed to the answer are shown as citations.
    All chunk data is preserved in state for the debug panel.
    """
    routing: RoutingDecision = state.get(
        "routing",
        RoutingDecision(
            topic=ComplianceTopic.GENERAL,
            risk_level=RiskLevel.LOW,
            owner=ComplianceOwner.COMPLIANCE_OFFICER,
        ),
    )

    # Use relevant_chunks (OOC-filtered) as source citations.
    # For refused queries, relevant_chunks is [] so no unrelated docs shown.
    # Fall back to all chunks only if relevant_chunks was never populated.
    citation_chunks = state.get("relevant_chunks")
    if citation_chunks is None:
        citation_chunks = state.get("chunks", [])

    # Build agent-level debug bundle for the debug panel
    agent_debug = {
        "rejected_chunks": state.get("rejected_chunks", []),
        "relevant_chunks": state.get("relevant_chunks", []),
        "all_chunks": state.get("chunks", []),
        "refusal_reason": state.get("refusal_reason"),
        "escalation_reason": state.get("escalation_reason", ""),
        "bypass_intent": state.get("bypass_intent", False),
        "routing_reasoning": routing.reasoning,
        "risk_level": routing.risk_level,
        "ooc_threshold": OOC_THRESHOLD,
        "keyword_refused": state.get("refusal_reason") == REASON_KEYWORD,
    }

    state["final_answer"] = ComplianceAnswer(
        question=state["question"],
        answer=state.get("answer_text", ""),
        sources=citation_chunks,
        confidence=state.get("confidence", 0.0),
        topic=routing.topic,
        risk_level=routing.risk_level,
        owner=routing.owner,
        escalated=state.get("escalated", False),
        requires_human_review=state.get("requires_human_review", False),
        refused=state.get("refused", False),
        retrieval_debug=state.get("retrieval_debug"),
        agent_debug=agent_debug,
    )
    return state


# ──────────────────────────────────────────────────────────────
# Graph construction
# ──────────────────────────────────────────────────────────────

def _build_graph() -> Any:
    """Build and compile the LangGraph compliance agent."""
    graph = StateGraph(AgentState)

    graph.add_node("retrieve", _node_retrieve)
    graph.add_node("route", _node_route)
    graph.add_node("govern", _node_govern)
    graph.add_node("answer", _node_answer)
    graph.add_node("escalate", _node_escalate)
    graph.add_node("output", _node_output)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "route")
    graph.add_edge("route", "govern")
    graph.add_edge("govern", "answer")
    graph.add_edge("answer", "escalate")
    graph.add_edge("escalate", "output")
    graph.add_edge("output", END)

    return graph.compile()


# Compiled graph singleton — built once at import time
_compliance_graph = _build_graph()


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

def run_compliance_agent(question: str, vectorstore: Any) -> ComplianceAnswer:
    """
    Execute the full compliance agent pipeline for a given question.

    Parameters
    ----------
    question:
        The employee's compliance question.
    vectorstore:
        An open Chroma vector store instance.

    Returns
    -------
    ComplianceAnswer
        Fully populated answer with sources, confidence, risk, and
        escalation status.
    """
    if not question or not question.strip():
        raise ValueError("Question must not be empty.")

    initial_state: AgentState = {
        "question": question.strip(),
        "vectorstore": vectorstore,
    }

    result_state = _compliance_graph.invoke(initial_state)
    answer: ComplianceAnswer = result_state["final_answer"]

    logger.info(
        f"Agent complete | risk={answer.risk_level} | "
        f"conf={answer.confidence:.2f} | escalated={answer.escalated} | "
        f"bypass={result_state.get('bypass_intent', False)}"
    )
    return answer
