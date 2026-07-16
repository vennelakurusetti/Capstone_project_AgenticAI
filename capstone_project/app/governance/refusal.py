"""
app/governance/refusal.py
--------------------------
Implements the anti-hallucination refusal gate.

This module is the SOLE enforcement point for deciding whether retrieved
chunks are good enough to answer from.  Two layers of defence:

  Layer 1 — Keyword guard (pre-retrieval)
    is_unsupported_standard() detects questions about known-unsupported
    regulations (DPDP, HIPAA, ISO 9001, SOC 2, PCI DSS, CCPA, ISO 27017, …)
    by matching keywords BEFORE the retriever is called.  This prevents the
    LLM from receiving GDPR chunks that superficially resemble the question
    and producing a misleading "GDPR is different from DPDP" answer.

  Layer 2 — Score gate (post-retrieval)
    should_refuse() checks the top-chunk relevance score against OOC_THRESHOLD.
    If the best match scores below the threshold the question is out-of-corpus
    and the LLM is NOT called.

Why both layers?
    DPDP / HIPAA / ISO questions contain data-protection vocabulary that
    overlaps with GDPR chunks, so cosine similarity can reach 0.35–0.42 even
    though the corpus has no relevant content.  A threshold alone cannot
    reliably separate these cases because the safe threshold would be too high
    and would start refusing legitimate GDPR questions.  The keyword guard
    catches all explicitly named unsupported standards deterministically.

Three refusal cases (Layer 2):
  1. EMPTY_RETRIEVAL  — retriever returned zero chunks.
  2. OUT_OF_CORPUS    — top chunk score < OOC_THRESHOLD (0.50).
  3. LOW_SIMILARITY   — safety-net for edge cases.

OOC_THRESHOLD raised from 0.35 → 0.50
    * Legitimate in-corpus questions (GDPR, AML, passwords) → 0.45–0.85
    * Unsupported-standard questions (DPDP, HIPAA, PCI-DSS) → 0.28–0.42
    The 0.50 boundary sits in the gap between these two bands.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from loguru import logger

from app.utils.models import DocumentChunk


# ── Refusal message (canonical — matches evaluation requirement) ────────────

REFUSAL_TEXT: str = (
    "This question is outside the available compliance corpus. "
    "I cannot provide a policy-based answer because the uploaded policy "
    "documents do not cover this topic. "
    "Please consult the appropriate Compliance Officer."
)

# Keep both names so existing imports from other modules don't break
OOC_REFUSAL_MESSAGE: str = REFUSAL_TEXT

REFUSAL_MESSAGE: str = (
    "This question is outside the available compliance corpus. "
    "I was unable to find relevant information in the loaded policy documents. "
    "Please consult your Compliance Officer directly or check whether the "
    "relevant policy document has been uploaded to the system."
)

# ── Reason codes ────────────────────────────────────────────────
REASON_EMPTY      = "EMPTY_RETRIEVAL"
REASON_OOC        = "OUT_OF_CORPUS"
REASON_LOW_SCORE  = "LOW_SIMILARITY"
REASON_KEYWORD    = "UNSUPPORTED_STANDARD"   # Layer 1 keyword guard
REASON_OK         = None                      # no refusal

# ── Thresholds ──────────────────────────────────────────────────
# Raised from 0.35 → 0.50 to close the vocabulary-overlap gap between
# GDPR chunks and DPDP/HIPAA/ISO questions.
OOC_THRESHOLD: float = 0.50

# Noise floor — chunks below this are ignored when building LLM context.
NOISE_FLOOR: float = 0.10


# ── Layer 1: Keyword guard for unsupported standards ────────────

# Each entry: (display_name, [regex_patterns])
# Patterns are tested against the lower-cased question.
_UNSUPPORTED_STANDARDS: List[Tuple[str, List[str]]] = [
    ("DPDP Act", [
        r"\bdpdp\b",
        r"digital personal data protection",
        r"india.*data protection act",
        r"data protection act.*india",
    ]),
    ("HIPAA", [
        r"\bhipaa\b",
        r"health insurance portability",
        r"protected health information",
        r"\bphi\b.*health",
    ]),
    ("ISO 9001", [
        r"\biso\s*9001\b",
        r"quality management system",
    ]),
    ("ISO 27017", [
        r"\biso\s*27017\b",
        r"cloud security controls.*iso",
    ]),
    ("ISO 27001", [
        r"\biso\s*27001\b",
    ]),
    ("SOC 2", [
        r"\bsoc\s*2\b",
        r"service organization control",
    ]),
    ("PCI DSS", [
        r"\bpci\s*dss\b",
        r"\bpci\b.*payment card",
        r"payment card industry data security",
    ]),
    ("CCPA", [
        r"\bccpa\b",
        r"california consumer privacy act",
        r"california privacy rights act",
        r"\bcpra\b",
    ]),
    ("NIST", [
        r"\bnist\b.*cybersecurity framework",
        r"national institute.*standards.*technology",
    ]),
    ("COBIT", [
        r"\bcobit\b",
    ]),
]


def is_unsupported_standard(question: str) -> Tuple[bool, Optional[str]]:
    """
    Layer 1 keyword guard.

    Check whether the question references a regulation or standard that is
    NOT covered by the loaded policy documents.

    Parameters
    ----------
    question:
        The user's compliance question.

    Returns
    -------
    (is_unsupported, standard_name)
        ``is_unsupported`` — True if the question matches a known unsupported standard.
        ``standard_name``  — Human-readable name of the matched standard, or None.
    """
    q = question.lower()

    for standard_name, patterns in _UNSUPPORTED_STANDARDS:
        for pattern in patterns:
            if re.search(pattern, q):
                logger.warning(
                    f"[refusal:keyword] REFUSE ({REASON_KEYWORD}): "
                    f"question matches unsupported standard '{standard_name}' "
                    f"via pattern r'{pattern}'."
                )
                return True, standard_name

    return False, None


# ── Layer 2: Score gate ──────────────────────────────────────────

def should_refuse(
    chunks: List[DocumentChunk],
    min_score: Optional[float] = None,
    ooc_threshold: Optional[float] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Layer 2 score gate.

    Decide whether the agent should refuse to answer based on retrieval
    scores alone.  The keyword guard (Layer 1) is separate and runs
    BEFORE retrieval in the agent pipeline.

    Parameters
    ----------
    chunks:
        Retrieved DocumentChunk objects, sorted descending by relevance_score.
    min_score:
        Noise floor — chunks below this are ignored (defaults to NOISE_FLOOR).
    ooc_threshold:
        Minimum top-chunk score to be in-corpus (defaults to OOC_THRESHOLD=0.50).

    Returns
    -------
    (refuse, reason)
        ``refuse`` — True → refuse, do not send context to the LLM.
        ``reason`` — REASON_EMPTY | REASON_OOC | REASON_LOW_SCORE | None.
    """
    floor     = min_score     if min_score     is not None else NOISE_FLOOR
    ooc_level = ooc_threshold if ooc_threshold is not None else OOC_THRESHOLD

    # ── Case 1: No chunks at all ───────────────────────────────
    if not chunks:
        logger.warning(
            f"[refusal] REFUSE ({REASON_EMPTY}): retriever returned zero chunks."
        )
        return True, REASON_EMPTY

    # ── Case 2: Out-of-corpus (top score too low) ──────────────
    sorted_chunks = sorted(chunks, key=lambda c: c.relevance_score, reverse=True)
    top_score  = sorted_chunks[0].relevance_score
    top_source = sorted_chunks[0].source

    if top_score < ooc_level:
        logger.warning(
            f"[refusal] REFUSE ({REASON_OOC}): "
            f"top chunk score={top_score:.4f} < OOC threshold={ooc_level:.2f}. "
            f"Best match: '{top_source}'. "
            f"Question is outside the loaded corpus — refusing without LLM call."
        )
        return True, REASON_OOC

    # ── Case 3: All above-noise chunks still below OOC level ──
    useful = [c for c in sorted_chunks if c.relevance_score >= floor]
    if not useful:
        logger.warning(
            f"[refusal] REFUSE ({REASON_LOW_SCORE}): "
            f"all {len(chunks)} chunk(s) below noise floor={floor:.2f}."
        )
        return True, REASON_LOW_SCORE

    # ── Pass ───────────────────────────────────────────────────
    above_ooc = [c for c in useful if c.relevance_score >= ooc_level]
    logger.info(
        f"[refusal] PASS: top_score={top_score:.4f} ≥ ooc_threshold={ooc_level:.2f} | "
        f"{len(above_ooc)}/{len(chunks)} chunks above OOC threshold | "
        f"top source='{top_source}'"
    )
    return False, REASON_OK


def filter_relevant_chunks(
    chunks: List[DocumentChunk],
    ooc_threshold: Optional[float] = None,
) -> List[DocumentChunk]:
    """
    Return only chunks that cleared the OOC threshold.

    Used by the answer node to build the LLM context block — ensures that
    unrelated chunks are never sent to the LLM even when the question as a
    whole passed the refusal gate.

    Parameters
    ----------
    chunks:
        All deduplicated chunks from the govern node.
    ooc_threshold:
        Minimum score to include. Defaults to OOC_THRESHOLD.

    Returns
    -------
    List[DocumentChunk]
        Filtered list, still sorted descending by relevance_score.
    """
    level = ooc_threshold if ooc_threshold is not None else OOC_THRESHOLD
    relevant = [c for c in chunks if c.relevance_score >= level]

    dropped = len(chunks) - len(relevant)
    if dropped:
        logger.debug(
            f"[refusal] filter_relevant_chunks: dropped {dropped} chunk(s) "
            f"below OOC threshold={level:.2f}. {len(relevant)} remain."
        )
    return relevant


def build_refusal_response(owner_name: str) -> str:
    """Build a contextualised refusal message that includes the escalation contact."""
    return (
        f"{REFUSAL_TEXT}\n\n"
        f"**Suggested contact:** {owner_name}"
    )
