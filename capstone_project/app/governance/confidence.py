"""
app/governance/confidence.py
-----------------------------
Calculates a confidence score (0.0 – 1.0) for a retrieval result.

Scoring model
-------------
Confidence reflects how well the retrieved corpus supports an answer.
It is built from three additive components:

  1. Top-score anchor (50%)
     The single strongest chunk dominates — if the best match is weak,
     overall confidence is low regardless of how many chunks there are.

  2. Supporting-chunks quality (35%)
     Mean similarity of all chunks that cleared the OOC threshold.
     Rewards broad coverage across multiple document sections.

  3. Coverage bonus (up to 15%)
     Scaled by the count of chunks above OOC threshold:
       ≥5 chunks → full 0.15 bonus
       4 chunks  → 0.12
       3 chunks  → 0.09
       2 chunks  → 0.05
       1 chunk   → 0.00

Resulting scale (approximate)
------------------------------
>0.85  Very High — multiple strong chunks from relevant documents
0.70–0.85  High — strong top chunk + decent coverage
0.55–0.70  Moderate — single or few mid-range chunks
0.40–0.55  Low — weak matches
<0.40  Very Low — near-threshold retrieval (→ human review)

All scores are hard-clamped to [0, 1] before processing so out-of-range
values from library updates never leak into the UI.
"""

from __future__ import annotations

from typing import List

from app.utils.models import DocumentChunk

# Import OOC threshold so confidence scoring uses the same floor as refusal
from app.governance.refusal import OOC_THRESHOLD

# Tuning weights — must sum to 1.0 for the base (bonus is additive)
_TOP_WEIGHT       = 0.50   # weight for the single highest score
_SUPPORT_WEIGHT   = 0.35   # weight for mean of all OOC-passing chunks
_COVERAGE_BONUS   = 0.15   # max bonus for chunk count

# Coverage bonus table: (min_chunks_required, bonus_value)
_COVERAGE_TABLE = [
    (5, 0.15),
    (4, 0.12),
    (3, 0.09),
    (2, 0.05),
    (1, 0.00),
]


def score_confidence(chunks: List[DocumentChunk]) -> float:
    """
    Compute an aggregate confidence score for a set of retrieved chunks.

    Parameters
    ----------
    chunks:
        DocumentChunk objects as returned by the retriever / dedup step.
        Expected sorted descending by relevance_score.

    Returns
    -------
    float
        Confidence strictly in [0.0, 1.0].
    """
    if not chunks:
        return 0.0

    # Hard-clamp every score to [0, 1]
    clamped = [min(1.0, max(0.0, c.relevance_score)) for c in chunks]

    # Only chunks that cleared the OOC threshold contribute positively
    relevant = sorted(
        [s for s in clamped if s >= OOC_THRESHOLD],
        reverse=True,
    )

    if not relevant:
        # All chunks are below OOC threshold — very low confidence
        top = max(clamped)
        return round(top * 0.5, 4)   # at most 0.175 for a 0.35 score

    top_score = relevant[0]

    # Supporting quality — mean of ALL relevant chunks (including top)
    support_mean = sum(relevant) / len(relevant)

    # Coverage bonus
    bonus = 0.0
    for min_count, bonus_val in _COVERAGE_TABLE:
        if len(relevant) >= min_count:
            bonus = bonus_val
            break

    score = _TOP_WEIGHT * top_score + _SUPPORT_WEIGHT * support_mean + bonus

    return round(min(float(score), 1.0), 4)


def confidence_label(score: float) -> str:
    """Human-readable label for a confidence score."""
    if score >= 0.85:
        return "Very High"
    if score >= 0.70:
        return "High"
    if score >= 0.55:
        return "Moderate"
    if score >= 0.40:
        return "Low"
    return "Very Low"


def confidence_color(score: float) -> str:
    """CSS colour for a confidence score."""
    if score >= 0.85:
        return "#00C853"   # bright green
    if score >= 0.70:
        return "#64DD17"   # light green
    if score >= 0.55:
        return "#00BCD4"   # teal
    if score >= 0.40:
        return "#FF9800"   # orange
    return "#F44336"       # red
