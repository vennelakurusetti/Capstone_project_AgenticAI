"""
app/governance/escalation.py
-----------------------------
Decides whether a compliance response requires human escalation.

Escalation is triggered by:
  1. HIGH risk classification (non-negotiable)
  2. Specific dangerous intent phrases in the question
  3. Answer text that contains approval language for high-risk actions

Escalated responses MUST NOT imply the action is approved —
the UI will display an ⚠ banner directing the user to a human reviewer.
"""

from __future__ import annotations

from app.utils.models import RiskLevel


# ──────────────────────────────────────────────────────────────
# Escalation triggers
# ──────────────────────────────────────────────────────────────

# Question-level phrases that always trigger escalation regardless of risk
_QUESTION_ESCALATION_PHRASES: list[str] = [
    "ignore gdpr",
    "bypass gdpr",
    "skip gdpr",
    "circumvent",
    "override compliance",
    "disable compliance",
    "ignore aml",
    "bypass sanctions",
    "skip sanctions",
    "sanctions screening",
    "store eu data in us",
    "transfer eu data",
    "cross-border transfer",
    "sell customer data",
    "share without consent",
    "delete audit trail",
    "remove audit log",
]

# Answer-level phrases indicating the LLM may have approved a risky action
# (should trigger escalation even if risk was classified as MEDIUM)
_ANSWER_APPROVAL_PHRASES: list[str] = [
    "you can proceed",
    "this is allowed",
    "it is permissible",
    "this is acceptable",
    "you may ignore",
    "this can be bypassed",
    "no need to comply",
]


def should_escalate(
    question: str,
    answer_text: str,
    risk_level: RiskLevel,
) -> bool:
    """
    Determine whether the response requires human escalation.

    Parameters
    ----------
    question:
        The original user question.
    answer_text:
        The generated LLM answer (or refusal message).
    risk_level:
        The risk classification assigned by the routing agent.

    Returns
    -------
    bool
        True → escalate to a human reviewer.
    """
    # Rule 1: Any HIGH risk question is automatically escalated
    if risk_level == RiskLevel.HIGH:
        return True

    q_lower = question.lower()
    a_lower = answer_text.lower()

    # Rule 2: Dangerous intent in question
    if any(phrase in q_lower for phrase in _QUESTION_ESCALATION_PHRASES):
        return True

    # Rule 3: LLM answer implies approval for a risky action
    if any(phrase in a_lower for phrase in _ANSWER_APPROVAL_PHRASES):
        return True

    return False


def escalation_banner_text(owner: str) -> str:
    """
    Return the warning message to display in the Streamlit UI
    when a question has been escalated.
    """
    return (
        "⚠️ **Human Review Required**\n\n"
        "This query has been flagged for escalation due to its risk level or content. "
        "**This response does not constitute approval of the described action.** "
        f"Please contact your **{owner}** for a formal compliance determination."
    )
