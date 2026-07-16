"""
app/utils/models.py
-------------------
Shared Pydantic data models used across the entire application.
Single source of truth for all typed data structures.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, List, Optional

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
# Enumerations
# ──────────────────────────────────────────────

class ComplianceTopic(str, Enum):
    GDPR = "GDPR"
    AML = "AML"
    VENDOR = "Vendor"
    SECURITY = "Security"
    PRIVACY = "Privacy"
    LEGAL = "Legal"
    GENERAL = "General Compliance"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class ComplianceOwner(str, Enum):
    DPO = "Data Protection Officer (DPO)"
    AML_OFFICER = "AML Officer"
    PROCUREMENT = "Procurement Team"
    SECURITY_TEAM = "Security Team"
    LEGAL_COUNSEL = "Legal Counsel"
    COMPLIANCE_OFFICER = "Compliance Officer"


# ──────────────────────────────────────────────
# RAG / Retrieval models
# ──────────────────────────────────────────────

class DocumentChunk(BaseModel):
    """A single retrieved document chunk from ChromaDB."""

    content: str
    source: str                   # filename
    page: Optional[int] = None
    chunk_index: int = 0
    relevance_score: float = 0.0  # cosine similarity (0–1)


class RetrievalResult(BaseModel):
    """Aggregated retrieval output for a single query."""

    query: str
    chunks: List[DocumentChunk]
    top_score: float = 0.0

    @property
    def has_results(self) -> bool:
        return len(self.chunks) > 0


# ──────────────────────────────────────────────
# Routing / Classification models
# ──────────────────────────────────────────────

class RoutingDecision(BaseModel):
    """Output of the routing agent."""

    topic: ComplianceTopic = ComplianceTopic.GENERAL
    risk_level: RiskLevel = RiskLevel.LOW
    owner: ComplianceOwner = ComplianceOwner.COMPLIANCE_OFFICER
    reasoning: str = ""


# ──────────────────────────────────────────────
# Compliance answer model
# ──────────────────────────────────────────────

class ComplianceAnswer(BaseModel):
    """Full answer payload returned to the UI and audit logger."""

    question: str
    answer: str
    sources: List[DocumentChunk] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    topic: ComplianceTopic = ComplianceTopic.GENERAL
    risk_level: RiskLevel = RiskLevel.LOW
    owner: ComplianceOwner = ComplianceOwner.COMPLIANCE_OFFICER
    escalated: bool = False
    requires_human_review: bool = False
    refused: bool = False          # True when question is outside corpus
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    # Retrieval debug info — populated by the agent, consumed by Streamlit debug panel.
    # Typed as Any to avoid a circular import with retriever.py.
    retrieval_debug: Optional[Any] = Field(default=None, exclude=True)
    # Agent-level debug info — rejected chunks, refusal/escalation/risk reasons.
    # Typed as Any to keep models.py free of agent/governance imports.
    agent_debug: Optional[Any] = Field(default=None, exclude=True)

    model_config = {"arbitrary_types_allowed": True}

    @property
    def source_names(self) -> List[str]:
        """Deduplicated list of source document filenames."""
        return list(dict.fromkeys(c.source for c in self.sources))


# ──────────────────────────────────────────────
# Audit record model
# ──────────────────────────────────────────────

class AuditRecord(BaseModel):
    """One row in the audit trail."""

    id: str                        # UUID
    timestamp: str                 # ISO-8601
    question: str
    answer: str
    sources: List[str]             # filenames only
    confidence: float
    topic: str
    risk_level: str
    owner: str
    escalated: bool
    refused: bool
    session_id: str = ""


# ──────────────────────────────────────────────
# Evaluation models
# ──────────────────────────────────────────────

class EvalCase(BaseModel):
    """A single evaluation test case."""

    name: str
    question: str
    expected_topic: Optional[ComplianceTopic] = None
    expected_risk: Optional[RiskLevel] = None
    expect_escalation: bool = False
    expect_refusal: bool = False
    description: str = ""


class EvalResult(BaseModel):
    """Result of running one evaluation case."""

    case: EvalCase
    passed: bool
    actual_topic: Optional[str] = None
    actual_risk: Optional[str] = None
    actual_escalated: bool = False
    actual_refused: bool = False
    actual_confidence: float = 0.0
    failure_reason: str = ""
