"""
app/rag/loader.py
-----------------
Dynamically discovers and loads ALL PDF files from the policies directory
using pathlib glob — no filenames are ever hardcoded.

For every PDF the loader:
  1. Discovers it via glob("*.pdf")  — works for any future additions
  2. Infers a ``document_type`` from the filename using keyword mapping
  3. Loads all pages via LangChain PyPDFLoader
  4. Attaches enriched metadata to each page:
       source        → filename (e.g. GDPR_Regulation.pdf)
       document_type → inferred type (e.g. "GDPR")
       page          → page number (0-indexed from PyPDFLoader)
       file_path     → absolute path string

Returns both a flat page list (for chunking/embedding) and a
``LoadedDocumentInfo`` summary list (for the sidebar table).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from loguru import logger

from app.utils.config import get_settings


# ──────────────────────────────────────────────────────────────
# Document-type inference
# ──────────────────────────────────────────────────────────────

# Ordered list of (keyword_patterns, document_type) pairs.
# Patterns are matched case-insensitively against the lowercase filename.
# First match wins — put more specific patterns earlier.
_DOC_TYPE_RULES: List[tuple[List[str], str]] = [
    (["gdpr"],                          "GDPR"),
    (["aml", "fatf", "anti-money"],     "AML"),
    (["vendor", "supplier", "third-party", "third_party"], "Vendor"),
    (["password", "log_management", "log-management",
      "physical_security", "physical-security",
      "incident_response", "incident-response",
      "incident", "log"],               "Security"),
    (["crf", "governance", "risk-model", "risk_model"], "Governance"),
    (["privacy", "dpo"],                "Privacy"),
    (["legal"],                         "Legal"),
]

_DEFAULT_DOC_TYPE = "General Policy"


def infer_document_type(filename: str) -> str:
    """
    Infer the compliance document type from a PDF filename.

    The function lower-cases the filename and checks each keyword group
    in priority order, returning the first match.

    Parameters
    ----------
    filename:
        The PDF filename, e.g. ``FATF_AML_Guidelines.pdf``.

    Returns
    -------
    str
        One of: GDPR · AML · Vendor · Security · Governance · Privacy ·
        Legal · General Policy
    """
    name_lower = filename.lower()
    for keywords, doc_type in _DOC_TYPE_RULES:
        if any(kw in name_lower for kw in keywords):
            return doc_type
    return _DEFAULT_DOC_TYPE


# ──────────────────────────────────────────────────────────────
# Data classes for loader output
# ──────────────────────────────────────────────────────────────

@dataclass
class LoadedDocumentInfo:
    """Summary of a single PDF loaded from the policies directory."""

    filename: str
    document_type: str
    pages: int
    chunks: int = 0             # populated by chunker after splitting
    embedding_status: str = "Pending"   # "✓ Done" | "✗ Failed" | "Pending"
    error: Optional[str] = None


# ──────────────────────────────────────────────────────────────
# Core loader
# ──────────────────────────────────────────────────────────────

def load_policies(
    policies_dir: Optional[Path] = None,
) -> tuple[List[Document], List[LoadedDocumentInfo]]:
    """
    Scan *policies_dir* for every ``*.pdf`` file and load all pages.

    No filenames are hardcoded — the function discovers documents at
    runtime via ``pathlib.Path.glob("*.pdf")``.

    Parameters
    ----------
    policies_dir:
        Override the default directory from settings (useful in tests).

    Returns
    -------
    (docs, infos)
        ``docs``  — flat list of LangChain Document objects (all pages, all PDFs)
        ``infos`` — one ``LoadedDocumentInfo`` entry per PDF
    """
    cfg = get_settings()
    target_dir: Path = policies_dir or cfg.policies_dir

    if not target_dir.exists():
        logger.warning(f"Policies directory not found: {target_dir}")
        return [], []

    # Dynamic discovery — sorted for reproducibility
    pdf_files = sorted(target_dir.glob("*.pdf"))

    if not pdf_files:
        logger.warning(f"No PDF files found in {target_dir}.")
        return [], []

    all_docs: List[Document] = []
    infos: List[LoadedDocumentInfo] = []

    for pdf_path in pdf_files:
        doc_type = infer_document_type(pdf_path.name)
        info = LoadedDocumentInfo(
            filename=pdf_path.name,
            document_type=doc_type,
            pages=0,
        )

        try:
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()

            # Attach enriched metadata to every page
            for page in pages:
                page.metadata["source"] = pdf_path.name
                page.metadata["document_type"] = doc_type
                page.metadata["file_path"] = str(pdf_path)
                # PyPDFLoader sets 'page' as 0-based int; keep it

            all_docs.extend(pages)
            info.pages = len(pages)
            info.embedding_status = "✓ Done"
            logger.info(
                f"Loaded {len(pages):>3} pages  [{doc_type:>16}]  {pdf_path.name}"
            )

        except Exception as exc:  # noqa: BLE001
            info.embedding_status = "✗ Failed"
            info.error = str(exc)
            logger.error(f"Failed to load {pdf_path.name}: {exc}")

        infos.append(info)

    logger.success(
        f"Loaded {len(all_docs)} total pages from {len(pdf_files)} PDF(s)."
    )
    return all_docs, infos


# ──────────────────────────────────────────────────────────────
# Lightweight helpers (no full load required)
# ──────────────────────────────────────────────────────────────

def list_policy_files(policies_dir: Optional[Path] = None) -> List[str]:
    """Return the names of all PDFs present in the policies directory."""
    cfg = get_settings()
    target_dir: Path = policies_dir or cfg.policies_dir

    if not target_dir.exists():
        return []
    return [f.name for f in sorted(target_dir.glob("*.pdf"))]


def get_policy_infos(policies_dir: Optional[Path] = None) -> List[LoadedDocumentInfo]:
    """
    Return ``LoadedDocumentInfo`` objects with real page counts read via pypdf.

    This is the lightweight path used when the vector store already exists
    and we don't need to re-embed. It reads only the page count from each
    PDF (fast — no text extraction) so the sidebar can show accurate stats.
    """
    cfg = get_settings()
    target_dir: Path = policies_dir or cfg.policies_dir

    if not target_dir.exists():
        return []

    infos = []
    for f in sorted(target_dir.glob("*.pdf")):
        pages = 0
        try:
            # pypdf is already installed; PdfReader reads only the xref table
            from pypdf import PdfReader
            reader = PdfReader(str(f), strict=False)
            pages = len(reader.pages)
        except Exception:  # noqa: BLE001
            pages = 0

        infos.append(
            LoadedDocumentInfo(
                filename=f.name,
                document_type=infer_document_type(f.name),
                pages=pages,
                embedding_status="✓ Done",   # DB exists, so embedding is done
            )
        )
    return infos
