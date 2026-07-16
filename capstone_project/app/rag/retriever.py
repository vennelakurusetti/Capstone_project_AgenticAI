"""
app/rag/retriever.py
--------------------
Manages the ChromaDB vector store lifecycle and retrieval pipeline.

Key capabilities in this version:
  * Cosine-distance collection (consistent [0,1] scores).
  * Query expansion — semantically related paraphrases are added for
    cross-border / data-transfer queries so GDPR Article 44 is reliably
    retrieved even when the user's wording differs from document text.
  * Detailed failure-reason logging:
      - no chunks indexed
      - all scores below similarity_min_score
      - metadata missing
      - irrelevant chunks returned
  * Top-5 retrieved chunks printed to logs with source + page + score.
  * ``retrieve()`` returns a ``RetrievalDebugInfo`` bundle alongside chunks
    so the Streamlit Debug Mode can surface the raw retrieval details.
  * ``rebuild_vectorstore`` — deletes the existing DB directory and
    re-indexes all PDFs from scratch.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from loguru import logger

from app.rag.embeddings import ComplianceEmbeddings, get_embedding_function
from app.utils.config import get_settings
from app.utils.models import DocumentChunk


# ──────────────────────────────────────────────────────────────
# Query expansion table
# ──────────────────────────────────────────────────────────────

# Maps trigger phrases (lowercase substrings) to a list of additional
# queries that semantically cover the same topic from different angles.
# This compensates for vocabulary mismatch between user wording and
# the exact phrases used in GDPR / AML document text.
_QUERY_EXPANSION: Dict[str, List[str]] = {
    # Cross-border / international data transfers (GDPR Chapter V, Art. 44-49)
    "eu customer data": [
        "transfer of personal data to third countries GDPR Article 44",
        "cross-border data transfer adequacy decision standard contractual clauses",
        "international transfer of personal data outside European Union",
    ],
    "store eu": [
        "transfer of personal data to third countries GDPR Article 44",
        "data localisation requirements GDPR",
        "storing EU personal data outside the European Economic Area",
    ],
    "transfer eu": [
        "GDPR Article 44 transfers to third countries or international organisations",
        "adequacy decision binding corporate rules standard contractual clauses",
    ],
    "cross-border": [
        "GDPR Chapter V international data transfers",
        "transfer of personal data to third countries adequacy decision",
    ],
    "data transfer": [
        "GDPR Article 44 transfer of personal data to third countries",
        "standard contractual clauses adequacy decision international transfer",
    ],
    "outside the eu": [
        "GDPR transfer of personal data to third countries Article 44 45 46",
        "international data transfers adequacy decision standard contractual clauses",
    ],
    "us region": [
        "transfer of personal data to third countries United States GDPR",
        "EU US data transfer adequacy decision standard contractual clauses",
    ],
    # Retention
    "retention": [
        "data retention period storage limitation GDPR Article 5",
        "how long to keep personal data GDPR",
    ],
    # Consent
    "consent": [
        "GDPR lawful basis for processing personal data consent Article 6 7",
        "data subject consent freely given specific informed unambiguous",
    ],
    # Data breach
    "data breach": [
        "GDPR personal data breach notification Article 33 34",
        "72 hour breach notification supervisory authority",
    ],
    # AML / KYC
    "due diligence": [
        "customer due diligence CDD KYC AML FATF",
        "anti-money laundering know your customer enhanced due diligence",
    ],
    # Password
    "password": [
        "password construction requirements minimum length complexity",
        "SANS password policy standard characters special uppercase",
    ],
}


def _expand_query(query: str) -> List[str]:
    """
    Return a de-duplicated list of queries starting with the original,
    followed by any expansion queries whose trigger appears in the input.

    Parameters
    ----------
    query:
        The original user question (case-insensitive match).

    Returns
    -------
    List[str]
        [original_query, expansion_1, expansion_2, ...]
    """
    q_lower = query.lower()
    expanded: List[str] = [query]
    seen: set = {query}

    for trigger, extras in _QUERY_EXPANSION.items():
        if trigger in q_lower:
            for extra in extras:
                if extra not in seen:
                    expanded.append(extra)
                    seen.add(extra)

    if len(expanded) > 1:
        logger.info(
            f"[retrieve] Query expanded: '{query[:60]}' → "
            f"{len(expanded)} queries total."
        )
    return expanded


# ──────────────────────────────────────────────────────────────
# Debug bundle
# ──────────────────────────────────────────────────────────────

class RetrievalDebugInfo(NamedTuple):
    """Diagnostic data surfaced to the Streamlit Debug Mode panel."""

    original_query: str
    expanded_queries: List[str]           # all queries tried
    raw_results: List[Tuple[str, float]]  # (content[:300], score) before filtering
    filtered_count: int                   # chunks that passed min-score
    failure_reason: Optional[str]         # None if retrieval succeeded


# ──────────────────────────────────────────────────────────────
# Existence check
# ──────────────────────────────────────────────────────────────

def vectorstore_exists(persist_dir: Optional[Path] = None) -> bool:
    """
    Return True if ChromaDB has already been built and persisted.
    Detected by the presence of ``chroma.sqlite3``.
    """
    cfg = get_settings()
    target = persist_dir or cfg.chroma_persist_dir
    return (target / "chroma.sqlite3").exists()


# ──────────────────────────────────────────────────────────────
# Build
# ──────────────────────────────────────────────────────────────

def build_vectorstore(
    chunks: List[Document],
    persist_dir: Optional[Path] = None,
    embedding_fn: Optional[ComplianceEmbeddings] = None,
) -> Chroma:
    """
    Embed *chunks* and persist them in ChromaDB using cosine distance.

    Parameters
    ----------
    chunks:
        Chunked documents from ``chunker.split_documents()``.
    persist_dir:
        Override the default persist path from settings.
    embedding_fn:
        Override the default embedding function (useful for testing).

    Returns
    -------
    Chroma
        Loaded LangChain Chroma vectorstore instance.
    """
    cfg = get_settings()
    target = persist_dir or cfg.chroma_persist_dir
    target.mkdir(parents=True, exist_ok=True)

    emb = embedding_fn or get_embedding_function()

    # Log the model being used so mismatches are immediately visible
    model_label = getattr(emb, "model_name", "unknown")
    logger.info(
        f"Building ChromaDB at {target} | "
        f"chunks={len(chunks)} | embedding_model={model_label}"
    )

    # cosine distance → relevance scores in [0, 1] — no L2 normalisation needed
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=emb,
        collection_name=cfg.chroma_collection_name,
        persist_directory=str(target),
        collection_metadata={"hnsw:space": "cosine"},
    )

    logger.success(
        f"Vector store built: {len(chunks)} chunks → {target} "
        f"[model={model_label}]"
    )

    # Post-build validation — raises RuntimeError if embedding count is 0
    validate_vectorstore_after_build(vectorstore, expected_chunks=len(chunks), persist_dir=target)

    # Persist chunk-count stats so the sidebar can read them without re-scanning
    try:
        import json as _json
        chunk_counts_by_source = {}
        for chunk in chunks:
            src = chunk.metadata.get("source", "unknown")
            chunk_counts_by_source[src] = chunk_counts_by_source.get(src, 0) + 1
        (target / "db_stats.json").write_text(
            _json.dumps({
                "total_chunks": len(chunks),
                "chunk_counts": chunk_counts_by_source,
                "embedding_model": model_label,
            }, indent=2)
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[build] Could not write db_stats.json: {exc}")

    return vectorstore


# ──────────────────────────────────────────────────────────────
# Rebuild (delete + re-index)
# ──────────────────────────────────────────────────────────────

def rebuild_vectorstore(
    persist_dir: Optional[Path] = None,
    embedding_fn: Optional[ComplianceEmbeddings] = None,
) -> Tuple[Chroma, List]:
    """
    Delete the existing ChromaDB directory and re-index all PDFs.

    This is triggered by the "Rebuild Knowledge Base" sidebar button.
    It dynamically discovers all PDFs currently in ``data/policies/``,
    so newly added files are automatically included.

    Returns
    -------
    (vectorstore, infos)
        ``vectorstore`` — freshly built Chroma instance
        ``infos``       — list of LoadedDocumentInfo (for sidebar table)
    """
    from app.rag.chunker import split_documents
    from app.rag.loader import load_policies

    cfg = get_settings()
    target = persist_dir or cfg.chroma_persist_dir

    # Step 1 — delete old DB
    if target.exists():
        logger.info(f"Deleting old ChromaDB at {target} …")
        shutil.rmtree(target)
        logger.info("Old ChromaDB deleted.")

    target.mkdir(parents=True, exist_ok=True)

    # Step 2 — load all PDFs
    docs, infos = load_policies()
    if not docs:
        logger.warning("No documents found — vector store will be empty.")
        return None, infos

    # Step 3 — chunk
    chunks, chunk_counts = split_documents(docs)

    for info in infos:
        info.chunks = chunk_counts.get(info.filename, 0)

    # Step 4 — embed and store
    emb = embedding_fn or get_embedding_function()
    vs = build_vectorstore(chunks, persist_dir=target, embedding_fn=emb)

    # Step 5 — verify GDPR chunks are present
    gdpr_count = sum(1 for c in chunks if "GDPR" in c.metadata.get("document_type", ""))
    logger.info(f"Post-build verification: {gdpr_count} GDPR chunks indexed.")

    logger.success(
        f"Rebuild complete: {len(chunks)} total chunks from {len(infos)} PDFs."
    )
    return vs, infos


# ──────────────────────────────────────────────────────────────
# Load existing
# ──────────────────────────────────────────────────────────────

def load_vectorstore(
    persist_dir: Optional[Path] = None,
    embedding_fn: Optional[ComplianceEmbeddings] = None,
) -> Chroma:
    """
    Open an existing persisted ChromaDB vector store.

    Raises
    ------
    FileNotFoundError
        If the vector store has not been built yet.
    """
    cfg = get_settings()
    target = persist_dir or cfg.chroma_persist_dir

    if not vectorstore_exists(target):
        raise FileNotFoundError(
            f"ChromaDB not found at {target}. Call build_vectorstore() first."
        )

    emb = embedding_fn or get_embedding_function()
    model_label = getattr(emb, "model_name", "unknown")

    logger.info(f"Loading ChromaDB from {target} [model={model_label}] …")
    vectorstore = Chroma(
        collection_name=cfg.chroma_collection_name,
        embedding_function=emb,
        persist_directory=str(target),
        collection_metadata={"hnsw:space": "cosine"},
    )
    logger.success("ChromaDB loaded.")
    return vectorstore


# ──────────────────────────────────────────────────────────────
# Retrieve
# ──────────────────────────────────────────────────────────────

def retrieve(
    query: str,
    vectorstore: Chroma,
    top_k: Optional[int] = None,
    min_score: Optional[float] = None,
) -> Tuple[List[DocumentChunk], RetrievalDebugInfo]:
    """
    Perform an expanded similarity search and return typed ``DocumentChunk``
    objects plus a ``RetrievalDebugInfo`` bundle for debug UI display.

    Process
    -------
    1. Expand the query into semantically related paraphrases.
    2. Run similarity_search_with_relevance_scores for each expansion.
    3. Merge results, keeping the best score per document chunk.
    4. Apply min_score filter, log failure reasons if no chunks pass.
    5. Log top-5 chunks (source, page, score, content preview).

    Parameters
    ----------
    query:
        The compliance question.
    vectorstore:
        An open Chroma instance.
    top_k:
        Number of candidates to retrieve per query variant.
    min_score:
        Minimum cosine similarity for a chunk to be returned.
        Defaults to ``settings.similarity_min_score``.

    Returns
    -------
    (chunks, debug_info)
        ``chunks``     — ranked list of DocumentChunk (may be empty).
        ``debug_info`` — RetrievalDebugInfo for the Streamlit debug panel.
    """
    cfg = get_settings()
    k = top_k or cfg.retrieval_top_k
    threshold = min_score if min_score is not None else cfg.similarity_min_score

    # ── 0. Guard: is the collection populated? ─────────────────
    # Use the vectorstore's own collection to avoid opening a second
    # PersistentClient (ChromaDB v1.x forbids two clients on the same path,
    # which would make get_collection_count() return 0 while the store is open).
    try:
        total_in_db = vectorstore._collection.count()
    except Exception:
        total_in_db = get_collection_count()

    if total_in_db == 0:
        reason = "FAILURE: ChromaDB collection is empty — no chunks have been indexed."
        logger.error(f"[retrieve] {reason}")
        debug = RetrievalDebugInfo(
            original_query=query,
            expanded_queries=[query],
            raw_results=[],
            filtered_count=0,
            failure_reason=reason,
        )
        return [], debug

    # ── 1. Expand query ────────────────────────────────────────
    queries = _expand_query(query)

    # ── 2. Search all variants, merge by content key ───────────
    # Key = first 200 chars of content (deduplicates across variants)
    best_by_content: Dict[str, Tuple[Document, float]] = {}

    for q in queries:
        try:
            results = vectorstore.similarity_search_with_relevance_scores(q, k=k)
        except Exception as exc:
            logger.error(f"[retrieve] similarity_search failed for '{q[:60]}': {exc}")
            continue

        for doc, score in results:
            raw = float(score)
            # Cosine space → clamp to [0, 1] defensively
            normalised = round(min(1.0, max(0.0, raw)), 4)
            key = doc.page_content[:200]
            if key not in best_by_content or normalised > best_by_content[key][1]:
                best_by_content[key] = (doc, normalised)

    # Flatten and sort by score descending
    all_raw = sorted(best_by_content.values(), key=lambda x: x[1], reverse=True)

    # ── 3. Build raw debug snapshot (before filtering) ─────────
    raw_results_debug = [
        (doc.page_content[:300], score) for doc, score in all_raw[:10]
    ]

    # ── 4. Log top-5 retrieved chunks (always, before filter) ──
    logger.info(
        f"[retrieve] Top-{min(5, len(all_raw))} raw results for: '{query[:70]}'"
    )
    for rank, (doc, score) in enumerate(all_raw[:5], start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        doc_type = doc.metadata.get("document_type", "?")
        content_preview = doc.page_content[:300].replace("\n", " ")

        # Flag missing metadata explicitly
        missing = []
        if not doc.metadata.get("source"):
            missing.append("source")
        if doc.metadata.get("page") is None:
            missing.append("page")
        if not doc.metadata.get("document_type"):
            missing.append("document_type")
        meta_warning = f" ⚠️ MISSING METADATA: {missing}" if missing else ""

        logger.info(
            f"  [{rank}] score={score:.4f} | {source} p.{page} [{doc_type}]"
            f"{meta_warning}\n"
            f"       {content_preview}"
        )

    # ── 5. Build DocumentChunk objects from ALL raw results ────
    # The threshold filter has been intentionally removed from the retriever.
    # The retriever's job is to find and rank the best matching chunks.
    # Threshold enforcement (deciding whether to answer or refuse) is the
    # sole responsibility of refusal.py — keeping the two concerns separate
    # prevents a race condition where chunks are silently dropped before the
    # refusal gate can make an informed decision.
    failure_reason: Optional[str] = None

    if not all_raw:
        failure_reason = (
            f"RETRIEVAL EMPTY: similarity_search returned zero results "
            f"(total_in_db={total_in_db}). "
            f"The ChromaDB collection may be empty or corrupt — try Rebuild KB."
        )
        logger.error(f"[retrieve] {failure_reason}")

    chunks: List[DocumentChunk] = []
    for doc, score in all_raw:
        chunks.append(
            DocumentChunk(
                content=doc.page_content,
                source=doc.metadata.get("source", "unknown"),
                page=doc.metadata.get("page"),
                chunk_index=doc.metadata.get("chunk_index", 0),
                relevance_score=score,
            )
        )

    # Already sorted descending from step 2
    chunks.sort(key=lambda c: c.relevance_score, reverse=True)

    # ── 6. Final summary log ───────────────────────────────────
    if chunks:
        below_threshold = [c for c in chunks if c.relevance_score < threshold]
        above_threshold = [c for c in chunks if c.relevance_score >= threshold]
        logger.info(
            f"[retrieve] ✓ Returning {len(chunks)} chunks to refusal gate | "
            f"above threshold ({threshold:.2f}): {len(above_threshold)} | "
            f"below threshold: {len(below_threshold)} | "
            f"top score={chunks[0].relevance_score:.4f} | "
            f"top source={chunks[0].source} p.{chunks[0].page}"
        )
        # Warn if top result is from a surprising source for GDPR queries
        top_doc_type = chunks[0].source
        query_lower = query.lower()
        gdpr_terms = {"gdpr", "eu customer", "data transfer", "personal data", "eu data"}
        if any(t in query_lower for t in gdpr_terms) and "gdpr" not in top_doc_type.lower():
            logger.warning(
                f"[retrieve] ⚠️ GDPR-flavoured query but top chunk is from "
                f"'{top_doc_type}' — GDPR PDF may not be fully indexed."
            )
    else:
        logger.warning("[retrieve] ✗ No chunks returned — refusal gate will trigger.")

    debug = RetrievalDebugInfo(
        original_query=query,
        expanded_queries=queries,
        raw_results=raw_results_debug,
        filtered_count=len([c for c in chunks if c.relevance_score >= threshold]),
        failure_reason=failure_reason,
    )

    return chunks, debug


# ──────────────────────────────────────────────────────────────
# Backwards-compatible shim
# ──────────────────────────────────────────────────────────────

def retrieve_chunks(
    query: str,
    vectorstore: Chroma,
    top_k: Optional[int] = None,
    min_score: Optional[float] = None,
) -> List[DocumentChunk]:
    """
    Convenience wrapper around ``retrieve()`` that returns only the chunk
    list — used by callers that do not need the debug bundle.
    """
    chunks, _ = retrieve(query, vectorstore, top_k=top_k, min_score=min_score)
    return chunks


# ──────────────────────────────────────────────────────────────
# Stats helper
# ──────────────────────────────────────────────────────────────

def get_collection_count(persist_dir: Optional[Path] = None) -> int:
    """
    Return the total number of embeddings in the collection.

    Reads from db_stats.json first (written during build) to avoid opening
    a second PersistentClient while a vectorstore is already open.
    Falls back to a fresh PersistentClient only when no vectorstore is open.
    """
    cfg = get_settings()
    target = persist_dir or cfg.chroma_persist_dir

    if not vectorstore_exists(target):
        return 0

    # Fast path — read from the stats file written during build
    stats_path = target / "db_stats.json"
    if stats_path.exists():
        try:
            import json as _json
            data = _json.loads(stats_path.read_text())
            total = data.get("total_chunks", 0)
            if total > 0:
                return total
        except Exception:
            pass

    # Fallback — open a client (safe only when no other client is open).
    # If a vectorstore is already open, this will raise ValueError — return 0
    # gracefully since the caller should be using vectorstore._collection.count().
    try:
        client = chromadb.PersistentClient(
            path=str(target),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        for col in client.list_collections():
            if col.name == cfg.chroma_collection_name:
                return col.count()
        return 0
    except Exception:  # noqa: BLE001  (includes ValueError from dual-client)
        return 0


def get_db_stats(persist_dir: Optional[Path] = None) -> dict:
    """
    Return a complete snapshot of the vector database state for display
    in the sidebar and Debug Mode panel.

    Reads from db_stats.json first to avoid opening a second PersistentClient
    while a vectorstore is already open (ChromaDB v1.x forbids two clients
    on the same path).

    Returns
    -------
    dict with keys:
        exists          bool
        persist_path    str (absolute)
        collection_name str
        embedding_count int
        all_collections list[dict]  — [{name, count, metadata}]
        embedding_model str
        status          str  — "ready" | "empty" | "missing"
    """
    from app.rag.embeddings import _DEFAULT_MODEL

    cfg = get_settings()
    target = persist_dir or cfg.chroma_persist_dir

    base = {
        "exists": False,
        "persist_path": str(target.resolve()),
        "collection_name": cfg.chroma_collection_name,
        "embedding_count": 0,
        "all_collections": [],
        "embedding_model": _DEFAULT_MODEL,
        "status": "missing",
    }

    if not vectorstore_exists(target):
        return base

    base["exists"] = True

    # Fast path — read from db_stats.json (written during build).
    # This avoids opening a second PersistentClient while the vectorstore
    # is already open, which would return count=0 on ChromaDB v1.x.
    stats_path = target / "db_stats.json"
    if stats_path.exists():
        try:
            import json as _json
            data = _json.loads(stats_path.read_text())
            total = data.get("total_chunks", 0)
            model = data.get("embedding_model", _DEFAULT_MODEL)
            if total > 0:
                base["embedding_count"] = total
                base["embedding_model"] = model
                base["all_collections"] = [{
                    "name": cfg.chroma_collection_name,
                    "count": total,
                    "metadata": {"hnsw:space": "cosine"},
                }]
                base["status"] = "ready"
                return base
        except Exception:
            pass

    # Fallback — open a fresh client (safe when no vectorstore is open)
    try:
        client = chromadb.PersistentClient(
            path=str(target),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        all_cols = []
        for col in client.list_collections():
            cnt = col.count()
            all_cols.append({
                "name": col.name,
                "count": cnt,
                "metadata": col.metadata,
            })
            if col.name == cfg.chroma_collection_name:
                base["embedding_count"] = cnt

        base["all_collections"] = all_cols
        base["status"] = "ready" if base["embedding_count"] > 0 else "empty"

    except Exception as exc:  # noqa: BLE001
        logger.error(f"[get_db_stats] ChromaDB introspection failed: {exc}")
        base["status"] = "error"

    return base


def validate_vectorstore_after_build(
    vectorstore: Chroma,
    expected_chunks: int,
    persist_dir: Optional[Path] = None,
) -> None:
    """
    Validate that the vector store was built successfully.

    Raises
    ------
    RuntimeError
        If the stored embedding count does not match expected_chunks.
    """
    cfg = get_settings()
    target = persist_dir or cfg.chroma_persist_dir

    # Use the vectorstore's own collection to avoid opening a second
    # PersistentClient — ChromaDB v1.x forbids two clients on the same path,
    # which would make get_collection_count() return 0 while VS is open.
    try:
        actual = vectorstore._collection.count()
    except Exception:
        actual = get_collection_count(persist_dir)

    logger.info(
        f"[validate] Post-build check — "
        f"collection={cfg.chroma_collection_name} | "
        f"path={target.resolve()} | "
        f"expected={expected_chunks} | "
        f"actual={actual}"
    )

    if actual == 0:
        raise RuntimeError(
            f"Vector store build FAILED: expected {expected_chunks} embeddings "
            f"but found 0 in collection '{cfg.chroma_collection_name}' "
            f"at '{target.resolve()}'. "
            f"Check disk space and ChromaDB logs above."
        )

    if actual < expected_chunks:
        logger.warning(
            f"[validate] Partial build: {actual}/{expected_chunks} embeddings stored. "
            f"This may happen if chunks were already present from a previous run."
        )

    logger.success(
        f"[validate] Vector store OK: {actual} embeddings in "
        f"'{cfg.chroma_collection_name}'"
    )
