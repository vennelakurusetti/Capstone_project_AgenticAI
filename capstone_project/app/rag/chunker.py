"""
app/rag/chunker.py
------------------
Splits loaded LangChain Documents into overlapping text chunks.

Key upgrade: the ``document_type`` metadata field set by the loader is
explicitly propagated into every child chunk so that downstream retrieval
and the sidebar table can display per-document chunk counts accurately.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

# Chunk configuration — tuned for dense compliance documents
_CHUNK_SIZE = 800
_CHUNK_OVERLAP = 150


def split_documents(
    documents: List[Document],
) -> Tuple[List[Document], Dict[str, int]]:
    """
    Split a list of LangChain Documents into smaller overlapping chunks.

    Every chunk inherits the full metadata of its parent document,
    including ``source``, ``page``, and ``document_type``.

    Parameters
    ----------
    documents:
        Raw pages as returned by ``loader.load_policies()``.

    Returns
    -------
    (chunks, chunk_counts)
        ``chunks``       — flat list of all chunk Documents
        ``chunk_counts`` — dict mapping ``source`` filename → chunk count
    """
    if not documents:
        logger.warning("split_documents received an empty document list.")
        return [], {}

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_CHUNK_SIZE,
        chunk_overlap=_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
        add_start_index=True,
    )

    chunks: List[Document] = splitter.split_documents(documents)

    # Tag each chunk with a sequential index per source and ensure
    # document_type is carried forward (splitter preserves metadata
    # but we make it explicit to be safe).
    source_counters: Dict[str, int] = defaultdict(int)
    chunk_counts: Dict[str, int] = defaultdict(int)

    for chunk in chunks:
        src = chunk.metadata.get("source", "unknown")
        source_counters[src] += 1
        chunk.metadata["chunk_index"] = source_counters[src]

        # Guarantee document_type survives the split
        if "document_type" not in chunk.metadata:
            chunk.metadata["document_type"] = "General Policy"

        chunk_counts[src] += 1

    logger.info(
        f"Split {len(documents)} pages into {len(chunks)} chunks "
        f"(size={_CHUNK_SIZE}, overlap={_CHUNK_OVERLAP})."
    )
    return chunks, dict(chunk_counts)
