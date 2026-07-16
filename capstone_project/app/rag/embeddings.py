"""
app/rag/embeddings.py
---------------------
Provides a cached SentenceTransformers embedding function compatible
with ChromaDB's embedding interface.

Model choice: ``multi-qa-mpnet-base-dot-v1``
  - 768-dim (vs. 384 for all-MiniLM-L6-v2) — more expressive
  - Explicitly trained on question-answer pairs using dot-product similarity
  - Outperforms MiniLM on asymmetric retrieval (short query → long doc passage)
  - Well-suited for legal/compliance documents where paraphrasing matters
  - SBERT benchmark: 57.60 on MS MARCO vs. 33.02 for all-MiniLM-L6-v2
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

from loguru import logger
from sentence_transformers import SentenceTransformer

# Upgraded model — optimised for Q&A retrieval over long documents
# Falls back to all-MiniLM-L6-v2 only if overridden via get_embedding_function()
_DEFAULT_MODEL = "multi-qa-mpnet-base-dot-v1"


class ComplianceEmbeddings:
    """
    Thin wrapper around SentenceTransformer that exposes the two
    callable interfaces expected by ChromaDB and LangChain:

    * ``__call__(input: List[str]) -> List[List[float]]``  — ChromaDB EmbeddingFunction
    * ``embed_documents(texts)`` / ``embed_query(text)``   — LangChain interface
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        self._model_name = model_name
        logger.info(f"Loading embedding model: {model_name}")
        self._model = SentenceTransformer(model_name)
        dim = self._model.get_sentence_embedding_dimension()
        logger.success(f"Embedding model loaded: {model_name}  (dim={dim})")

    @property
    def model_name(self) -> str:
        return self._model_name

    # ── ChromaDB EmbeddingFunction interface ──────────────────────────
    def __call__(self, input: List[str]) -> List[List[float]]:  # noqa: A002
        """Callable interface required by ChromaDB's EmbeddingFunction."""
        return self._model.encode(input, show_progress_bar=False).tolist()

    # ── LangChain Embeddings interface ────────────────────────────────
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of document chunks."""
        return self._model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""
        return self._model.encode([text], show_progress_bar=False)[0].tolist()


@lru_cache(maxsize=1)
def get_embedding_function(model_name: str = _DEFAULT_MODEL) -> ComplianceEmbeddings:
    """
    Return a cached singleton embedding function.

    Multiple callers share the same loaded model to avoid redundant memory
    allocation. The cache key includes the model name so a model switch
    (e.g., in tests) creates a fresh instance.
    """
    return ComplianceEmbeddings(model_name)
