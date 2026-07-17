"""
ingest.py
---------
One-time script to build (or rebuild) the ChromaDB vector database from the
policy PDFs in ``data/policies/``.

Run this BEFORE starting the Streamlit app:

    python ingest.py

The script:
  1. Scans data/policies/ for every *.pdf
  2. Loads and chunks all pages
  3. Embeds chunks using SentenceTransformers (all-MiniLM-L6-v2)
  4. Persists the ChromaDB to data/chroma_db/
  5. Writes a .model sentinel so the app can detect embedding-model changes
  6. Writes db_stats.json for the sidebar statistics panel

Re-run any time you add, remove, or replace PDFs.  The existing database is
deleted and rebuilt from scratch on each run.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

# Ensure project root is on the path when run directly
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from loguru import logger

from app.rag.chunker import split_documents
from app.rag.embeddings import _DEFAULT_MODEL as _EMBEDDING_MODEL
from app.rag.loader import load_policies
from app.rag.retriever import build_vectorstore, vectorstore_exists
from app.utils.config import get_settings


def main() -> None:
    cfg = get_settings()
    sentinel_path = cfg.chroma_persist_dir / ".model"

    logger.info("=" * 60)
    logger.info("Compliance Advisory — Knowledge Base Ingestion")
    logger.info("=" * 60)
    logger.info(f"Policies dir : {cfg.policies_dir.resolve()}")
    logger.info(f"ChromaDB dir : {cfg.chroma_persist_dir.resolve()}")
    logger.info(f"Embedding model: {_EMBEDDING_MODEL}")

    # ── Step 1: Check for source PDFs ─────────────────────────
    pdf_files = sorted(cfg.policies_dir.glob("*.pdf"))
    if not pdf_files:
        logger.error(
            f"No PDF files found in {cfg.policies_dir.resolve()}.\n"
            "Add your policy PDFs to data/policies/ and re-run."
        )
        sys.exit(1)

    logger.info(f"Found {len(pdf_files)} PDF(s):")
    for f in pdf_files:
        logger.info(f"  · {f.name}")

    # ── Step 2: Wipe existing database ────────────────────────
    if cfg.chroma_persist_dir.exists():
        logger.info(f"Deleting existing ChromaDB at {cfg.chroma_persist_dir} …")
        shutil.rmtree(cfg.chroma_persist_dir, ignore_errors=True)
        logger.info("Old ChromaDB removed.")

    cfg.chroma_persist_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 3: Load PDFs ──────────────────────────────────────
    logger.info("Loading PDF pages …")
    docs, infos = load_policies()

    if not docs:
        logger.error("No documents were loaded — check loader errors above.")
        sys.exit(1)

    total_pages = sum(i.pages for i in infos)
    logger.info(f"Loaded {total_pages} page(s) from {len(infos)} document(s).")

    # ── Step 4: Chunk ──────────────────────────────────────────
    logger.info("Splitting pages into chunks …")
    chunks, chunk_counts = split_documents(docs)

    for info in infos:
        info.chunks = chunk_counts.get(info.filename, 0)

    logger.info(f"Created {len(chunks)} chunk(s).")

    # ── Step 5: Embed and persist ──────────────────────────────
    logger.info("Embedding chunks and building ChromaDB …")
    try:
        vs = build_vectorstore(chunks)
    except RuntimeError as exc:
        logger.error(f"Build failed: {exc}")
        sys.exit(1)

    # ── Step 6: Write .model sentinel ─────────────────────────
    sentinel_path.write_text(_EMBEDDING_MODEL)
    logger.info(f"Wrote embedding sentinel: {sentinel_path} → '{_EMBEDDING_MODEL}'")

    # ── Done ───────────────────────────────────────────────────
    logger.success("=" * 60)
    logger.success("Ingestion complete!")
    logger.success(f"  Documents : {len(infos)}")
    logger.success(f"  Pages     : {total_pages}")
    logger.success(f"  Chunks    : {len(chunks)}")
    logger.success(f"  ChromaDB  : {cfg.chroma_persist_dir.resolve()}")
    logger.success("=" * 60)
    logger.success("You can now start the app:  streamlit run main.py")


if __name__ == "__main__":
    main()
