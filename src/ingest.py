"""Embed every verse and store it in ChromaDB.

Run:  python -m src.ingest
Re-run whenever you change GITA_EMBEDDER or SEARCH_FIELDS.
"""
from __future__ import annotations
import json
import logging
import shutil
import time

import chromadb

from src import config
from src.embedders import get_embedder

logger = logging.getLogger(__name__)


def search_text(verse: dict) -> str:
    """The text we embed and keyword-index for a verse (one verse = one chunk)."""
    return "\n".join(verse[f] for f in config.SEARCH_FIELDS if verse.get(f))


def build_collection(client: chromadb.ClientAPI) -> None:
    started = time.monotonic()
    verses = json.loads(config.VERSES_PATH.read_text(encoding="utf-8"))
    texts = [search_text(v) for v in verses]
    logger.info(
        "Embedding generation started: %d verses, embedder=%s.",
        len(texts),
        config.EMBEDDER,
    )

    embedder_started = time.monotonic()
    embedder = get_embedder()
    logger.info(
        "Embedder '%s' is ready (initialization took %.2f seconds).",
        config.EMBEDDER,
        time.monotonic() - embedder_started,
    )

    embedding_started = time.monotonic()
    vectors = embedder.embed_documents(texts)
    logger.info(
        "Embedding generation completed: %d vectors, dimension=%d, elapsed=%.2f seconds.",
        len(vectors),
        vectors.shape[1],
        time.monotonic() - embedding_started,
    )

    storage_started = time.monotonic()
    col = client.create_collection(
        config.COLLECTION,
        metadata={"hnsw:space": "cosine", "embedder": config.EMBEDDER},
    )
    col.add(
        ids=[v["ref"] for v in verses],
        embeddings=vectors.tolist(),
        documents=texts,
        metadatas=[{"chapter": v["chapter"], "verse": v["verse"]} for v in verses],
    )
    logger.info(
        "Stored %d verses in Chroma at %s (write took %.2f seconds; total build %.2f seconds).",
        col.count(),
        config.CHROMA_DIR,
        time.monotonic() - storage_started,
        time.monotonic() - started,
    )


def main() -> None:
    # start from a clean store so stale vectors from another embedder can't linger
    if config.CHROMA_DIR.exists():
        shutil.rmtree(config.CHROMA_DIR)
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    build_collection(client)


if __name__ == "__main__":
    main()
