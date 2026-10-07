"""Runtime setup helpers for the packaged Chroma vector index."""
from __future__ import annotations

import logging
import time

import chromadb

from src import config

logger = logging.getLogger(__name__)


def ensure_chroma_collection() -> bool:
    """Check that the packaged verse collection exists without building it at startup."""
    started = time.monotonic()
    logger.info(
        "Checking Chroma index at %s (collection=%s, embedder=%s).",
        config.CHROMA_DIR,
        config.COLLECTION,
        config.EMBEDDER,
    )
    if not config.VERSES_PATH.exists():
        raise FileNotFoundError(
            f"Verse dataset not found at {config.VERSES_PATH}. "
            "Include data/gita_verses.json in the deployed repository."
        )

    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    collections = client.list_collections()
    collection_names = {
        getattr(collection, "name", collection)
        for collection in collections
    }
    if config.COLLECTION in collection_names:
        logger.info(
            "Chroma collection '%s' already exists; skipping index and embedding build "
            "(check completed in %.2f seconds).",
            config.COLLECTION,
            time.monotonic() - started,
        )
        return False

    message = (
        f"Packaged Chroma collection '{config.COLLECTION}' is missing from {config.CHROMA_DIR}. "
        "Build it before deployment with `GITA_EMBEDDER=tfidf python -m src.ingest`, "
        "then include both data/chroma and data/tfidf.pkl."
    )
    logger.error("%s", message)
    raise FileNotFoundError(message)
