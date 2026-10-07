"""Runtime setup helpers for deployments that don't ship a built vector index."""
from __future__ import annotations

import logging
import time

import chromadb

from src import config

logger = logging.getLogger(__name__)


def ensure_chroma_collection() -> bool:
    """Build the configured collection if it is missing; return whether it was built."""
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

    logger.info(
        "Chroma collection '%s' is missing; starting deployment-time index build from %s.",
        config.COLLECTION,
        config.VERSES_PATH,
    )
    from src.ingest import build_collection

    build_collection(client)
    logger.info(
        "Deployment-time index build completed (total startup build time %.2f seconds).",
        time.monotonic() - started,
    )
    return True
