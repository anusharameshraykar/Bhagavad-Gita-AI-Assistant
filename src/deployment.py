"""Runtime setup helpers for deployments that don't ship a built vector index."""
from __future__ import annotations

import chromadb

from src import config


def ensure_chroma_collection() -> bool:
    """Build the configured collection if it is missing; return whether it was built."""
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
        return False

    from src.ingest import build_collection

    build_collection(client)
    return True
