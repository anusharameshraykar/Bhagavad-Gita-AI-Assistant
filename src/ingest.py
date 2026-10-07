"""Embed every verse and store it in ChromaDB.

Run:  python -m src.ingest
Re-run whenever you change GITA_EMBEDDER or SEARCH_FIELDS.
"""
from __future__ import annotations
import json
import shutil

import chromadb

from src import config
from src.embedders import get_embedder


def search_text(verse: dict) -> str:
    """The text we embed and keyword-index for a verse (one verse = one chunk)."""
    return "\n".join(verse[f] for f in config.SEARCH_FIELDS if verse.get(f))


def main() -> None:
    verses = json.loads(config.VERSES_PATH.read_text(encoding="utf-8"))
    texts = [search_text(v) for v in verses]
    print(f"Embedding {len(texts)} verses with '{config.EMBEDDER}' ...")
    vectors = get_embedder().embed_documents(texts)

    # start from a clean store so stale vectors from another embedder can't linger
    if config.CHROMA_DIR.exists():
        shutil.rmtree(config.CHROMA_DIR)
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
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
    print(f"Stored {col.count()} verses in {config.CHROMA_DIR}")


if __name__ == "__main__":
    main()
