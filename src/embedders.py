"""Embedding backends behind one tiny interface.

    embedder = get_embedder()
    doc_vecs   = embedder.embed_documents(list_of_texts)
    query_vec  = embedder.embed_query("what is karma yoga?")

IMPORTANT: use the SAME embedder for ingestion and querying. If you change
GITA_EMBEDDER, re-run `python -m src.ingest`.
"""
from __future__ import annotations

import logging
import pickle

import numpy as np

from src import config

logger = logging.getLogger(__name__)


class SentenceTransformerEmbedder:
    def __init__(self, key: str):
        from sentence_transformers import SentenceTransformer  # lazy: heavy import

        self.key = key
        model_name = config.EMBEDDER_MODELS[key]
        logger.info("Loading sentence-transformer model %s.", model_name)
        try:
            self.model = SentenceTransformer(model_name)
        except Exception:
            logger.exception("Could not load sentence-transformer model %s.", model_name)
            raise
        self.query_prefix = config.QUERY_PREFIX.get(key, "")
        logger.info("Sentence-transformer model %s is ready.", model_name)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True
        )

    def embed_query(self, text: str) -> np.ndarray:
        return self.model.encode(self.query_prefix + text, normalize_embeddings=True)


class TfidfEmbedder:
    """Offline fallback: no downloads, runs anywhere. Lexical, so weaker than a
    neural model, but lets you test the whole pipeline without internet.
    The fitted vectorizer is saved next to the Chroma store so queries can reuse it."""

    path = config.DATA_DIR / "tfidf.pkl"

    def __init__(self):
        self.vec = None
        self.svd = None
        if self.path.exists():
            self.vec, self.svd = pickle.loads(self.path.read_bytes())

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vec = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2
        )
        x = self.vec.fit_transform(texts)
        self.svd = TruncatedSVD(n_components=256, random_state=0)
        z = self.svd.fit_transform(x)
        self.path.write_bytes(pickle.dumps((self.vec, self.svd)))
        return _normalize(z)

    def embed_query(self, text: str) -> np.ndarray:
        if self.vec is None:
            raise RuntimeError("TF-IDF model not found. Run `python -m src.ingest` first.")
        return _normalize(self.svd.transform(self.vec.transform([text])))[0]


def _normalize(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.clip(n, 1e-9, None)


def get_embedder():
    if config.EMBEDDER == "tfidf":
        return TfidfEmbedder()
    if config.EMBEDDER in config.EMBEDDER_MODELS:
        return SentenceTransformerEmbedder(config.EMBEDDER)
    raise ValueError(f"Unknown GITA_EMBEDDER={config.EMBEDDER!r}")
