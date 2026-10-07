"""Stage 1b (part 2): retrieval.

    from src.retrieve import retrieve
    results = retrieve("what is sthitaprajna?")
    results = retrieve("explain BG 2.47")          # direct lookup, no search

Strategy
  1. Direct reference lookup ("2.47", "chapter 2 verse 47") -> exact verse, skip search.
  2. Query expansion (src/glossary.py) bridges Sanskrit terms / vocabulary gaps.
  3. Otherwise hybrid search: vector similarity + BM25 keywords, merged with
     Reciprocal Rank Fusion (RRF). Vectors capture meaning ("fear of failure"),
     BM25 catches exact Sanskrit terms ("sthitaprajna", "sankhya").
"""
from __future__ import annotations

import json
import logging
import re
import unicodedata
from functools import lru_cache

import chromadb
from rank_bm25 import BM25Okapi

from src import config
from src.embedders import get_embedder
from src.glossary import expand_keywords, expand_query

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# Text normalisation (so "śraddhā", "shraddha" and "Shraddha" all match)
# ----------------------------------------------------------------------------
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "of", "to", "in", "on", "and",
    "or", "for", "with", "what", "does", "do", "did", "how", "why", "who", "which",
    "say", "says", "said", "about", "gita", "bhagavad", "tell", "me", "explain",
    "meaning", "mean", "means", "i", "my", "you", "your", "it", "this", "that",
    "can", "should", "will", "krishna", "arjuna",
}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def tokenize(text: str) -> list[str]:
    """Lowercase, strip diacritics, split on non-alphanumerics. Hyphenated Sanskrit
    compounds (karma-phala-hetuh) are indexed both split AND joined, so a query for
    'sthitaprajna' can hit 'sthita-prajna'."""
    tokens: list[str] = []
    for word in normalize(text).split():
        parts = re.findall(r"[a-z0-9]+", word)
        tokens.extend(parts)
        if len(parts) > 1:
            tokens.append("".join(parts))
    return tokens


# ----------------------------------------------------------------------------
# Direct reference lookup
# ----------------------------------------------------------------------------
_REF_PATTERNS = [
    # "2.47", "2:47", "BG 2.47", "Gita 2.47"
    re.compile(r"(?<![\d.])(\d{1,2})\s*[.:]\s*(\d{1,3})(?![\d])"),
    # "chapter 2 verse 47", "ch 2, v 47", "chapter 2 shloka 47"
    re.compile(
        r"\b(?:chapter|ch)\.?\s*(\d{1,2})\s*[,;]?\s*(?:verse|shloka|sloka|v)\.?\s*(\d{1,3})\b",
        re.I,
    ),
    # "verse 47 of chapter 2"
    re.compile(
        r"\b(?:verse|shloka|sloka)\s*(\d{1,3})\s*(?:of|in|from)\s*(?:chapter|ch)\.?\s*(\d{1,2})\b",
        re.I,
    ),
]


def parse_references(query: str, valid_refs: set[str]) -> list[str]:
    """Return verse refs ('BG 2.47') mentioned in the query, in order, only if they exist."""
    found: list[tuple[int, str]] = []
    for i, pat in enumerate(_REF_PATTERNS):
        for m in pat.finditer(query):
            a, b = int(m.group(1)), int(m.group(2))
            chapter, verse = (b, a) if i == 2 else (a, b)  # pattern 3 is verse-first
            ref = f"BG {chapter}.{verse}"
            if ref in valid_refs:
                found.append((m.start(), ref))
    ordered: list[str] = []
    for _, ref in sorted(found):
        if ref not in ordered:
            ordered.append(ref)
    return ordered


# ----------------------------------------------------------------------------
# Retriever
# ----------------------------------------------------------------------------
class Retriever:
    def __init__(self):
        logger.info("Loading verse data and building keyword index.")
        self.verses = json.loads(config.VERSES_PATH.read_text(encoding="utf-8"))
        self.by_ref = {v["ref"]: v for v in self.verses}
        self.refs = [v["ref"] for v in self.verses]

        # keyword index
        docs = ["\n".join(v[f] for f in config.SEARCH_FIELDS if v.get(f)) for v in self.verses]
        tokenized = [tokenize(d) for d in docs]
        self.bm25 = BM25Okapi(tokenized)
        self.vocab = {t for doc in tokenized for t in doc}

        # vector index
        client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        self.col = client.get_collection(config.COLLECTION)
        built_with = (self.col.metadata or {}).get("embedder")
        if built_with and built_with != config.EMBEDDER:
            raise RuntimeError(
                f"Index was built with '{built_with}' but GITA_EMBEDDER='{config.EMBEDDER}'. "
                "Re-run `python -m src.ingest` or set the matching env var."
            )
        self.embedder = get_embedder()
        logger.info("Retriever ready with %d verses.", len(self.verses))

    # -- individual retrievers ------------------------------------------------
    def vector_search(self, query: str, n: int) -> list[str]:
        return [ref for ref, _ in self.vector_search_with_distances(query, n)]

    def vector_search_with_distances(self, query: str, n: int) -> list[tuple[str, float]]:
        logger.info("Running vector search (candidate limit=%d).", n)
        q = self.embedder.embed_query(query)
        res = self.col.query(
            query_embeddings=[q.tolist()],
            n_results=n,
            include=["distances"],
        )
        matches = list(zip(res["ids"][0], res["distances"][0]))
        logger.info(
            "Vector search returned %d matches; closest distance=%s.",
            len(matches),
            f"{matches[0][1]:.3f}" if matches else "none",
        )
        return matches

    def _expand(self, tokens: list[str]) -> list[str]:
        """Prefix-expand longer query words against the vocabulary, so the query
        'sthitaprajna' also matches 'sthitaprajnasya' (Sanskrit inflection)."""
        out = list(tokens)
        for t in tokens:
            if len(t) >= 5:
                out.extend(sorted(w for w in self.vocab if w.startswith(t) and w != t)[:5])
        return out

    def keyword_search(self, query: str, n: int) -> list[str]:
        tokens = [t for t in tokenize(query) if t not in STOPWORDS]
        if not tokens:
            logger.info("Keyword search skipped: no searchable terms.")
            return []
        scores = self.bm25.get_scores(self._expand(tokens))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
        matches = [self.refs[i] for i in ranked if scores[i] > 0]
        logger.info("Keyword search returned %d matches.", len(matches))
        return matches

    # -- public API -------------------------------------------------------------
    def retrieve(self, query: str, k: int | None = None) -> list[dict]:
        k = k or config.TOP_K
        logger.info("Hybrid retrieval started (top_k=%d).", k)

        direct = parse_references(query, set(self.refs))
        if direct:
            logger.info("Direct reference lookup found %d verses.", min(len(direct), k))
            return [self._pack(r, found_by=["reference"]) for r in direct[:k]]

        # Sanskrit terms -> the words the verses use (separate expansions; see src/glossary.py)
        vec = self.vector_search_with_distances(expand_query(query), config.CANDIDATES)
        vector_distances = dict(vec)
        vec_refs = [ref for ref, _ in vec]
        kw = self.keyword_search(expand_keywords(query), config.CANDIDATES)

        scores: dict[str, float] = {}
        origin: dict[str, list[str]] = {}
        for name, ranking in (("vector", vec_refs), ("keyword", kw)):
            for rank, ref in enumerate(ranking):
                scores[ref] = scores.get(ref, 0.0) + 1.0 / (config.RRF_K + rank + 1)
                origin.setdefault(ref, []).append(f"{name}#{rank + 1}")

        top = sorted(scores, key=scores.get, reverse=True)[:k]
        results = [
            self._pack(
                r,
                score=scores[r],
                found_by=origin[r],
                vector_distance=vector_distances.get(r),
            )
            for r in top
        ]
        logger.info("Hybrid retrieval completed with %d fused results.", len(results))
        return results

    def _pack(
        self,
        ref: str,
        score: float = 1.0,
        found_by: list[str] | None = None,
        vector_distance: float | None = None,
    ) -> dict:
        v = self.by_ref[ref]
        return {
            "ref": ref,
            "chapter": v["chapter"],
            "verse": v["verse"],
            "sanskrit": v["sanskrit"],
            "transliteration": v["transliteration"],
            "translation": v["translation"],
            "translation_alt": v.get("translation_alt", ""),
            "word_meanings": v.get("word_meanings", ""),
            "score": round(score, 5),
            "found_by": found_by or [],
            "vector_distance": vector_distance,
        }


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    return Retriever()


def retrieve(query: str, k: int | None = None) -> list[dict]:
    return get_retriever().retrieve(query, k)


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "What does the Gita say about fear of failure?"
    print(f"Q: {q}\n")
    for r in retrieve(q):
        print(f"{r['ref']:<10} {r['found_by']}  score={r['score']}")
        print(f"   {r['translation'][:160]}")
