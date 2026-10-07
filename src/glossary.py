"""Query expansion for Sanskrit terms and vocabulary gaps.

Why: the translations rarely use Sanskrit words ("guna" shows up as "qualities ... purity,
passion, ignorance") and users rarely use the translators' words ("take birth" vs
"I reincarnate myself"). A small embedding model and BM25 can't bridge that gap, so we
append the vocabulary the verses actually use before searching.

Two expansions, because the two searches behave differently (measured on the dataset):
  expand_query(q)     -> used for VECTOR search: every matching entry is added.
  expand_keywords(q)  -> used for BM25 keyword search: only entries flagged `keyword=True`.
                         Where the Sanskrit word already matches the verses' transliteration
                         (yajna, tapas, atman...) extra English words dilute BM25 and made
                         ranks worse, so those entries are vector-only.

Set GITA_EXPAND=0 to switch expansion off entirely (for A/B testing).

Rules for adding entries
  * Every expansion word must occur in the dataset (tests/test_glossary.py enforces this).
  * Keep expansions short and on-topic; long ones dilute the embedding.
  * Decide `keyword` by measurement: compare ranks with `python -m eval.rank_check`.
  * This patches known gaps; it does not replace a better retriever.
"""
from __future__ import annotations

import re

from src import config

# (pattern, expansion, also_use_for_keyword_search)
GLOSSARY: list = [
    (r"\bgunas?\b|\bgunah\b",
     "gunah qualities of nature purity passion ignorance sattva rajas tamas", True),
    (r"\bavatars?\b|\bavataras?\b|\bincarnat\w*|\b(?:take|takes|taking|took)\s+birth\b|\bdescend\w*",
     "manifest myself whenever decline of righteousness unrighteousness reincarnate reborn "
     "from age to age protect the righteous destroy the wicked", True),
    (r"\babhimanyu\b|\bsubhadra\b",
     "abhimanyu subhadra arjuna", True),
    (r"\byama\b|\bshani\b|\bsani\b",
     "yama death governors", True),
    (r"\bkarma[- ]?yoga\b", "right action selfless action without attachment to results", True),
    (r"\bmoksha\b|\bmoksa\b|\bmukti\b|\bnirvana\b",
     "liberation freedom release from bondage supreme peace", True),
    (r"\bmaya\b", "illusion delusion divine nature", True),
    # vector-only: the Sanskrit term itself already matches in keyword search
    (r"\bbhakti\b", "devotion love worship", False),
    (r"\bjnana\b|\bgyan\b", "knowledge wisdom", False),
    (r"\batman\b|\batma\b", "self soul spirit embodied", False),
    (r"\bsamsara\b|\bsansara\b", "rebirth birth and death", False),
    (r"\byajnas?\b|\byagya\b", "sacrifice offering worship", False),
    (r"\btapas\b|\btapasya\b", "austerity discipline", False),
    (r"\bshraddha\b|\bsraddha\b", "faith", False),
    (r"\bsannyas\w*|\bsanyas\w*|\btyaga\b", "renunciation renounce abandon", False),
    (r"\bvairagya\b", "detachment dispassion", False),
    (r"\bdhyana\b", "meditation concentration", False),
]

_COMPILED = [(re.compile(p, re.I), e, kw) for p, e, kw in GLOSSARY]


def _expand(query: str, keyword_only: bool) -> str:
    if not config.QUERY_EXPANSION:
        return query
    extras = [e for pat, e, kw in _COMPILED if (kw or not keyword_only) and pat.search(query)]
    return query if not extras else f"{query}. " + " ".join(extras)


def expand_query(query: str) -> str:
    """For vector search: append the verses' vocabulary for every glossary term found."""
    return _expand(query, keyword_only=False)


def expand_keywords(query: str) -> str:
    """For BM25: only entries where the Sanskrit term alone would not match."""
    return _expand(query, keyword_only=True)
