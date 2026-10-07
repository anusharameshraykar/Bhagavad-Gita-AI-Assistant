"""Diagnose a retrieval miss: where do the expected verses rank?

    python -m eval.rank_check c05 c07          # question ids from eval/questions.json
    python -m eval.rank_check "When does God take birth?" BG4.7 BG4.8   # ad-hoc

For each expected verse it prints its rank in vector search, keyword search and the
fused (hybrid) list. Rank 8 means "raising top-K would fix it"; rank 60+ means retrieval
itself needs work (query expansion, a better embedder, a reranker).
"""
from __future__ import annotations

import json
import sys

from eval.run_eval import QUESTIONS_PATH
from src.glossary import expand_keywords, expand_query
from src.retrieve import get_retriever

DEPTH = 100


def rank_of(ref: str, ranking: list) -> str:
    return str(ranking.index(ref) + 1) if ref in ranking else f">{len(ranking)}"


def main(argv: list) -> int:
    if not argv:
        print(__doc__)
        return 1
    r = get_retriever()
    by_id = {q["id"]: q for q in json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))}
    jobs = []
    if all(a in by_id for a in argv):
        jobs = [(by_id[a]["question"], by_id[a].get("expect", [])) for a in argv]
    else:
        exp = [("BG " + a[2:].lstrip(" ")) for a in argv[1:]]
        jobs = [(argv[0], exp)]

    for question, expected in jobs:
        expanded = expand_query(question)          # same expansions retrieve() applies
        vec = r.vector_search(expanded, DEPTH)
        kw = r.keyword_search(expand_keywords(question), DEPTH)
        hyb = [x["ref"] for x in r.retrieve(question, 40)]
        print(f"\nQ: {question}")
        if expanded != question:
            print(f"   expanded: {expanded[len(question):].lstrip('. ')}")
        print(f"   {'verse':<9} {'vector':>7} {'keyword':>8} {'hybrid':>7}")
        for e in expected:
            print(f"   {e:<9} {rank_of(e, vec):>7} {rank_of(e, kw):>8} {rank_of(e, hyb):>7}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
