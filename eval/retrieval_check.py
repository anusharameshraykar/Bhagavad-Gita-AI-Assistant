"""Check that the right verse appears in the top-K results.

Run:  python -m eval.retrieval_check
Add your own questions to QUESTIONS. `expect` = any ONE of these refs counts as a hit.
"""
from __future__ import annotations
from src import config
from src.retrieve import get_retriever

QUESTIONS = [
    # (question, expected refs - any one is a hit)
    ("Explain BG 2.47", ["BG 2.47"]),
    ("what does chapter 18 verse 66 say?", ["BG 18.66"]),
    ("What is karma yoga? How to act without attachment to results?", ["BG 2.47", "BG 2.48", "BG 3.19", "BG 3.9", "BG 5.10", "BG 18.6"]),
    ("What is a sthitaprajna, a person of steady wisdom?", ["BG 2.54", "BG 2.55", "BG 2.56", "BG 2.57", "BG 2.58"]),
    ("How can I control a restless mind?", ["BG 6.34", "BG 6.35", "BG 6.26", "BG 6.5", "BG 6.6"]),
    ("Is the soul immortal? What happens at death?", ["BG 2.20", "BG 2.22", "BG 2.23", "BG 2.27"]),
    ("What are the three gunas sattva rajas and tamas?", ["BG 14.5", "BG 14.6", "BG 14.7", "BG 14.8", "BG 14.9"]),
    ("What are the three gates to hell?", ["BG 16.21"]),
    ("Whenever dharma declines, God manifests himself", ["BG 4.7", "BG 4.8"]),
    ("How to stay balanced in pleasure and pain?", ["BG 2.14", "BG 2.15", "BG 2.38", "BG 12.13", "BG 12.18"]),
    ("Why was Arjuna grieving at the start of the battle?", ["BG 1.28", "BG 1.29", "BG 1.30", "BG 1.47", "BG 2.7"]),
    ("What is bhakti, devotion to God?", ["BG 9.22", "BG 9.26", "BG 9.34", "BG 12.2", "BG 12.8"]),
    ("How many paths are there for moksha?", ["BG 3.3", "BG 5.4", "BG 5.5"]),
    ("What are the paths to moksha?", ["BG 3.3", "BG 5.4", "BG 5.5"]),
]


def main(k: int | None = None) -> None:
    k = k or config.TOP_K
    r = get_retriever()
    hits = 0
    print(f"embedder={config.EMBEDDER}  top_k={k}\n")
    for q, expected in QUESTIONS:
        got = [x["ref"] for x in r.retrieve(q, k)]
        ok = any(e in got for e in expected)
        hits += ok
        print(f"[{'HIT ' if ok else 'MISS'}] {q}")
        print(f"        got      : {', '.join(g.replace('BG ', '') for g in got)}")
        if not ok:
            print(f"        expected : {', '.join(e.replace('BG ', '') for e in expected)}")
    print(f"\nHit rate @{k}: {hits}/{len(QUESTIONS)} = {hits / len(QUESTIONS):.0%}")


if __name__ == "__main__":
    main()
