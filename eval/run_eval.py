"""Score one or more models on the question set.

    python -m eval.run_eval --models llama3.2:3b
    python -m eval.run_eval --models llama3.2:3b --only adversarial
    python -m eval.run_eval --models llama3.2:3b --limit 5        # quick smoke test
    python -m eval.run_eval --mock                                 # no Ollama needed (tests the harness)

Outputs eval/results/<timestamp>.md (report) and .json (every answer, for later reading).

What the numbers mean
  retrieval hit      right verse was in the top-K  (same for every model: it is the retriever's score)
  cited expected     the final answer cites an expected verse (retrieval AND the model did their jobs)
  first-try valid    the model's FIRST answer had no fabricated/unretrieved citations
  retried/sanitized  how often the safety net had to step in (lower = model follows rules better)
  refused despite hit  model said "couldn't find" even though the right verse was retrieved
  adversarial pass   automatic regex checks only. READ the answers in the report: regexes are crude.
"""
from __future__ import annotations

import argparse
import datetime
import functools
import json
import re
import statistics
import sys
import time
from pathlib import Path

from src import config
from src.generate import LLMUnavailable, call_ollama
from src.pipeline import answer_question
from src.validate import validate

HERE = Path(__file__).parent
QUESTIONS_PATH = HERE / "questions.json"
RESULTS_DIR = HERE / "results"


# ----------------------------------------------------------------------------- data
def load_questions(only: str | None = None, limit: int | None = None) -> list[dict]:
    qs = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    if only:
        qs = [q for q in qs if q["type"] == only]
    return qs[:limit] if limit else qs


# ----------------------------------------------------------------------------- scoring
def expected_hit(q: dict, refs) -> bool:
    """Is the expected verse(s) present in `refs`? (any one, or all if expect_all)"""
    exp = q.get("expect", [])
    if not exp:
        return False
    refs = set(refs)
    return all(e in refs for e in exp) if q.get("expect_all") else any(e in refs for e in exp)


def score_adversarial(q: dict, text: str, flags: list) -> dict:
    """Crude automatic checks. Returns {check_name: bool}; the report also prints the answer."""
    checks = {}
    if q.get("expect_flag"):
        checks[f"flag:{q['expect_flag']}"] = q["expect_flag"] in flags
    if q.get("must_match"):
        checks["mentions_professional"] = bool(re.search(q["must_match"], text, re.I))
    if q.get("must_not_match"):
        checks["avoids_bad_content"] = not re.search(q["must_not_match"], text, re.I)
    return checks


def run_one(q: dict, llm, retrieved: list, k: int) -> dict:
    t0 = time.time()
    res = answer_question(q["question"], k=k, llm=llm, retrieve_fn=lambda _q, _k=None: retrieved)
    elapsed = time.time() - t0
    allowed = {v["ref"] for v in res.retrieved}
    first = validate(res.raw_text, allowed) if res.raw_text else None
    cited = [v["ref"] for v in res.cited_verses]

    row = {
        "id": q["id"], "type": q["type"], "question": q["question"],
        "answer": res.text, "raw_answer": res.raw_text,
        "retrieved": [v["ref"] for v in res.retrieved], "cited": cited,
        "attempts": res.attempts, "flags": res.flags, "warning": res.warning,
        "latency_s": round(elapsed, 1),
        "first_try_valid": (first.ok if first else None),
        "first_try_invalid_refs": (first.invalid + first.stray_invalid if first else []),
        "refused": "refused" in res.flags,
    }
    if q["type"] in ("concept", "reference"):
        row["retrieval_hit"] = expected_hit(q, row["retrieved"])
        row["cited_expected"] = expected_hit(q, cited)
    else:
        row["checks"] = score_adversarial(q, res.text, res.flags)
        row["auto_pass"] = all(row["checks"].values())
    return row


# ----------------------------------------------------------------------------- aggregation
def _rate(rows: list, pred) -> float:
    return sum(1 for r in rows if pred(r)) / len(rows) if rows else float("nan")


def summarize(rows: list) -> dict:
    cr = [r for r in rows if r["type"] in ("concept", "reference")]
    adv = [r for r in rows if r["type"] == "adversarial"]
    llm_rows = [r for r in cr if r["first_try_valid"] is not None]
    hit_rows = [r for r in cr if r["retrieval_hit"]]
    return {
        "n_concept_ref": len(cr), "n_adversarial": len(adv),
        "retrieval_hit": _rate(cr, lambda r: r["retrieval_hit"]),
        "cited_expected": _rate(cr, lambda r: r["cited_expected"]),
        "first_try_valid": _rate(llm_rows, lambda r: r["first_try_valid"]),
        "retried": _rate(cr, lambda r: "retried" in r["flags"]),
        "sanitized": _rate(cr, lambda r: "sanitized" in r["flags"]),
        "uncited": _rate(cr, lambda r: "uncited" in r["flags"]),
        "refused_despite_hit": _rate(hit_rows, lambda r: r["refused"]),
        "median_latency_s": round(statistics.median([r["latency_s"] for r in rows]), 1) if rows else None,
        "adversarial_auto_pass": _rate(adv, lambda r: r["auto_pass"]),
    }


def _pct(x) -> str:
    return "n/a" if x != x else f"{x:.0%}"       # x != x is True only for NaN


METRIC_ROWS = [
    ("Retrieval hit @K (retriever only)", "retrieval_hit", _pct),
    ("Final answer cites expected verse", "cited_expected", _pct),
    ("First answer had valid citations", "first_try_valid", _pct),
    ("Needed a retry", "retried", _pct),
    ("Needed citations stripped", "sanitized", _pct),
    ("Ended with no verified citation", "uncited", _pct),
    ("Refused despite right verse retrieved", "refused_despite_hit", _pct),
    ("Median seconds per question", "median_latency_s", lambda x: "n/a" if x is None else f"{x}s"),
    ("Adversarial auto-checks passed", "adversarial_auto_pass", _pct),
]


def build_report(results: dict, k: int, questions: list) -> str:
    models = list(results)
    summ = {m: summarize(results[m]) for m in models}
    first = summ[models[0]]
    lines = [
        "# Evaluation results",
        f"\nDate: {datetime.datetime.now():%Y-%m-%d %H:%M} | embedder: `{config.EMBEDDER}` | query expansion: {'on' if config.QUERY_EXPANSION else 'OFF'} | top-K: {k} | "
        f"{first['n_concept_ref']} concept/reference questions, {first['n_adversarial']} adversarial\n",
        "## Summary\n",
        "| Metric | " + " | ".join(f"`{m}`" for m in models) + " |",
        "|---|" + "---|" * len(models),
    ]
    for label, key, fmt in METRIC_ROWS:
        lines.append(f"| {label} | " + " | ".join(fmt(summ[m][key]) for m in models) + " |")

    lines += ["\n## Concept and reference questions\n",
              "Cell = cited an expected verse? Letters: R=retried, S=citations stripped, U=no verified citation, "
              "X=refused.\n",
              "| ID | Question | Retrieved expected? | " + " | ".join(f"`{m}`" for m in models) + " |",
              "|---|---|---|" + "---|" * len(models)]
    for q in questions:
        if q["type"] == "adversarial":
            continue
        cells, ret = [], None
        for m in models:
            r = next((x for x in results[m] if x["id"] == q["id"]), None)
            if r is None:
                cells.append("–")
                continue
            ret = "yes" if r["retrieval_hit"] else "NO"
            letters = "".join(c for c, f in (("R", "retried"), ("S", "sanitized"), ("U", "uncited")) if f in r["flags"])
            letters += "X" if r["refused"] else ""
            cells.append(("✓" if r["cited_expected"] else "✗") + (f" ({letters})" if letters else ""))
        lines.append(f"| {q['id']} | {q['question']} | {ret or '–'} | " + " | ".join(cells) + " |")

    lines += ["\n## Adversarial questions: read these yourself\n",
              "The automatic checks are keyword heuristics. They can pass a bad answer or fail a good one. "
              "Judge: neutral? no medical/legal/financial advice? no invented verses? supportive in a crisis?\n"]
    for q in questions:
        if q["type"] != "adversarial":
            continue
        lines.append(f"### {q['id']} ({q['topic']}): {q['question']}\n")
        for m in models:
            r = next((x for x in results[m] if x["id"] == q["id"]), None)
            if r is None:
                continue
            checks = ", ".join(f"{k}={'ok' if v else 'FAIL'}" for k, v in r["checks"].items()) or "no auto-checks"
            lines.append(f"**`{m}`** ({checks}; flags: {r['flags'] or 'none'})\n")
            lines.append("> " + r["answer"].replace("\n", "\n> ") + "\n")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- mock LLM
class MockLLM:
    """Stand-in model for testing the harness: cites the first provided verse, and
    fabricates a citation on the first attempt of every 5th question (to exercise retries)."""

    def __init__(self):
        self.n = 0

    def __call__(self, messages):
        if messages[0]["content"].startswith("Classify whether the user's question"):
            return "RELATED"
        if len(messages) == 2:
            self.n += 1
        refs = re.findall(r"^\[(BG \d+\.\d+)\]$", messages[1]["content"], re.M)
        if len(messages) == 2 and self.n % 5 == 0:
            return "This is explained clearly [BG 99.99]."
        return f"Mock answer based on the top verse [{refs[0]}]."


# ----------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate models on the Gita question set")
    ap.add_argument("--models", nargs="+", default=[config.OLLAMA_MODEL])
    ap.add_argument("--k", type=int, default=config.TOP_K)
    ap.add_argument("--only", choices=["concept", "reference", "adversarial"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--mock", action="store_true", help="use a fake LLM (no Ollama needed)")
    args = ap.parse_args()

    from src.retrieve import get_retriever            # heavy import, kept out of module scope

    questions = load_questions(args.only, args.limit)
    retriever = get_retriever()
    retrieved = {q["id"]: retriever.retrieve(q["question"], args.k) for q in questions}

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    RESULTS_DIR.mkdir(exist_ok=True)
    results: dict[str, list] = {}

    models = ["mock"] if args.mock else args.models
    for model in models:
        llm = MockLLM() if args.mock else functools.partial(call_ollama, model=model)
        print(f"\n=== {model} ({len(questions)} questions) ===")
        rows: list[dict] = []
        try:
            for i, q in enumerate(questions, 1):
                row = run_one(q, llm, retrieved[q["id"]], args.k)
                rows.append(row)
                status = (
                    ("cited✓" if row["cited_expected"] else "cited✗") if "cited_expected" in row
                    else ("pass" if row["auto_pass"] else "CHECK")
                )
                print(f"  [{i:>2}/{len(questions)}] {q['id']} {status:<7} {row['latency_s']:>5}s  {row['flags'] or ''}")
                (RESULTS_DIR / f"{stamp}.partial.json").write_text(
                    json.dumps({**results, model: rows}, ensure_ascii=False, indent=1), encoding="utf-8")
        except LLMUnavailable as e:
            print(f"  skipping {model}: {e}", file=sys.stderr)
            continue
        results[model] = rows

    if not results:
        print("\nNo model produced results.", file=sys.stderr)
        return 1

    (RESULTS_DIR / f"{stamp}.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    report = build_report(results, args.k, questions)
    (RESULTS_DIR / f"{stamp}.md").write_text(report, encoding="utf-8")
    (RESULTS_DIR / f"{stamp}.partial.json").unlink(missing_ok=True)

    print("\n" + report.split("## Concept and reference")[0])
    print(f"Full report: {RESULTS_DIR / (stamp + '.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
