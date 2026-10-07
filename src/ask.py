"""Ask the Gita a question from the command line.

    python -m src.ask "What is karma yoga?"
    python -m src.ask "Explain BG 2.47" --model llama3.2:3b
    python -m src.ask "How to control anger?" --debug      # show retrieval + validation details
"""
from __future__ import annotations

import argparse
import functools
import sys
import uuid

from src import config
from src.generate import LLMUnavailable, call_ollama
from src.glossary import expand_query
from src.logging_config import configure_logging
from src.pipeline import answer_question
from src.request_context import bind_request_context


def main() -> int:
    configure_logging()
    p = argparse.ArgumentParser(description="Ask a question about the Bhagavad Gita")
    p.add_argument("question", nargs="+")
    p.add_argument("--model", default=config.OLLAMA_MODEL, help="Ollama model name")
    p.add_argument("-k", type=int, default=config.TOP_K, help="verses to retrieve")
    p.add_argument("--debug", action="store_true", help="show retrieval and validation details")
    args = p.parse_args()
    question = " ".join(args.question)

    llm = functools.partial(call_ollama, model=args.model)
    try:
        with bind_request_context("internal", "cli", uuid.uuid4().hex[:12]):
            res = answer_question(question, k=args.k, llm=llm)
    except LLMUnavailable as e:
        print(f"\n[LLM problem] {e}", file=sys.stderr)
        return 1

    print(f"\nQ: {question}\n")
    print(res.text)

    if res.warning:
        print(f"\n!! {res.warning}")

    if res.cited_verses:
        print("\n--- Verses cited ---")
        for v in res.cited_verses:
            print(f"\n{v['ref']}")
            print(v["sanskrit"])
            print(v["transliteration"])
            print(f"“{v['translation']}”")

    if res.web_sources:
        print("\n--- Web sources ---")
        for i, source in enumerate(res.web_sources, start=1):
            print(f"[{i}] {source['title']} — {source['url']}")

    if args.debug:
        print("\n--- Debug ---")
        print("model     :", args.model)
        expanded = expand_query(question)
        if expanded != question:
            print("expanded  :", expanded[len(question):].lstrip(". "))
        print("retrieved :", ", ".join(v["ref"].replace("BG ", "") for v in res.retrieved))
        print("attempts  :", res.attempts, "| flags:", res.flags or "none")
        if res.raw_text and res.raw_text != res.text:
            print("raw answer (before cleanup):\n", res.raw_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
