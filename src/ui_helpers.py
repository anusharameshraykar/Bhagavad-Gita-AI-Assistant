"""Stage 1e: small pure helpers for the UI (no Streamlit import, so they are easy to test)."""
from __future__ import annotations

import re

EXAMPLE_QUESTIONS = [
    "What is karma yoga?",
    "How can I control a restless mind?",
    "Explain BG 2.47",
    "What are the three gunas?",
]

DISCLAIMER = (
    "AI-generated from retrieved verses, which can still be wrong or incomplete. "
    "Please check the verses shown, the original text, or a teacher. "
    "Not medical, legal, financial or mental-health advice."
)

# one [BG 2.47] (not already bold) -> **[BG 2.47]**
_CITATION = re.compile(r"(?<!\*)\[(BG \d{1,2}\.\d{1,3})\](?!\*)")
_SPECIAL_FLAGS = {
    "crisis",
    "referred",
    "declined",
    "refused",
    "no_retrieval",
    "unrelated",
    "relevance_uncertain",
}


def highlight_citations(text: str) -> str:
    """Make verse citations stand out in the chat answer."""
    return _CITATION.sub(r"**[\1]**", text)


def lines_markdown(text: str) -> str:
    """Keep the line breaks of a verse when rendered as markdown."""
    return "  \n".join(ln.strip() for ln in text.split("\n") if ln.strip())


def snippet(text: str, n: int = 110) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def is_plain_reply(flags: list) -> bool:
    """Replies that are not verse-based answers (crisis, referral, refusal): no verse cards."""
    return any(f in _SPECIAL_FLAGS for f in flags)


def demo_llm(messages: list) -> str:
    """Stand-in model for running the UI without Ollama (GITA_MOCK_LLM=1)."""
    if messages[0]["content"].startswith("Classify whether the user's question"):
        return "RELATED"
    refs = re.findall(r"^\[(BG \d+\.\d+)\]$", messages[1]["content"], re.M)
    if not refs:
        return "I couldn't find this in the verses retrieved."
    return (f"(Demo mode: no language model is running.) The most relevant verse found is [{refs[0]}]"
            + (f", and also [{refs[1]}]." if len(refs) > 1 else "."))


def chapter_verse_numbers(verses: list, chapter: int) -> list:
    return [v["verse"] for v in verses if v["chapter"] == chapter]
