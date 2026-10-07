from src.ui_helpers import (EXAMPLE_QUESTIONS, chapter_verse_numbers, demo_llm, highlight_citations,
                            is_plain_reply, lines_markdown, snippet)


def test_highlight_citations_bolds_each_once():
    assert highlight_citations("See [BG 2.47] and [BG 3.19].") == "See **[BG 2.47]** and **[BG 3.19]**."
    assert highlight_citations("Already **[BG 2.47]** bold") == "Already **[BG 2.47]** bold"
    assert highlight_citations("no citations here") == "no citations here"


def test_lines_markdown_keeps_line_breaks():
    assert lines_markdown("line one\n\n line two \n") == "line one  \nline two"


def test_snippet_truncates_on_words():
    assert snippet("short text", 50) == "short text"
    assert snippet("word " * 40, 30).endswith("…") and len(snippet("word " * 40, 30)) <= 30


def test_plain_reply_flags():
    for f in ("crisis", "referred", "declined", "refused", "no_retrieval"):
        assert is_plain_reply([f])
    assert not is_plain_reply(["retried"]) and not is_plain_reply([])


def test_demo_llm_cites_only_provided_verses():
    msg = [{"role": "system", "content": "s"}, {"role": "user", "content": "SOURCE VERSES:\n[BG 2.47]\nT\n\n[BG 3.19]\nT\n\nQUESTION: x"}]
    out = demo_llm(msg)
    assert "[BG 2.47]" in out and "[BG 3.19]" in out
    assert demo_llm([{"role": "system", "content": "s"}, {"role": "user", "content": "nothing"}]).startswith("I couldn't find")


def test_chapter_verse_numbers():
    verses = [{"chapter": 1, "verse": 1}, {"chapter": 1, "verse": 2}, {"chapter": 2, "verse": 1}]
    assert chapter_verse_numbers(verses, 1) == [1, 2]


def test_examples_are_questions():
    assert len(EXAMPLE_QUESTIONS) == 4 and all(q.endswith("?") or q.startswith("Explain") for q in EXAMPLE_QUESTIONS)
