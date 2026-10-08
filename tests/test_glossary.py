import json
import re

import pytest

from src import config
from src.glossary import GLOSSARY, expand_keywords, expand_query
from src.retrieve import tokenize


def test_expands_known_terms_and_keeps_the_original_question():
    q = "What are the three gunas?"
    out = expand_query(q)
    assert out.startswith(q) and "purity passion ignorance" in out


def test_vocabulary_gap_terms_expand():
    assert "reincarnate" in expand_query("When does God take birth in the world?")
    assert "reincarnate" in expand_query("What is an avatar?")
    assert "right action" in expand_query("What is karma-yoga?")
    assert "subhadra" in expand_query("Does Krishna talk about Abhimanyu?")
    assert "yama death governors" in expand_keywords("Does Krishna talk about Shani and Yama?")


def test_moksha_path_questions_expand_to_the_explicit_twofold_path():
    for question in (
        "How many paths are there for moksha?",
        "What are the paths to moksha?",
    ):
        expanded = expand_keywords(question)
        assert "twofold path wisdom knowledge action yoga" in expanded


def test_unrelated_questions_are_unchanged():
    for q in ("How can I control a restless mind?", "Explain BG 2.47", "What is anger?"):
        assert expand_query(q) == q


def test_expansion_is_applied_to_the_users_words_only():
    # the avatar expansion contains 'righteousness', 'reborn' etc.; none should trigger more entries
    out = expand_query("avatar")
    assert out.count(". ") == 1


def test_every_expansion_word_exists_in_the_dataset():
    """Stops us adding words that look right but never occur in the verses."""
    if not config.VERSES_PATH.exists():
        pytest.skip("run `python -m src.build_data` first")
    verses = json.loads(config.VERSES_PATH.read_text(encoding="utf-8"))
    vocab = {t for v in verses for f in config.SEARCH_FIELDS for t in tokenize(v.get(f, ""))}
    stop = {"of", "the", "to", "from", "and", "in", "without", "age", "a"}
    missing = []
    for _, expansion, _kw in GLOSSARY:
        for w in re.findall(r"[a-z]+", expansion.lower()):
            if w not in stop and w not in vocab:
                missing.append(w)
    assert not missing, f"words never used in the dataset: {sorted(set(missing))}"


def test_keyword_expansion_is_a_subset_of_vector_expansion():
    # yajna's Sanskrit term already matches BM25, so it expands for vector search only
    assert "sacrifice offering" in expand_query("What is yajna?")
    assert expand_keywords("What is yajna?") == "What is yajna?"
    # gunas has no direct BM25 match (dataset token is 'gunah'), so it expands for both
    assert "purity passion ignorance" in expand_keywords("What are the three gunas?")
    assert "reincarnate" in expand_keywords("When does God take birth?")


def test_expansion_can_be_switched_off(monkeypatch):
    from src import config
    monkeypatch.setattr(config, "QUERY_EXPANSION", False)
    assert expand_query("What are the three gunas?") == "What are the three gunas?"
    assert expand_keywords("What are the three gunas?") == "What are the three gunas?"
