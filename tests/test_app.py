"""Headless UI tests with Streamlit's AppTest. Skipped unless streamlit, the verse data and
the index are present. Uses a demo LLM, so Ollama is not needed."""
import os

import pytest

from src import config

st_testing = pytest.importorskip("streamlit.testing.v1")
pytestmark = pytest.mark.skipif(
    not (config.VERSES_PATH.exists() and config.CHROMA_DIR.exists()),
    reason="run `python -m src.build_data && python -m src.ingest` first",
)


@pytest.fixture(autouse=True)
def demo_mode(monkeypatch):
    monkeypatch.setenv("GITA_MOCK_LLM", "1")


def run_app(path="app.py"):
    return st_testing.AppTest.from_file(str(config.ROOT / path), default_timeout=120).run()


def test_chat_page_loads_with_examples():
    at = run_app()
    assert not at.exception
    assert any("Ask the Bhagavad Gita" in t.value for t in at.title)
    assert len(at.button) >= 4                      # example questions (+ Clear chat)


def test_question_produces_answer_and_verse_cards():
    at = run_app()
    at.chat_input[0].set_value("What is karma yoga?").run()
    assert not at.exception
    assert len(at.chat_message) == 2                # the question and the answer
    assert any("[BG" in m.value for m in at.markdown)
    assert any("Verses cited" in c.value for c in at.caption)
    assert len(at.expander) >= 1                    # verse cards


def test_direct_reference_shows_that_verse_card():
    at = run_app()
    at.chat_input[0].set_value("Explain BG 2.47").run()
    assert not at.exception
    assert any("BG 2.47" in e.label for e in at.expander)
    card = next(e for e in at.expander if "BG 2.47" in e.label)
    assert any(e.label == "Word meanings" for e in card.expander)
    assert not next(e for e in card.expander if e.label == "Word meanings").proto.expanded


def test_alternate_app_browse_shows_word_meanings():
    at = run_app("src/app.py")
    assert at.radio[0].options == ["Gita Chatbot", "Search Sholkas"]
    assert at.selectbox[0].label == "LLM provider"
    assert at.selectbox[0].options == ["Gemini", "Ollama"]
    at.radio[0].set_value("Search Sholkas").run()
    assert not at.exception
    assert any("Word meanings" in m.value for m in at.markdown)
    assert any("land of dharma" in m.value for m in at.markdown)


def test_alternate_app_gemini_provider_exposes_model_selector():
    at = run_app("src/app.py")
    at.selectbox[0].select("Gemini").run()
    assert not at.exception
    assert any(widget.label == "Gemini model" for widget in at.text_input)


def test_model_options_are_hidden_until_toggled():
    at = run_app()
    assert not at.exception
    assert at.toggle[0].label == "Show model options"
    assert not at.toggle[0].value
    assert not any(widget.label == "LLM provider" for widget in at.selectbox)
    assert not any(widget.label in {"Gemini model", "Groq model", "Ollama model"} for widget in at.text_input)

    at.toggle[0].set_value(True).run()
    assert not at.exception
    assert at.selectbox[0].options == ["Gemini", "Groq", "Ollama"]
    at.selectbox[0].select("Groq").run()
    assert not at.exception
    assert any(widget.label == "Groq model" for widget in at.text_input)


def test_crisis_message_shows_support_text_and_no_verse_cards():
    at = run_app()
    at.chat_input[0].set_value("I want to end my life").run()
    assert not at.exception
    assert any("emergency number" in m.value for m in at.markdown)
    assert not any("Verses cited" in c.value for c in at.caption)


def test_refusal_shows_no_verse_cards():
    at = run_app()
    at.chat_input[0].set_value("Which political party does the Gita support?").run()
    assert not at.exception
    assert any("political positions" in m.value for m in at.markdown)
    assert not any("Verses cited" in c.value for c in at.caption)


def test_search_sholkas_view_shows_selected_verse():
    at = run_app()
    assert not at.exception
    at.radio[0].set_value("Search Sholkas").run()
    assert any("Search Sholkas" in title.value for title in at.title)
    at.selectbox[0].select(2).run()
    at.selectbox[1].select(47).run()
    assert not at.exception
    assert any("karma" in m.value.lower() for m in at.markdown)
    assert any("in prescribed duties" in m.value for m in at.markdown)
