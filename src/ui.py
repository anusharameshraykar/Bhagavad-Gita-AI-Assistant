"""Stage 1e: Streamlit components shared by the chat page and the verse browser."""
from __future__ import annotations

import json

import streamlit as st

from src import config
from src.ui_helpers import (DISCLAIMER, highlight_citations, is_plain_reply, lines_markdown,
                            snippet)


@st.cache_data(show_spinner=False)
def load_verses() -> list:
    if not config.VERSES_PATH.exists():
        st.error("Verse data not found. In a terminal run:  `python -m src.build_data`")
        st.stop()
    return json.loads(config.VERSES_PATH.read_text(encoding="utf-8"))


def verse_card(
    v: dict,
    expanded: bool = False,
    word_meanings_expandable: bool = True,
) -> None:
    """One verse: Sanskrit, transliteration, translation, and word meanings."""
    with st.expander(f"**{v['ref']}**  ·  {snippet(v['translation'], 80)}", expanded=expanded):
        st.markdown(f"#### {lines_markdown(v['sanskrit'])}")
        st.markdown("*" + lines_markdown(v["transliteration"]).replace("  \n", "*  \n*") + "*")
        st.markdown(f"> {v['translation']}")
        if v.get("word_meanings"):
            if word_meanings_expandable:
                with st.expander("Word meanings"):
                    st.markdown("\n".join(
                        f"- {meaning.strip()}"
                        for meaning in v["word_meanings"].split(";")
                        if meaning.strip()
                    ))
            else:
                st.markdown("**Word meanings**")
                st.markdown("\n".join(
                    f"- {meaning.strip()}"
                    for meaning in v["word_meanings"].split(";")
                    if meaning.strip()
                ))


def render_answer(res, show_debug: bool = False) -> None:
    """Render an Answer: text, warnings, cited verse cards, other retrieved verses, debug info."""
    st.markdown(highlight_citations(res.text))
    if res.warning:
        st.warning(res.warning)

    if res.web_sources:
        st.caption("Web sources")
        for index, source in enumerate(res.web_sources, start=1):
            st.link_button(f"[{index}] {snippet(source['title'], 80)}", source["url"])

    plain = is_plain_reply(res.flags)
    if res.cited_verses and not plain:
        st.caption("Verses cited (tap to read)")
        for v in res.cited_verses:
            verse_card(v)

        cited = {v["ref"] for v in res.cited_verses}
        others = [v for v in res.retrieved if v["ref"] not in cited]
        if others:
            with st.expander("Other verses retrieved"):
                for v in others:
                    st.markdown(f"**{v['ref']}**: {snippet(v['translation'], 140)}")

    if show_debug:
        with st.expander("Debug"):
            st.write({
                "retrieved": [v["ref"] for v in res.retrieved],
                "cited": [v["ref"] for v in res.cited_verses],
                "attempts": res.attempts,
                "flags": res.flags,
            })


def sidebar_footer() -> None:
    st.sidebar.divider()
    st.sidebar.caption(DISCLAIMER)
