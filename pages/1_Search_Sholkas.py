"""Gita Q&A: verse browser. Read any verse directly (no model needed)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # so `import src` works

import streamlit as st  # noqa: E402

from src.ui import load_verses, sidebar_footer, verse_card  # noqa: E402
from src.ui_helpers import chapter_verse_numbers  # noqa: E402

st.set_page_config(page_title="Search Sholkas", page_icon="📖", layout="centered")
st.title("Search Sholkas")
st.caption("Read the Sanskrit, transliteration and translation of any verse. No AI involved.")

verses = load_verses()
by_key = {(v["chapter"], v["verse"]): v for v in verses}

chapter = st.selectbox("Chapter", list(range(1, 19)), key="chapter")
numbers = chapter_verse_numbers(verses, chapter)
verse_no = st.selectbox(f"Verse (chapter {chapter} has {len(numbers)})", numbers, key="verse")

verse_card(by_key[(chapter, verse_no)], expanded=True, word_meanings_expandable=False)

with st.expander(f"All verses in chapter {chapter}"):
    for n in numbers:
        v = by_key[(chapter, n)]
        st.markdown(f"**{v['ref']}**: {v['translation']}")

sidebar_footer()
