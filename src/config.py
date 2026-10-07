"""Central settings. Override any of them with environment variables."""
from __future__ import annotations
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
VERSES_PATH = DATA_DIR / "gita_verses.json"   # cleaned dataset (output of build_data.py)
CHROMA_DIR = DATA_DIR / "chroma"               # vector store (output of ingest.py)
COLLECTION = "gita"

# ---- Data source (public-domain repo; see README "Licensing") -----------------
RAW_BASE = "https://raw.githubusercontent.com/gita/gita/main/data"

# Author ids in gita/gita translation.json
TRANSLATION_ID = 21       # Shri Purohit Swami (1935) - shown to users
TRANSLATION_ALT_ID = 16   # Swami Sivananda - used only to improve search recall

# ---- Embeddings ---------------------------------------------------------------
# "bge-small-en" : fast, ~130 MB, good for English queries (default)
# "bge-m3"       : ~2 GB, multilingual (Hindi/Sanskrit queries), needs more RAM
# "tfidf"        : offline fallback, no download, weaker semantics (good for testing)
EMBEDDER = os.getenv("GITA_EMBEDDER", "bge-small-en")
EMBEDDER_MODELS = {
    "bge-small-en": "BAAI/bge-small-en-v1.5",
    "bge-m3": "BAAI/bge-m3",
}
# bge-small works best when queries carry this prefix (documents get none)
QUERY_PREFIX = {"bge-small-en": "Represent this sentence for searching relevant passages: "}

# Fields concatenated to build the text that gets embedded / BM25-indexed
SEARCH_FIELDS = ["translation", "translation_alt", "transliteration", "word_meanings"]

# ---- Retrieval ----------------------------------------------------------------
TOP_K = int(os.getenv("GITA_TOP_K", "6"))
CANDIDATES = 20           # per retriever, before fusion
QUERY_EXPANSION = os.getenv("GITA_EXPAND", "1") != "0"   # glossary expansion; GITA_EXPAND=0 turns it off (A/B test)
RRF_K = 60                # reciprocal-rank-fusion constant
MAX_RELEVANT_VECTOR_DISTANCE = 0.45  # larger distances trigger the optional web fallback

# ---- Generation ---------------------------------------------------------------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("GITA_MODEL", "qwen2.5:7b")   # chosen by the evaluation run
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
LLM_TEMPERATURE = 0.2     # low = sticks to the sources
LLM_TIMEOUT_S = 300       # first call can be slow while Ollama loads the model
MAX_RETRIES = 1           # re-ask once if the citation check fails

# Shown when the question looks like a self-harm crisis (the LLM is NOT called).
# Add a verified local helpline here before sharing the app publicly.
CRISIS_RESOURCES = ""
