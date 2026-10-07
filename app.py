import functools
import logging
import os
import streamlit as st
import sys
import uuid
from pathlib import Path
from streamlit.errors import StreamlitSecretNotFoundError

# Add project root to path so config and src imports work seamlessly
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    secrets = dict(st.secrets)
except StreamlitSecretNotFoundError:
    secrets = {}

# Cloud deployments do not include the local Chroma index. Use TF-IDF there
# unless the deployment explicitly selects another embedder.
if secrets.get("GITA_EMBEDDER"):
    os.environ.setdefault("GITA_EMBEDDER", secrets["GITA_EMBEDDER"])
elif not os.getenv("GITA_EMBEDDER"):
    chroma_dir = ROOT_DIR / "data" / "chroma"
    collection_exists = False
    if chroma_dir.exists():
        import chromadb

        client = chromadb.PersistentClient(path=str(chroma_dir))
        collection_names = {
            getattr(collection, "name", collection)
            for collection in client.list_collections()
        }
        collection_exists = "gita" in collection_names
        if collection_exists:
            metadata = client.get_collection("gita").metadata or {}
            if metadata.get("embedder"):
                os.environ["GITA_EMBEDDER"] = metadata["embedder"]
    if not collection_exists:
        os.environ["GITA_EMBEDDER"] = "tfidf"

from src.generate import LLMUnavailable, call_ollama
from src.generate import call_gemini
from src import config
from src.deployment import ensure_chroma_collection
from src.logging_config import configure_logging
from src.pipeline import answer_question
from src.request_context import bind_request_context, classify_client_source
from src.retrieve import get_retriever

configure_logging()

# Page Configuration
st.set_page_config(
    page_title="Gita Chatbot",
    page_icon="🕉️",
    layout="wide"
)

# Custom Styling
st.markdown("""
    <style>
        .stChatMessage { border-radius: 8px; }
        .sanskrit { font-family: 'Sanskrit', serif; font-size: 1.1em; color: #8e44ad; }
    </style>
""", unsafe_allow_html=True)


def render_verse_card(verse: dict):
    """Render structured verse card matching your dataset dictionary format."""
    st.markdown(f"**Chapter {verse.get('chapter')}, Verse {verse.get('verse')}** (`{verse.get('ref')}`)")
    st.markdown(f"<div class='sanskrit'>{verse.get('sanskrit', '')}</div>", unsafe_allow_html=True)
    st.caption(f"*Transliteration:* {verse.get('transliteration', '')}")
    st.write(f"**Translation:** {verse.get('translation', '')}")
    word_meanings = verse.get("word_meanings", "")
    if word_meanings:
        with st.expander("Word meanings"):
            st.markdown("\n".join(
                f"- {meaning.strip()}" for meaning in word_meanings.split(";") if meaning.strip()
            ))


def render_web_sources(sources: list[dict[str, str]]) -> None:
    if not sources:
        return
    with st.expander("Web sources"):
        for index, source in enumerate(sources, start=1):
            st.link_button(f"[{index}] {source['title']}", source["url"])


# --- Sidebar Navigation ---
st.sidebar.title("🕉️ Gita AI Assistant")
app_mode = st.sidebar.radio("Select View", ["Gita Chatbot", "Search Sholkas"])
provider_options = ["Gemini", "Ollama"]
default_provider = 0 if secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY") else 1
provider = st.sidebar.selectbox(
    "LLM provider",
    provider_options,
    index=default_provider,
)
if provider == "Gemini":
    model = st.sidebar.text_input("Gemini model", value=config.GEMINI_MODEL)
else:
    model = st.sidebar.text_input(
        "Ollama model",
        value=os.getenv("GITA_MODEL", "qwen2.5:7b"),
        help="Must already be pulled: ollama pull <name>",
    )
gemini_api_key = secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))

# Load cached retriever instance
@st.cache_resource(show_spinner="Preparing the verse search index...")
def load_retriever():
    if ensure_chroma_collection():
        get_retriever.cache_clear()
    return get_retriever()


try:
    retriever_inst = load_retriever()
except Exception as e:
    st.error(
        "Failed to initialize the verse search index. The app will build the ChromaDB "
        f"collection from the verse dataset when available. Details: {e}"
    )
    st.stop()

# ==========================================
# VIEW 1: AI CHATBOT (RAG + Guardrails)
# ==========================================
if app_mode == "Gita Chatbot":
    st.title("Gita Chatbot")
    st.caption("Ask questions about life, duty, yoga, and philosophy grounded strictly in Gita verses.")

    # Initialize Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "client_id" not in st.session_state:
        st.session_state.client_id = uuid.uuid4().hex[:12]

    # Display past chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander("📖 Cited Verses", expanded=False):
                    for verse in msg["sources"]:
                        render_verse_card(verse)
            render_web_sources(msg.get("web_sources", []))

    # Chat Input Box
    if user_query := st.chat_input("e.g., What does Sri Krishna say about controlling the mind?"):
        request_id = uuid.uuid4().hex[:12]
        request_source = classify_client_source(
            st.context.ip_address,
            st.context.headers,
        )
        with bind_request_context(request_source, st.session_state.client_id, request_id):
            logging.getLogger("src.request").info("Question received.")

        # 1. Render User Message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # 2. RAG Process
        with st.chat_message("assistant"):
            with st.spinner("Retrieving verses & generating answer..."):
                try:
                    llm = (
                        functools.partial(
                            call_gemini,
                            api_key=gemini_api_key,
                            model=model,
                        )
                        if provider == "Gemini"
                        else functools.partial(call_ollama, model=model)
                    )
                    with bind_request_context(
                        request_source,
                        st.session_state.client_id,
                        request_id,
                    ):
                        result = answer_question(user_query, llm=llm)
                except LLMUnavailable as err:
                    st.error(f"LLM Error: {err}")
                    st.stop()

            # Render output
            st.markdown(result.text)
            if result.warning:
                st.warning(result.warning)

            render_web_sources(result.web_sources)

            if result.cited_verses:
                with st.expander("📖 Cited Verses (Sources)", expanded=True):
                    for v in result.cited_verses:
                        render_verse_card(v)

        # 3. Append Assistant Message
        st.session_state.messages.append({
            "role": "assistant",
            "content": result.text,
            "sources": result.cited_verses,
            "web_sources": result.web_sources,
        })

# ==========================================
# VIEW 2: VERSE BROWSER
# ==========================================
elif app_mode == "Search Sholkas":
    st.title("📖 Search Sholkas")
    st.write("Select any Chapter and Verse number to read directly.")

    all_verses = retriever_inst.verses  # Accesses internal parsed JSON list

    col1, col2 = st.columns(2)
    with col1:
        chapter_num = st.selectbox("Select Chapter", options=list(range(1, 19)))

    # Filter verses matching selected chapter
    chapter_verses = [v for v in all_verses if v.get("chapter") == chapter_num]
    verse_numbers = [v.get("verse") for v in chapter_verses]

    with col2:
        selected_verse_num = st.selectbox("Select Verse", options=verse_numbers)

    # Retrieve selected verse object
    matching_verse = next(
        (v for v in chapter_verses if v.get("verse") == selected_verse_num), None
    )

    if matching_verse:
        st.divider()
        st.subheader(f"Chapter {chapter_num}, Verse {selected_verse_num} (`{matching_verse.get('ref')}`)")
        st.markdown(f"### Sanskrit\n`{matching_verse.get('sanskrit')}`")
        st.markdown(f"### Transliteration\n*{matching_verse.get('transliteration')}*")
        st.markdown(f"### Translation\n{matching_verse.get('translation')}")
        word_meanings = matching_verse.get("word_meanings", "")
        if word_meanings:
            st.markdown("### Word meanings")
            st.markdown("\n".join(
                f"- {meaning.strip()}" for meaning in word_meanings.split(";")
                if meaning.strip()
            ))
    else:
        st.warning("Verse not found in dataset.")
