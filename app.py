import functools
import logging
import os
import sys
import uuid
from pathlib import Path
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

# Add project root to path so config and src imports work seamlessly
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    secrets = dict(st.secrets)
except StreamlitSecretNotFoundError:
    secrets = {}

from src import config

# Use the embedder recorded in the packaged Chroma index unless explicitly overridden.
configured_embedder = os.getenv("GITA_EMBEDDER") or secrets.get("GITA_EMBEDDER")
if configured_embedder:
    config.EMBEDDER = configured_embedder
else:
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
                config.EMBEDDER = metadata["embedder"]
    if not collection_exists:
        config.EMBEDDER = "tfidf"

from src.deployment import ensure_chroma_collection
from src.generate import LLMUnavailable, call_gemini, call_groq_then_gemini, call_ollama
from src.logging_config import configure_logging
from src.pipeline import answer_question
from src.request_context import bind_request_context, classify_client_source
from src.retrieve import get_retriever

configure_logging()

# Page Configuration
st.set_page_config(
    page_title="Gita Chatbot",
    page_icon="🕉️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom Styling
st.markdown(
    """
    <style>
        /* 1. Hide Streamlit native header bar & footer */
        header[data-testid="stHeader"], footer {
            display: none !important;
        }

        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 1rem !important;
        }

        div[data-testid="stLayoutWrapper"]:has(> .stVerticalBlock.st-key-chat_history) {
            height: max(100px, min(560px, calc(100vh - 400px))) !important;
            max-height: max(100px, calc(100vh - 400px)) !important;
        }

        /* 2. Left-side Krishna image */
        div[data-testid="stImage"] img {
            width: 100%;
            max-height: calc(100vh - 180px);
            object-fit: contain;
            border-radius: 16px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5);
            opacity: 0.88;
        }

        /* 3. Legible chat messages with slight translucent dark backdrop */
        .stChatMessage {
            border-radius: 10px;
            background-color: rgba(15, 23, 42, 0.75) !important;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .sanskrit {
            font-family: 'Sanskrit', serif;
            font-size: 1.25em;
            color: #f1c40f;
        }

        .om-mark {
            color: #a64b1a;
        }

        /* 4. Navigation buttons use the dark saffron palette. */
        .st-key-chatbot_nav button,
        .st-key-search_nav button {
            width: 100% !important;
            height: 60px !important;
            font-size: 1.35rem !important;
            font-weight: 900 !important;
            letter-spacing: 0.5px !important;
            border-radius: 12px !important;
            border: 2px solid #70401f !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.4) !important;
        }

        .st-key-chatbot_nav button[data-testid="stBaseButton-primary"],
        .st-key-search_nav button[data-testid="stBaseButton-primary"] {
            background-color: #33251b !important;
            color: #ffffff !important;
            border-color: #a85d27 !important;
        }

        .st-key-chatbot_nav button:hover,
        .st-key-search_nav button:hover {
            border-color: #c27636 !important;
        }

        div[data-testid="stPopover"] > button {
            min-height: 2rem !important;
            padding: 0.3rem 0.65rem !important;
            font-size: 0.9rem !important;
        }

        /* Compact spacing */
        hr {
            margin-top: 8px !important;
            margin-bottom: 12px !important;
        }

        .stCaption {
            margin-bottom: 4px !important;
        }

        div[data-testid="stVerticalBlock"] > div {
            gap: 0.5rem !important;
        }
    </style>
""",
    unsafe_allow_html=True,
)


def render_verse_card(verse: dict):
    """Render structured verse card matching dataset dictionary format."""
    st.markdown(
        f"**Chapter {verse.get('chapter')}, Verse {verse.get('verse')}** (`{verse.get('ref')}`)"
    )
    st.markdown(
        f"<div class='sanskrit'>{verse.get('sanskrit', '')}</div>",
        unsafe_allow_html=True,
    )
    st.caption(f"*Transliteration:* {verse.get('transliteration', '')}")
    st.write(f"**Translation:** {verse.get('translation', '')}")
    word_meanings = verse.get("word_meanings", "")
    if word_meanings:
        with st.expander("Word meanings"):
            st.markdown(
                "\n".join(
                    f"- {meaning.strip()}"
                    for meaning in word_meanings.split(";")
                    if meaning.strip()
                )
            )


def render_web_sources(sources: list[dict[str, str]]) -> None:
    if not sources:
        return
    with st.expander("Web sources"):
        for index, source in enumerate(sources, start=1):
            st.link_button(f"[{index}] {source['title']}", source["url"])


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
        "Failed to initialize the packaged verse search index. Include the matching "
        "`data/chroma` index and `data/tfidf.pkl` model in the deployment. "
        f"Details: {e}"
    )
    st.stop()

# Page heading and view navigation sit above the image and page content.
st.markdown(
    '## <span class="om-mark">ॐ</span> Bhagavad Gita AI',
    unsafe_allow_html=True,
)
st.space("small")

if "active_view" not in st.session_state:
    st.session_state.active_view = "Gita Chatbot"

# =========================================================
# MAIN LAYOUT: LEFT SIDE IMAGE | RIGHT SIDE PAGE CONTENT
# =========================================================
left_col, right_col = st.columns([1, 4.5], gap="small")

with right_col:
    col1, col2 = st.columns([1, 1], gap="small")

    with col1:
        btn_type_1 = "primary" if st.session_state.active_view == "Gita Chatbot" else "secondary"
        if st.button(
            "💬 Gita Chatbot",
            type=btn_type_1,
            width="stretch",
            key="chatbot_nav",
        ):
            st.session_state.active_view = "Gita Chatbot"
            st.rerun()

    with col2:
        btn_type_2 = "primary" if st.session_state.active_view == "Search Shlokas" else "secondary"
        if st.button(
            "📖 Search Shlokas",
            type=btn_type_2,
            width="stretch",
            key="search_nav",
        ):
            st.session_state.active_view = "Search Shlokas"
            st.rerun()

    st.divider()

# --- LEFT COLUMN: VISHWAROOPA IMAGE ---
with left_col:
    st.image(
        str(ROOT_DIR / "assets" / "krishna.png"),
        width="stretch",
        alt="Krishna standing beneath a radiant celestial sky",
    )
    st.space("small")
    provider = "Groq"
    model = config.GROQ_MODEL
    with st.container(horizontal=True, gap="small", vertical_alignment="center"):
        with st.popover("⚙️ Models", width="content"):
            provider = st.selectbox("LLM Provider", ["Groq", "Gemini", "Llama"])
            if provider == "Gemini":
                model = st.text_input(
                    "Gemini Model",
                    value=config.GEMINI_MODEL,
                    disabled=True,
                )
            elif provider == "Groq":
                model = st.text_input(
                    "Groq Model",
                    value=config.GROQ_MODEL,
                    disabled=True,
                )
            else:
                model = st.text_input(
                    "Llama Model",
                    value=os.getenv("GITA_MODEL", "llama3.2:3b"),
                    help="Must already be pulled: ollama pull <name>",
                    disabled=True,
                )
        if st.session_state.active_view == "Gita Chatbot" and st.button(
            "Clear chat",
            icon=":material/delete_outline:",
            type="secondary",
            width="content",
        ):
            st.session_state.messages = []
            st.rerun()

# --- RIGHT COLUMN: APP CONTENT & CHATBOT ---
with right_col:
    gemini_api_key = secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
    groq_api_key = secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))

    # ==========================================
    # VIEW 1: AI CHATBOT (RAG + Guardrails)
    # ==========================================
    if st.session_state.active_view == "Gita Chatbot":
        st.caption(
            "Ask questions about life, duty, yoga, and philosophy grounded strictly in Gita verses."
        )

        # Initialize Chat History
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "client_id" not in st.session_state:
            st.session_state.client_id = uuid.uuid4().hex[:12]

        has_conversation = bool(st.session_state.messages)
        if has_conversation:
            with st.bottom:
                _, composer_col = st.columns([1, 4.5], gap="small")
                with composer_col:
                    user_query = st.chat_input(
                        "e.g., What does Sri Krishna say about controlling the mind?"
                    )
        else:
            user_query = st.chat_input(
                "e.g., What does Sri Krishna say about controlling the mind?"
            )

        pending_query = st.session_state.pop("pending_question", None)
        question_to_answer = pending_query or user_query
        request_id = None
        request_source = None
        if user_query:
            st.session_state.messages.append({"role": "user", "content": user_query})
            if not has_conversation:
                st.session_state.pending_question = user_query
                st.rerun()

        if question_to_answer:
            request_id = uuid.uuid4().hex[:12]
            request_source = classify_client_source(
                st.context.ip_address,
                st.context.headers,
            )
            with bind_request_context(
                request_source, st.session_state.client_id, request_id
            ):
                logging.getLogger("src.request").info("Question received.")

        with st.container(
            height=560,
            border=False,
            key="chat_history",
            autoscroll=True,
        ):
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if "sources" in msg and msg["sources"]:
                        with st.expander("📜 Cited Verses", expanded=False):
                            for verse in msg["sources"]:
                                render_verse_card(verse)
                    if msg.get("warning"):
                        st.warning(msg["warning"])
                    render_web_sources(msg.get("web_sources", []))
            response_slot = st.empty()

        if question_to_answer:
            assert request_id is not None and request_source is not None
            with response_slot.container():
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
                                else functools.partial(
                                    call_groq_then_gemini,
                                    groq_api_key=groq_api_key,
                                    gemini_api_key=gemini_api_key,
                                    groq_model=model,
                                )
                                if provider == "Groq"
                                else functools.partial(call_ollama, model=model)
                            )
                            with bind_request_context(
                                request_source,
                                st.session_state.client_id,
                                request_id,
                            ):
                                result = answer_question(question_to_answer, llm=llm)
                        except LLMUnavailable as err:
                            st.error(f"LLM Error: {err}")
                            st.stop()

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result.text,
                    "sources": result.cited_verses,
                    "web_sources": result.web_sources,
                    "warning": result.warning,
                }
            )
            st.rerun()

    # ==========================================
    # VIEW 2: VERSE BROWSER
    # ==========================================
    elif st.session_state.active_view == "Search Shlokas":
        st.subheader("📖 Search Shlokas")
        st.write("Select any Chapter and Verse number to read directly.")

        all_verses = retriever_inst.verses

        c1, c2 = st.columns(2)
        with c1:
            chapter_num = st.selectbox("Select Chapter", options=list(range(1, 19)))

        # Filter verses matching selected chapter
        chapter_verses = [v for v in all_verses if v.get("chapter") == chapter_num]
        verse_numbers = [v.get("verse") for v in chapter_verses]

        with c2:
            selected_verse_num = st.selectbox("Select Verse", options=verse_numbers)

        # Retrieve selected verse object
        matching_verse = next(
            (v for v in chapter_verses if v.get("verse") == selected_verse_num),
            None,
        )

        if matching_verse:
            st.divider()
            st.subheader(
                f"Chapter {chapter_num}, Verse {selected_verse_num} (`{matching_verse.get('ref')}`)"
            )
            st.markdown(f"### Sanskrit\n`{matching_verse.get('sanskrit')}`")
            st.markdown(
                f"### Transliteration\n*{matching_verse.get('transliteration')}*"
            )
            st.markdown(f"### Translation\n{matching_verse.get('translation')}")
            word_meanings = matching_verse.get("word_meanings", "")
            if word_meanings:
                st.markdown("### Word meanings")
                st.markdown(
                    "\n".join(
                        f"- {meaning.strip()}"
                        for meaning in word_meanings.split(";")
                        if meaning.strip()
                    )
                )
        else:
            st.warning("Verse not found in dataset.")