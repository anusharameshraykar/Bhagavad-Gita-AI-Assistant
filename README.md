# gita-rag — Bhagavad Gita Q&A with RAG (free, local-first)

Current status: setup, data, retrieval, generation + citation checks, evaluation + tuning, and the UI are in place.

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements-local.txt
```
(Local Ollama is optional if you choose Gemini in the app.)

## Run
```bash
python -m src.build_data        # download + clean 701 verses -> data/gita_verses.json
python -m src.ingest            # embed verses -> data/chroma  (first run downloads ~130 MB model)
python -m src.retrieve "What does the Gita say about fear of failure?"
python -m eval.retrieval_check  # 12 questions with expected verses -> hit rate
python -m pytest -q tests       # unit tests (reference parser, tokenizer)
```
Offline / no-download mode (weaker, for testing only): `GITA_EMBEDDER=tfidf python -m src.ingest`
Multilingual mode (needs ~2 GB + more RAM): `GITA_EMBEDDER=bge-m3`. Re-run ingest after changing it.

## Run the web app
```bash
pip install -r requirements-local.txt
streamlit run app.py                     # opens http://localhost:8501
GITA_MOCK_LLM=1 streamlit run app.py     # look around without Ollama (placeholder answers)
```
- **Chat page:** ask a question; cited verses appear as tappable cards (Sanskrit, transliteration, translation).
  Crisis messages, medical/financial decisions, and religion/politics questions get fixed replies with no verse cards.
- **Search Sholkas** (sidebar): read any verse directly; no AI involved.
- Sidebar: choose Gemini or Ollama, the model, number of verses retrieved, and debug info.
- Open on your phone (same Wi-Fi): use the "Network URL" Streamlit prints.
- Hybrid retrieval does not trigger an LLM relevance check when it already has usable verses within
  the relevance cutoff. Only empty or weak hybrid retrieval leads to an LLM check: if the question is
  Gita-related, the app does a Tavily web search; if it is not, it returns a brief in-scope response.
  Exact verse lookups skip this check. Unclear classifications fail closed. Set `TAVILY_API_KEY`
  before starting Streamlit; only Gita-related questions with weak or empty retrieval are sent to Tavily.
- Terminal logs label chat requests as `internal`, `external`, or `unknown`, with an opaque
  per-session client ID and per-question request ID. Raw IP addresses and question text are not
  logged; IDs identify a browser session, not a person's real identity.

## Deploy publicly (Streamlit Community Cloud)

The root `app.py` is the deployment entrypoint. The deployment uses Gemini for answers and the
lightweight TF-IDF embedder to avoid downloading a neural model during cloud startup. A fresh
deployment builds its local Chroma index from the checked-in `data/gita_verses.json` on first start.

1. Push this project to a GitHub repository you control. Do not commit `.env`, API keys,
   `.streamlit/secrets.toml`, `.venv`, or `data/chroma`.
2. In Streamlit Community Cloud, create an app from that repository and set the main file to
   `app.py`. It installs the lean runtime dependencies from `requirements.txt`.
3. Add this in the app's **Secrets** settings, replacing the placeholder with a key from Google AI
   Studio:

   ```toml
   GEMINI_API_KEY = "your-gemini-api-key"
   ```

   Optional web-search fallback (Tavily):

   ```toml
   TAVILY_API_KEY = "your-tavily-api-key"
   ```

   The LLM provider defaults to Gemini when `GEMINI_API_KEY` is present. Without Tavily, a selected
   web-search action returns a clear unavailable message; verse answering continues to work.
4. Deploy, then test a direct verse lookup, a normal question, and the app's safety responses.
   Gemini usage is billed/limited according to your Google AI Studio account; review its quotas
   before sharing the public URL widely.

For local use of the neural embedding model, install `requirements-local.txt`; cloud installs should
use only `requirements.txt`.

## Ask questions — needs Ollama running
```bash
ollama pull qwen2.5:7b                       # default model
python -m src.ask "What is karma yoga?"
python -m src.ask "How to control anger?" --debug          # shows retrieved verses + validation
python -m src.ask "Explain BG 2.47" --model llama3.2:3b    # try another model
```
The answering flow is orchestrated by **LangGraph** as a bounded tool-using workflow: safety checks
(crisis; personal medical/financial decisions -> fixed referral; neutral declines) -> hybrid retrieval.
If hybrid retrieval finds verses, they are passed directly to the answer model without an additional
relevance-check call. If it finds none, the model may select exactly one allowed action: rewrite and
retry verse search, search the web for a clearly Gita-related question, or decline an out-of-scope
question. Tool implementations and safety checks remain deterministic; citations are validated and
the answer can be retried once. If the model still refuses or produces no verifiable citation, relevant
retrieved verses are shown as possible context rather than discarded. Self-harm messages get a fixed supportive reply and never reach the model
(`src/safety.py`; add a verified local helpline in `config.CRISIS_RESOURCES`). This is framework-based
RAG with a bounded agent: the model can select a limited next action, while tool implementations,
safety checks, and citation validation remain controlled by the application.

## Evaluate models
```bash
ollama pull qwen2.5:7b
python -m eval.run_eval --models llama3.2:3b qwen2.5:7b
python -m eval.run_eval --models llama3.2:3b --limit 5     # quick smoke test
python -m eval.run_eval --mock                             # harness check, no Ollama
```
35 questions (24 concept, 4 direct-reference, 7 adversarial). Writes `eval/results/<time>.md` (comparison table +
adversarial answers to read yourself) and `.json` (every answer). Re-run after any change to the prompt,
`TOP_K`, chunking or embedder, and compare. Edit `eval/questions.json` to add your own questions
(check expected verses against the actual text first; a test verifies they exist).

## Diagnose a retrieval miss
```bash
python -m eval.rank_check c05 c07      # rank of each expected verse in vector / keyword / hybrid search
```
Rank ~8-10 => raising `TOP_K` helps. Rank 50+ => retrieval itself needs work (query expansion, better embedder, reranker).

## Query expansion (src/glossary.py)
Bridges Sanskrit terms and vocabulary gaps ("gunas" -> "qualities of nature, purity, passion, ignorance";
"take birth" -> "I manifest myself / reborn"). Vector search gets every matching entry; BM25 only gets entries
flagged `keyword=True`, because for terms like yajna/atman the Sanskrit word already matches and extra words made
ranks worse (measured). `GITA_EXPAND=0` switches it off, so you can A/B on your real embedder:
```bash
GITA_EXPAND=0 python -m eval.run_eval --models qwen2.5:7b     # without
python -m eval.run_eval --models qwen2.5:7b                    # with (default)
```
Embedder experiment: `bge-m3` (2 GB) did NOT fix the vocabulary gaps (c07 got worse), so the default stays
`bge-small-en`. If your index was built with bge-m3: `unset GITA_EMBEDDER && python -m src.ingest`.

## How retrieval works
1. **Direct lookup** — "2.47", "chapter 2 verse 47", "verse 47 of chapter 2" fetch that verse, no search.
2. **Hybrid search** — vector similarity (meaning) + BM25 (exact Sanskrit terms, diacritics
   normalised, hyphenated compounds split *and* joined), merged with Reciprocal Rank Fusion.

## Known limitations
- A bare number like "python 3.12" is read as verse 3.12 (it is a valid verse). Acceptable for a POC.
- Devanagari queries are not matched by BM25 (only romanised text is indexed). Use `bge-m3` + add Sanskrit to
  `SEARCH_FIELDS` if you need that.
- Only English + transliteration is searched; Hindi translations are not used.

## Licensing — read before going public
The repo `gita/gita` is public domain, but the *texts inside it* have their own rights.
- Displayed translation: "Shri Purohit Swami" (author id 21). The 1935 original is old, but this copy is
  visibly modernised/edited (e.g. "Lord Shri Krishna replied"), so confirm its terms yourself.
- Search-only helper text: Swami Sivananda translation + word meanings (never shown to users).
- Fine for a private learning project. **Before a public demo, verify each source's license** or swap in a
  clearly public-domain translation (e.g. Edwin Arnold / Besant) by changing the ids in `src/config.py`.
