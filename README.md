# NEXORA Knowledge Studio

NEXORA is a Streamlit document research app. Upload PDF, DOCX, CSV, JSON, TXT, or Markdown files, then ask questions with source passages attached to each answer.

## Run locally

```bash
pip install -r requirements.txt
streamlit run rag.py
```

Enter your own OpenAI API key in the app. NEXORA uses it for embedding and chat requests during that Streamlit session; it does not read a shared `OPENAI_API_KEY` environment secret or save the key in chat history or uploaded files.

## Streamlit Community Cloud

- Use `rag.py` as the app entrypoint.
- Keep `requirements.txt` at the repository root; it lists the app's Python dependencies and pins Streamlit for a repeatable UI runtime.
- `.streamlit/config.toml` sets the dark forest theme and green accent used by the app.
- Select the Python version in the app's Community Cloud advanced settings. The requirements file does not select Python itself.
- Keep API keys out of Git. Each visitor enters their own key on the connection screen.

## Project files

- `rag.py` — Streamlit interface, session state, and chat flow.
- `rag_core.py` — document readers, multi-file indexing, citations, and recent chat history formatting.
- `tests/test_rag_core.py` — offline checks for supported file handling and core RAG behavior.

