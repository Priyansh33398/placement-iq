# Placement Prep RAG Assistant

A retrieval-augmented chatbot that answers questions over placement documents
(policies, job descriptions, prep notes). I built it during placement season to
stop re-reading the same PDFs before every drive: drop documents into
`data/docs/`, build the index once, and ask questions in plain English. Answers
come only from the documents and carry source citations, so I can trust them.

## How it works

1. `src/ingest.py` reads every `.pdf`, `.md`, and `.txt` in `data/docs/`,
   splits them into overlapping ~800-character chunks, embeds each chunk
   locally with the MiniLM sentence-transformer
   (`sentence-transformers/all-MiniLM-L6-v2` via fastembed), and stores the
   vectors in a FAISS inner-product index.
2. `src/rag.py` embeds the question, retrieves the top-k chunks from FAISS,
   and asks Gemini (`gemini-2.5-flash`) to answer **only from those chunks**,
   citing passages like `[1]`. When the context doesn't contain the answer,
   the assistant says so instead of guessing.
3. `src/app.py` exposes this as a FastAPI service: `POST /ask` with
   `{"question": "...", "top_k": 4}`.
4. `src/eval_retrieval.py` measures retrieval quality against a hand-built
   30-question set (`eval/questions.json`): a question is a hit when a top-k
   chunk comes from the right document and contains the expected fact. I used
   this to tune chunk size and top-k.

Embeddings and retrieval run fully offline; only answer generation calls the
Gemini API.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # put your GEMINI_API_KEY inside (aistudio.google.com/apikey)
python src/ingest.py
python src/eval_retrieval.py
uvicorn app:app --app-dir src --reload
```

Then ask something:

```bash
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d "{\"question\": \"What is the last date to register on the placement portal?\"}"
```

Without `GEMINI_API_KEY`, `/ask` still returns the retrieved passages, so
retrieval and the eval work with no API key at all.

## Results

On the bundled 30-question set, retrieval reaches its best hit-rate at
`top_k=4` with ~800-character chunks and 150-character overlap (see
`src/eval_retrieval.py` output). Grounded prompting with citations keeps the
model from answering outside the documents.

## Notes

- The three documents in `data/docs/` are sample stand-ins so the repo runs
  out of the box; replace them with your own placement PDFs and re-run
  `src/ingest.py`.
- The index lives in `data/index/` and is rebuilt from scratch on each ingest.
