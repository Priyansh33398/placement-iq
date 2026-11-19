"""Retrieval and grounded answering over the local FAISS index."""
import json
import os
import pathlib

import faiss
import numpy as np
from fastembed import TextEmbedding

ROOT = pathlib.Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT / "data" / "index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-2.5-flash"

PROMPT = """You are a placement-preparation assistant for a college student.
Answer the question using ONLY the numbered context passages below.
Cite the passages you used like [1] or [2] after each claim.
If the context does not contain the answer, reply exactly:
"I don't know from the provided documents."

Context:
{context}

Question: {question}

Answer:"""


class Retriever:
    def __init__(self) -> None:
        self.index = faiss.read_index(str(INDEX_DIR / "chunks.faiss"))
        self.records = json.loads((INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))
        self.model = TextEmbedding(MODEL_NAME)

    def search(self, question: str, top_k: int = 4) -> list[dict]:
        vector = np.array(list(self.model.embed([question])), dtype="float32")
        faiss.normalize_L2(vector)
        scores, ids = self.index.search(vector, top_k)
        return [
            {**self.records[i], "score": round(float(s), 4)}
            for s, i in zip(scores[0], ids[0])
            if i != -1
        ]


def answer(question: str, retriever: Retriever, top_k: int = 4) -> dict:
    hits = retriever.search(question, top_k)
    context = "\n\n".join(f"[{n}] ({h['source']}) {h['text']}" for n, h in enumerate(hits, 1))

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return {
            "question": question,
            "answer": None,
            "note": "GEMINI_API_KEY not set - returning retrieved passages only.",
            "sources": hits,
        }

    from google import genai

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=PROMPT.format(context=context, question=question),
    )
    return {"question": question, "answer": response.text, "sources": hits}
