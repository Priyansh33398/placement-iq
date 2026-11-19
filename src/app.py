"""FastAPI app: POST /ask with a question, get a grounded, cited answer."""
import pathlib

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

load_dotenv(pathlib.Path(__file__).resolve().parents[1] / ".env")

from rag import Retriever, answer  # noqa: E402

app = FastAPI(title="Placement Prep RAG Assistant")
retriever = Retriever()


class AskRequest(BaseModel):
    question: str
    top_k: int = 4


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "chunks": len(retriever.records)}


@app.post("/ask")
def ask(request: AskRequest) -> dict:
    return answer(request.question, retriever, request.top_k)
