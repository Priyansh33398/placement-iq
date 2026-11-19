"""Chunk the documents in data/docs, embed them, and build the FAISS index."""
import json
import pathlib
import sys

import faiss
import numpy as np
from fastembed import TextEmbedding

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "data" / "docs"
INDEX_DIR = ROOT / "data" / "index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_CHARS = 800
OVERLAP_CHARS = 150


def read_document(path: pathlib.Path) -> str:
    if path.suffix.lower() == ".pdf":
        import pymupdf

        with pymupdf.open(path) as doc:
            return "\n".join(page.get_text() for page in doc)
    return path.read_text(encoding="utf-8", errors="replace")


def chunk_text(text: str) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= CHUNK_CHARS:
            current = f"{current}\n\n{para}".strip()
            continue
        if current:
            chunks.append(current)
        while len(para) > CHUNK_CHARS:
            chunks.append(para[:CHUNK_CHARS])
            para = para[CHUNK_CHARS - OVERLAP_CHARS :]
        current = para
    if current:
        chunks.append(current)
    return chunks


def main() -> None:
    files = sorted(
        p for p in DOCS_DIR.iterdir() if p.suffix.lower() in (".pdf", ".txt", ".md")
    )
    if not files:
        sys.exit(f"No documents found in {DOCS_DIR}. Add .pdf/.md/.txt files first.")

    records = []
    for path in files:
        for chunk in chunk_text(read_document(path)):
            records.append({"id": len(records), "source": path.name, "text": chunk})
    print(f"{len(files)} documents -> {len(records)} chunks")

    model = TextEmbedding(MODEL_NAME)
    vectors = np.array(list(model.embed([r["text"] for r in records])), dtype="float32")
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_DIR / "chunks.faiss"))
    (INDEX_DIR / "chunks.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"Index written to {INDEX_DIR}")


if __name__ == "__main__":
    main()
