"""Retrieval hit-rate over the hand-built question set in eval/questions.json.

A question counts as a hit at k when a top-k chunk comes from the expected
source document AND contains the expected keyword.
"""
import json
import pathlib

from rag import Retriever

ROOT = pathlib.Path(__file__).resolve().parents[1]
QUESTIONS = json.loads((ROOT / "eval" / "questions.json").read_text(encoding="utf-8"))


def hit(chunks: list[dict], source: str, keyword: str) -> bool:
    return any(
        c["source"] == source and keyword.lower() in c["text"].lower() for c in chunks
    )


def main() -> None:
    retriever = Retriever()
    for k in (1, 2, 4):
        hits = 0
        misses = []
        for q in QUESTIONS:
            if hit(retriever.search(q["question"], k), q["source"], q["keyword"]):
                hits += 1
            else:
                misses.append(q["question"])
        print(f"hit-rate@{k}: {hits}/{len(QUESTIONS)} = {hits / len(QUESTIONS):.0%}")
        if k == 4 and misses:
            print("misses@4:")
            for m in misses:
                print("  -", m)


if __name__ == "__main__":
    main()
