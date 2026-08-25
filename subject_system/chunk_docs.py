import json
import re
import uuid
from pathlib import Path

IN_PATH = Path(__file__).parent.parent / "corpus_build" / "filtered_docs.jsonl"
OUT_PATH = Path(__file__).parent / "chunks.jsonl"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def strip_frontmatter(text):
    if text.startswith("---"):
        end = text.find("---", 3)
        if end != -1:
            return text[end + 3:].strip()
    return text


def chunk_text(text, size, overlap):
    words = text.split()
    if len(words) <= size:
        return [text]
    chunks = []
    start = 0
    while start < len(words):
        end = start + size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - overlap
    return chunks


def main():
    docs = [json.loads(l) for l in open(IN_PATH)]
    out = []
    for doc in docs:
        text = strip_frontmatter(doc["text"])
        text = re.sub(r"\n{3,}", "\n\n", text)
        pieces = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        for i, piece in enumerate(pieces):
            out.append({
                "chunk_id": str(uuid.uuid4())[:8],
                "version": doc["version"],
                "source_path": doc["path"],
                "chunk_index": i,
                "text": piece,
            })

    with open(OUT_PATH, "w") as f:
        for c in out:
            f.write(json.dumps(c) + "\n")

    by_version = {}
    for c in out:
        by_version[c["version"]] = by_version.get(c["version"], 0) + 1
    print(by_version)
    print(f"total {len(out)} chunks -> {OUT_PATH}")


if __name__ == "__main__":
    main()