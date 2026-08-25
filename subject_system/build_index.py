import json
import pickle
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = Path(__file__).parent / "chunks.jsonl"
INDEX_PATH = Path(__file__).parent / "vector_index.pkl"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
BATCH_SIZE = 16


def main():
    chunks = [json.loads(l) for l in open(CHUNKS_PATH)]
    model = SentenceTransformer(EMBED_MODEL)
    texts = [c["text"] for c in chunks]

    vectors = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True, normalize_embeddings=True)
    matrix = np.array(vectors, dtype=np.float32)

    index = {
        "vectors": matrix,
        "chunks": chunks,
        "model": EMBED_MODEL,
    }
    with open(INDEX_PATH, "wb") as f:
        pickle.dump(index, f)

    print(f"{len(chunks)} vectors -> {INDEX_PATH}")


if __name__ == "__main__":
    main()