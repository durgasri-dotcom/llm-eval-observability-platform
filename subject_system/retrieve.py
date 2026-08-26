import pickle
from pathlib import Path

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

INDEX_PATH = Path(__file__).parent / "vector_index.pkl"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RERANK_MODEL = "BAAI/bge-reranker-base"
CANDIDATE_K = 20

_model = None
_index = None
_reranker = None


def load():
    global _model, _index, _reranker
    if _index is None:
        with open(INDEX_PATH, "rb") as f:
            _index = pickle.load(f)
        _model = SentenceTransformer(_index["model"])
        _reranker = CrossEncoder(RERANK_MODEL)
    return _model, _index, _reranker


def retrieve(query, top_k=5, version_filter=None, candidate_k=CANDIDATE_K):
    model, index, reranker = load()
    q_vec = model.encode([QUERY_PREFIX + query], normalize_embeddings=True)[0]

    vectors = index["vectors"]
    chunks = index["chunks"]

    scores = vectors @ q_vec

    if version_filter:
        mask = np.array([c["version"] == version_filter for c in chunks])
        scores = np.where(mask, scores, -np.inf)

    candidate_idx = np.argsort(scores)[::-1][:candidate_k]
    candidate_idx = [i for i in candidate_idx if scores[i] > -np.inf]

    pairs = [(query, chunks[i]["text"]) for i in candidate_idx]
    rerank_scores = reranker.predict(pairs)

    ranked = sorted(zip(candidate_idx, rerank_scores), key=lambda x: x[1], reverse=True)

    seen_sources = set()
    top = []
    for i, r in ranked:
        source = chunks[i]["source_path"]
        if source in seen_sources:
            continue
        seen_sources.add(source)
        top.append((i, r))
        if len(top) >= top_k:
            break

    return [
        {"chunk": chunks[i], "embed_score": float(scores[i]), "rerank_score": float(r)}
        for i, r in top
    ]


if __name__ == "__main__":
    results = retrieve("how do I initialize a chat model in langchain", top_k=5)
    for r in results:
        print(f"rerank={r['rerank_score']:.3f} embed={r['embed_score']:.3f} {r['chunk']['version']} {r['chunk']['source_path']}")
        print(r["chunk"]["text"][:150])
        print("---")