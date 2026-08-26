from retrieve import retrieve

TEST_QUERIES = [
    "how do I initialize a chat model in langchain",
    "what changed between langchain v0.1 and v1",
    "how do I use a vector store for retrieval",
    "what is the LLMChain class",
    "does langchain support streaming responses",
]

for q in TEST_QUERIES:
    print(f"QUERY: {q}")
    results = retrieve(q, top_k=5)
    scores = [r["rerank_score"] for r in results]
    spread = max(scores) - min(scores)
    print(f"rerank score spread: {spread:.4f} (max {max(scores):.4f}, min {min(scores):.4f})")
    for r in results:
        print(f"  rerank={r['rerank_score']:.3f} embed={r['embed_score']:.3f} {r['chunk']['source_path']}")
    print()