from langchain.chat_models import init_chat_model

from retrieve import retrieve

SYSTEM_PROMPT = """You are a documentation assistant for LangChain. Answer using only the provided context. If the answer isn't in the context, or the method being asked about doesn't exist or is deprecated, say so explicitly rather than guessing."""

DEFAULT_MODEL = "openai:gpt-4o-mini"

_model_cache = {}


def get_model(model_id=DEFAULT_MODEL):
    if model_id not in _model_cache:
        _model_cache[model_id] = init_chat_model(model_id)
    return _model_cache[model_id]


def build_context(results):
    blocks = []
    for r in results:
        c = r["chunk"]
        blocks.append(f"[{c['version']} | {c['source_path']}]\n{c['text']}")
    return "\n\n---\n\n".join(blocks)


def answer_question(query, model_id=DEFAULT_MODEL, top_k=5, version_filter=None):
    results = retrieve(query, top_k=top_k, version_filter=version_filter)
    context = build_context(results)

    model = get_model(model_id)
    messages = [
        ("system", SYSTEM_PROMPT),
        ("human", f"Context:\n{context}\n\nQuestion:\n{query}"),
    ]

    response = model.invoke(messages)
    usage = response.usage_metadata or {}

    return {
        "answer": response.content,
        "prompt_tokens": usage.get("input_tokens", 0),
        "completion_tokens": usage.get("output_tokens", 0),
        "sources": [r["chunk"]["source_path"] for r in results],
    }


if __name__ == "__main__":
    result = answer_question("how do I initialize a chat model in langchain")
    print(result["answer"])
    print()
    print("sources:", result["sources"])
    print("tokens:", result["prompt_tokens"], result["completion_tokens"])
    