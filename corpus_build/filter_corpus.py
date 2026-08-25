import json
from pathlib import Path

IN_PATH = Path(__file__).parent / "raw_docs.jsonl"
OUT_PATH = Path(__file__).parent / "filtered_docs.jsonl"

V1_KEEP_PREFIXES = ("oss/langchain/", "oss/python/")
V1_DROP_PREFIXES = ("oss/python/integrations/",)


def keep(doc):
    if doc["version"] in ("v0.1", "v0.3"):
        return True
    if doc["version"] == "v1":
        path = doc["path"]
        if not path.startswith(V1_KEEP_PREFIXES):
            return False
        if path.startswith(V1_DROP_PREFIXES) and "chat" not in path and "vectorstores" not in path:
            return False
        return True
    return False


def main():
    docs = [json.loads(l) for l in open(IN_PATH)]
    filtered = [d for d in docs if keep(d)]
    with open(OUT_PATH, "w") as f:
        for d in filtered:
            f.write(json.dumps(d) + "\n")
    by_version = {}
    for d in filtered:
        by_version[d["version"]] = by_version.get(d["version"], 0) + 1
    print(by_version)
    print(f"total {len(filtered)} -> {OUT_PATH}")


if __name__ == "__main__":
    main()