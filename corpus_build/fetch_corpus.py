import json
import subprocess
from pathlib import Path

VERSIONS = [
    {"tag": "langchain==0.1.20", "repo": "https://github.com/langchain-ai/langchain.git", "docs_subpath": "docs/docs", "label": "v0.1"},
    {"tag": "langchain==0.3.0", "repo": "https://github.com/langchain-ai/langchain.git", "docs_subpath": "docs/docs", "label": "v0.3"},
    {"tag": "main", "repo": "https://github.com/langchain-ai/docs.git", "docs_subpath": "src", "label": "v1"},
]

WORK_DIR = Path(__file__).parent / "clones"
OUT_PATH = Path(__file__).parent / "raw_docs.jsonl"


def clone(version):
    dest = WORK_DIR / version["label"]
    if dest.exists():
        return dest
    cmd = ["git", "clone", "--depth", "1"]
    if version["tag"] != "main":
        cmd += ["--branch", version["tag"]]
    cmd += [version["repo"], str(dest)]
    subprocess.run(cmd, check=True, capture_output=True)
    return dest


def collect_docs(clone_path, subpath, label):
    root = clone_path / subpath
    if not root.exists():
        return []
    docs = []
    for ext in ("*.mdx", "*.md"):
        for f in root.rglob(ext):
            text = f.read_text(errors="ignore")
            if len(text.strip()) < 200:
                continue
            docs.append({
                "version": label,
                "path": f.relative_to(root).as_posix(),
                "text": text,
            })
    return docs


def main():
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    all_docs = []
    for v in VERSIONS:
        clone_path = clone(v)
        docs = collect_docs(clone_path, v["docs_subpath"], v["label"])
        print(f"{v['label']}: {len(docs)} docs")
        all_docs.extend(docs)

    with open(OUT_PATH, "w") as f:
        for d in all_docs:
            f.write(json.dumps(d) + "\n")

    print(f"total {len(all_docs)} docs -> {OUT_PATH}")


if __name__ == "__main__":
    main()