import argparse
import json
import sys
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "subject_system"))
from answer import answer_question


@dataclass
class RunResult:
    run_id: str
    question_id: str
    question: str
    answer: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    estimated_cost_usd: float
    config_name: str
    timestamp: float


def load_dataset(path):
    with open(path) as f:
        return json.load(f)


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def call_subject_system(question, config):
    top_k = config.get("retrieval", {}).get("top_k", 5)
    result = answer_question(question, model_id=config["model"], top_k=top_k)
    return {
        "answer": result["answer"],
        "prompt_tokens": result["prompt_tokens"],
        "completion_tokens": result["completion_tokens"],
    }


def estimate_cost(prompt_tokens, completion_tokens, config):
    rate = config.get("cost_per_1k_tokens", {"prompt": 0.0, "completion": 0.0})
    return (prompt_tokens / 1000 * rate["prompt"]) + (completion_tokens / 1000 * rate["completion"])


def run(dataset_path, config_path, output_path):
    dataset = load_dataset(dataset_path)
    config = load_config(config_path)
    run_id = str(uuid.uuid4())[:8]
    results = []

    for item in dataset["items"]:
        start = time.time()
        try:
            output = call_subject_system(item["question"], config)
        except NotImplementedError:
            print(f"[SKIP] {item['id']}")
            continue
        latency_ms = (time.time() - start) * 1000

        result = RunResult(
            run_id=run_id,
            question_id=item["id"],
            question=item["question"],
            answer=output["answer"],
            latency_ms=latency_ms,
            prompt_tokens=output["prompt_tokens"],
            completion_tokens=output["completion_tokens"],
            estimated_cost_usd=estimate_cost(output["prompt_tokens"], output["completion_tokens"], config),
            config_name=config.get("name", "unnamed"),
            timestamp=time.time(),
        )
        results.append(asdict(result))
        print(f"[OK] {item['id']} {latency_ms:.0f}ms")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"{run_id}: {len(results)} results -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="../results/run_output.json")
    args = parser.parse_args()
    run(args.dataset, args.config, args.output)