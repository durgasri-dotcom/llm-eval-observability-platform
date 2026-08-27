import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "evaluators"))
from llm_judge import score_answer

RESULTS_DIR = Path(__file__).parent.parent / "results"
DEFAULT_THRESHOLD = 4.0


def run_eval(dataset, config, output):
    subprocess.run(
        [sys.executable, str(Path(__file__).parent / "run_eval.py"),
         "--dataset", dataset, "--config", config, "--output", output],
        check=True,
    )


def score_and_check(output_path, threshold):
    results = json.load(open(output_path))
    scores = []
    for r in results:
        score = score_answer(r["question"], r["answer"])
        scores.append(score)
        print(f"{r['question_id']}: {score}")

    mean_score = sum(s for s in scores if s is not None) / len(scores)
    print(f"mean faithfulness: {mean_score:.2f} (threshold: {threshold})")

    if mean_score < threshold:
        print(f"FAIL: mean score {mean_score:.2f} below threshold {threshold}")
        return False

    print("PASS")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="../datasets/golden_smoke.json")
    parser.add_argument("--config", default="../configs/baseline.yaml")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()

    output_path = RESULTS_DIR / "ci_run_output.json"
    run_eval(args.dataset, args.config, str(output_path))
    passed = score_and_check(output_path, args.threshold)

    sys.exit(0 if passed else 1)