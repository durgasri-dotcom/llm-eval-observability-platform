import json
import sys
from pathlib import Path

import numpy as np

N_BOOTSTRAP = 10000


def load_scores(path):
    data = json.load(open(path))
    return {d["question_id"]: d["judge_score"] for d in data}


def paired_bootstrap_ci(scores_a, scores_b, n_bootstrap=N_BOOTSTRAP, alpha=0.05):
    ids = sorted(set(scores_a) & set(scores_b))
    a = np.array([scores_a[i] for i in ids])
    b = np.array([scores_b[i] for i in ids])
    diffs = a - b

    boot_means = []
    rng = np.random.default_rng(0)
    for _ in range(n_bootstrap):
        sample = rng.choice(diffs, size=len(diffs), replace=True)
        boot_means.append(sample.mean())
    boot_means = np.array(boot_means)

    lower = np.percentile(boot_means, 100 * alpha / 2)
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    observed_mean_diff = diffs.mean()
    significant = not (lower <= 0 <= upper)

    return {
        "n": len(ids),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "mean_diff": float(observed_mean_diff),
        "ci_95": [float(lower), float(upper)],
        "significant_at_0.05": significant,
    }


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python significance.py <scored_a.json> <scored_b.json>")
        sys.exit(1)

    results_dir = Path(__file__).parent.parent / "results"
    scores_a = load_scores(results_dir / sys.argv[1])
    scores_b = load_scores(results_dir / sys.argv[2])

    result = paired_bootstrap_ci(scores_a, scores_b)
    print(json.dumps(result, indent=2))