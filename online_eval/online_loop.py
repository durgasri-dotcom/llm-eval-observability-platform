import json
import time
import uuid
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Optional

from tracing.tracer import Tracer


@dataclass
class OnlineEvalRecord:
    record_id: str
    trace_id: str
    query: str
    answer: str
    judge_score: float
    latency_ms: float
    timestamp: float
    flagged: bool


class DriftMonitor:
    def __init__(self, baseline_mean, window_size=50, threshold=0.15):
        self.baseline_mean = baseline_mean
        self.window = deque(maxlen=window_size)
        self.threshold = threshold

    def update(self, score):
        self.window.append(score)
        if len(self.window) < self.window.maxlen:
            return None
        rolling_mean = sum(self.window) / len(self.window)
        if self.baseline_mean - rolling_mean > self.threshold:
            return f"drift: rolling_mean={rolling_mean:.3f} baseline={self.baseline_mean:.3f}"
        return None


def score_with_judge(query, answer):
    raise NotImplementedError


def run_online_loop(query_source, subject_system, baseline_mean, flag_threshold, feedback_queue_path, max_iterations=100):
    tracer = Tracer()
    monitor = DriftMonitor(baseline_mean=baseline_mean)
    records = []

    for _ in range(max_iterations):
        query = query_source()
        trace_id = tracer.start_trace()
        start = time.time()

        with tracer.span("subject_system_call"):
            answer = subject_system(query)
        with tracer.span("online_judge_score"):
            score = score_with_judge(query, answer)

        latency_ms = (time.time() - start) * 1000
        flagged = score < flag_threshold

        record = OnlineEvalRecord(
            record_id=str(uuid.uuid4())[:8],
            trace_id=trace_id,
            query=query,
            answer=answer,
            judge_score=score,
            latency_ms=latency_ms,
            timestamp=time.time(),
            flagged=flagged,
        )
        records.append(record)

        if flagged:
            _append_to_feedback_queue(record, feedback_queue_path)

        alert = monitor.update(score)
        if alert:
            print(alert)

    return records


def _append_to_feedback_queue(record, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if Path(path).exists():
        with open(path) as f:
            existing = json.load(f)
    existing.append(asdict(record))
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)