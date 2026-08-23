# Eval Platform

An evaluation and regression-gating harness for LLM/RAG systems, built and
tested against a LangChain-documentation Q&A system spanning versions
v0.1-v0.3.

Status: Week 1 in progress (golden dataset drafted, baseline runner scaffolded,
subject RAG system not yet wired up).

## Why this project

Most GenAI portfolios show a RAG chatbot. Very few show a system that
decides, automatically and with statistical rigor, whether a change to a
prompt or model is safe to ship. This is that second thing.

## What it does (target state)

Offline eval (Weeks 1-3, original scope):

- Runs a golden Q&A set against a subject RAG system and scores answers on
  faithfulness, retrieval precision, and hallucination rate
- Validates its own LLM-judge against a human-labeled subset (reports
  agreement rate, not just judge scores)
- Compares two configs with confidence intervals, not just raw score deltas
- Blocks a GitHub PR automatically if eval metrics regress below threshold

Production-pattern extensions (added to match current 2026 practice --
LLM observability has shifted from offline-only eval toward trace-based,
continuous production evaluation; see README "Why the extra scope" below):

- OpenTelemetry-instrumented tracing: every retrieval, LLM call, and tool
  invocation is a nested span, not a flat log line
- A simulated production traffic loop with continuous online LLM-judge
  scoring and drift alerting, not just one-shot offline eval
- A trace-to-dataset feedback loop: flagged/low-scoring production-simulated
  queries get curated back into the golden dataset automatically

## Why the extra scope

As of 2026, LLM observability tooling (Langfuse, LangSmith, Braintrust,
Arize, Opik) treats the trace -- not the offline batch eval run -- as the
primary object, and evaluation has shifted toward continuous scoring of
live traffic rather than pre-deployment-only testing. An eval harness that
only runs offline batch jobs demonstrates 2023-era practice. Building the
trace + online-eval loop is what makes this reflect current production
patterns rather than a static regression script.

## Repo structure

- `datasets/` -- golden Q&A sets, versioned
- `configs/` -- prompt/model/retrieval configs
- `evaluators/` -- LLM-judge, retrieval metrics, human-label loader
- `runners/` -- executes a config against a dataset (offline eval)
- `tracing/` -- OpenTelemetry instrumentation for the subject system
- `online_eval/` -- simulated production loop, continuous scoring, drift
  alerting, trace-to-dataset feedback
- `stats/` -- significance testing between runs
- `dashboard/` -- results viewer (offline + online eval trends)
- `.github/workflows/` -- CI eval gate

## Failure log

(Populated as real issues surface during the build -- this is the
highest-value section of the README for interviews. Keep it honest.)

- Live/time-sensitive data sources break reproducible eval unless snapshotted
  as fixtures (surfaced while drafting an earlier domain draft)
