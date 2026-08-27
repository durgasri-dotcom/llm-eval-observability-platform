# eval-platform

Eval + regression gating for LLM/RAG systems. Built it against a LangChain-docs
Q&A system (v0.1 -> v1.x) so there'd be an actual moving target to test on.

Still building this. Offline eval (dataset, judge, CI gate) mostly works.
Online/production loop is scaffolded but not wired up yet.

## why

Most portfolios have a RAG chatbot. Almost nobody ships something that decides,
automatically, whether a prompt/model change is safe to merge. That's this.

## what's actually working right now

- Corpus pulled from real LangChain repos at 3 version anchors (v0.1.20, v0.3.0,
  and current). Had to split it across two repos since LangChain moved its docs
  out of the main repo at some point after v0.3 and I didn't know that going in.
- Retrieval: bge-base-en-v1.5 embeddings + bge-reranker-base on top. Started
  with a fancier embedding model (Qwen3, top of the open-source leaderboard)
  and it OOM'd my machine trying to allocate 32GB for a causal attention mask.
  Switched to something that actually runs.
- Answer generation via `init_chat_model()` so the LLM provider isn't hardcoded
  — same pattern LangChain itself recommends, which felt right given what this
  project is testing.
- LLM judge (faithfulness, 1-5), checked against my own manual scoring on the
  same 7 questions. Agreed with me 4/7 times. See below, this was more useful
  than it sounds.

## what's not built yet

- CI gate (GitHub Action blocking a PR on regression)
- Confidence intervals for comparing two configs
- The online/production simulation loop (tracing + drift alerting scaffolding
  exists in `tracing/` and `online_eval/`, not connected to anything real)
- Golden dataset is only 7 questions right now, needs to be 30-50

## config comparison (top_k=5 vs top_k=2)

Ran the same 30-question set through baseline (top_k=5) and a reduced-context
config (top_k=2), judged both, compared with a paired bootstrap CI (10k
resamples) on the score differences rather than a t-test, since judge scores
are discrete 1-5 values, not continuous/normal.

mean baseline: 4.77, mean reduced: 4.87, 95% CI on the difference: [-0.2, 0.0]

Not significant. Can't confidently say cutting context in half hurt
faithfulness in this sample. Worth being precise here: reduced context scored
numerically higher, but the CI straddles zero, so that's noise, not a real
"less context is better" finding -- don't want to overclaim a result the
stats don't support.

This also lines up with the judge-resolution problem below: a metric that
can't detect the difference between 5 retrieved chunks and 2 retrieved
chunks may not have the resolution to catch real quality differences,
independent of whether those differences exist.

## judge calibration

Scored the same 7 answers myself before looking at the judge's output.
Agreement: 4/7.

Two disagreements were me being stricter than the judge — when the answer
said "the context doesn't mention this" for a made-up method/parameter, I
scored that lower than the judge did, because it dodges the question instead
of actually saying "this doesn't exist." The judge doesn't currently
distinguish those two things.

One disagreement went the other way — judge scored a detailed, well-grounded
migration answer a point lower than I did, and I genuinely can't tell why
from the content. Not assuming the judge was right just because it's the
"objective" one.

n=7 is small. Not rewriting the judge prompt off this alone — revisiting once
the dataset's bigger.

## real problems hit along the way

- Golden dataset built on live/updating data sources isn't reproducible —
  run the same eval twice, get different ground truth. Learned this before
  even picking a domain, when I almost built this against a live threat-intel
  feed instead.
- Retrieval ranked a page that _explains_ what a chat model is above the page
  that actually _shows you how to make one_, for a "how do I..." query.
  Added reranking, which fixed this specific case but not universally —
  same topic, different phrasing of the question, still sometimes misses the
  right doc.
- Reranker scores cluster tight (0.96-0.99) when several results are actually
  good — that's the model being confident about all of them, not broken. Can't
  use the raw score as a hard threshold later for the online drift alerting.
- First embedding model pick crashed on my hardware. Top of a leaderboard
  doesn't mean it fits your machine.
- CI gate took 5 real fixes to actually go green: an unquoted `on:` in the
  workflow YAML got parsed as the boolean `True` instead of the string key
  (classic YAML "Norway problem"), which silently killed the trigger entirely;
  a few files saved empty in the editor and made it into commits without
  content, caught only by checking the raw file on GitHub, not locally;
  `actions/checkout@v4` doesn't pull Git LFS content by default, so the vector
  index came through as a tiny LFS pointer file and crashed `pickle.load`
  with a cryptic "invalid load key" error until `lfs: true` got added; and a
  GitHub Secret got set to the literal string `OPENAI_API_KEY=sk-...` instead
  of just the key, which only showed up as a 401 once the run actually
  reached the API call. None of these were logic bugs in the eval code
  itself -- all infrastructure/config, and all things that only show up once
  something runs somewhere other than your own machine.

## repo layout

```
datasets/       golden Q&A sets
configs/        prompt/model configs
evaluators/     llm judge, retrieval metrics
runners/        offline eval runner
subject_system/ the actual RAG system being tested
corpus_build/   scripts that pull and filter the langchain docs corpus
tracing/        otel-style tracing, scaffolded, not wired up
online_eval/    production-sim loop, scaffolded, not wired up
```
