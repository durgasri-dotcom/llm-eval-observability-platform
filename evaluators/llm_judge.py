import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "subject_system"))
from langchain.chat_models import init_chat_model

JUDGE_MODEL = "openai:gpt-4o-mini"

JUDGE_PROMPT = """You are evaluating an AI system's answer to a documentation question.

Question: {question}

Answer given: {answer}

Score the answer's faithfulness on a 1-5 scale:
5 = fully grounded, no fabrication, correctly declines if info is missing
4 = mostly grounded, minor unsupported detail
3 = partially grounded, some invented specifics
2 = mostly fabricated, little grounding
1 = confidently wrong or fully fabricated

Respond with only a single digit 1-5, nothing else."""

_judge = None


def get_judge():
    global _judge
    if _judge is None:
        _judge = init_chat_model(JUDGE_MODEL)
    return _judge


def score_answer(question, answer):
    judge = get_judge()
    prompt = JUDGE_PROMPT.format(question=question, answer=answer)
    response = judge.invoke([("human", prompt)])
    text = response.content.strip()
    try:
        return int(text[0])
    except (ValueError, IndexError):
        return None


def score_run_output(run_output_path, out_path):
    results = json.load(open(run_output_path))
    scored = []
    for r in results:
        score = score_answer(r["question"], r["answer"])
        scored.append({**r, "judge_score": score})
        print(f"{r['question_id']}: {score}")

    with open(out_path, "w") as f:
        json.dump(scored, f, indent=2)

    print(f"{len(scored)} scored -> {out_path}")


if __name__ == "__main__":
    import sys
    input_name = sys.argv[1] if len(sys.argv) > 1 else "run_output.json"
    output_name = "scored_" + input_name
    score_run_output(
        Path(__file__).parent.parent / "results" / input_name,
        Path(__file__).parent.parent / "results" / output_name,
    )