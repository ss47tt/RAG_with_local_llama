"""
Evaluate GENERATION quality — as opposed to eval/retrieval_eval.py,
which only checks whether the right chunk was retrieved. This runs the
full live pipeline (retrieval through generation) for each question in
a testset, then asks an LLM judge to score the resulting answer on:

- FAITHFULNESS (1-5): is every claim in the answer actually supported
  by the context that was retrieved for it? Catches hallucination —
  answers that go beyond, or contradict, what was actually found.
- RELEVANCE (1-5): does the answer actually address the question
  asked? A perfectly faithful answer can still be non-responsive.

These are genuinely separate failure modes from retrieval quality: a
pipeline can retrieve the exact right chunk and still generate a
faithful-but-irrelevant or unfaithful-but-on-topic answer, which
eval/retrieval_eval.py has no way to catch.

The judge is the SAME local Llama-3.2-3B-Instruct instance already
loaded for generation (reused via generate_once, not a second copy —
loading the model twice would likely blow your VRAM budget). Worth
being upfront about the tradeoff: a 3B model judging its own outputs
is a weaker, more self-correlated signal than an independent stronger
judge would be — it can share the same blind spots it's supposed to
be catching. If you have API access to a larger model, swapping the
judge call in _judge() below for an API request is the natural upgrade
path; the rest of this script (parsing, aggregation, flagging) doesn't
need to change.

Usage:
    python -m eval.generation_eval --testset eval/testset.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

try:
    from rag_app.graph import RAGPipeline, format_context, generate_once
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from rag_app.graph import RAGPipeline, format_context, generate_once

JUDGE_SYSTEM_PROMPT = (
    "You are an evaluator for a retrieval-augmented QA system. You will "
    "be given CONTEXT (passages retrieved for a question), a QUESTION, "
    "and an ANSWER the system generated. Score the ANSWER on two "
    "dimensions, each from 1 to 5.\n\n"
    "FAITHFULNESS (1-5): is every claim in the ANSWER actually "
    "supported by the CONTEXT? 5 = every claim is directly supported, "
    "no fabrication. 1 = the answer contains significant claims not "
    "found in, or contradicted by, the context. An answer that "
    "correctly says the context doesn't contain the information should "
    "score 5 for faithfulness, even though it doesn't fully answer the "
    "question — that's a relevance issue, not a faithfulness one.\n\n"
    "RELEVANCE (1-5): does the ANSWER actually address the QUESTION "
    "asked? 5 = fully and directly answers it. 1 = off-topic or "
    "non-responsive, even if accurate about something else.\n\n"
    "Respond with ONLY a JSON object in exactly this format, nothing "
    "else, no markdown fences:\n"
    '{"faithfulness": <int 1-5>, "faithfulness_reason": "<one short '
    'sentence>", "relevance": <int 1-5>, "relevance_reason": "<one '
    'short sentence>"}'
)

JUDGE_MAX_NEW_TOKENS = 200
# Scores at or below this are printed as flagged examples worth a
# manual look, rather than buried in an averaged number.
FLAG_THRESHOLD = 2

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_FAITHFULNESS_FALLBACK_RE = re.compile(r'"?faithfulness"?\s*[:=]\s*(\d)')
_RELEVANCE_FALLBACK_RE = re.compile(r'"?relevance"?\s*[:=]\s*(\d)')


def _parse_judge_output(raw: str) -> dict | None:
    """Extract the judge's scores from its (hopefully JSON) output.

    do_sample=False means retrying an identical prompt just gets the
    identical malformed output again, so there's no point retrying —
    instead this tries a strict JSON parse first, then falls back to
    pulling the two numbers out with regex if the model wrapped the
    JSON in prose or a markdown fence despite being told not to.
    """
    match = _JSON_BLOCK_RE.search(raw)
    if match:
        try:
            data = json.loads(match.group(0))
            return {
                "faithfulness": int(data["faithfulness"]),
                "faithfulness_reason": str(data.get("faithfulness_reason", "")),
                "relevance": int(data["relevance"]),
                "relevance_reason": str(data.get("relevance_reason", "")),
            }
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            pass

    faith_match = _FAITHFULNESS_FALLBACK_RE.search(raw)
    rel_match = _RELEVANCE_FALLBACK_RE.search(raw)
    if faith_match and rel_match:
        return {
            "faithfulness": int(faith_match.group(1)),
            "faithfulness_reason": "",
            "relevance": int(rel_match.group(1)),
            "relevance_reason": "",
        }

    return None


def _judge(pipeline: RAGPipeline, question: str, context: str, answer: str) -> dict | None:
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nANSWER:\n{answer}",
        },
    ]
    # Reuses the pipeline's own tokenizer/model — no second model load.
    raw = generate_once(
        pipeline.tokenizer, pipeline.model, messages, max_new_tokens=JUDGE_MAX_NEW_TOKENS
    )
    return _parse_judge_output(raw)


def evaluate(testset: list, pipeline: RAGPipeline) -> dict:
    scored = []
    parse_failures = 0

    for i, entry in enumerate(testset, start=1):
        question = entry["question"]
        print(f"\n[generation_eval] ({i}/{len(testset)}) {question}")

        # Each testset question is independent — without this, RAGPipeline's
        # multi-turn memory would feed every PRIOR unrelated eval question
        # into this one's rewrite_query and generate prompts, silently
        # corrupting both retrieval and the answer being judged.
        pipeline.reset_history()

        result = pipeline.query(question)
        answer = result["answer"]
        context = format_context(result["documents"])

        judgment = _judge(pipeline, question, context, answer)
        if judgment is None:
            parse_failures += 1
            print("[generation_eval]   (judge output could not be parsed — skipped)")
            continue

        judgment["question"] = question
        judgment["answer"] = answer
        scored.append(judgment)
        print(
            f"[generation_eval]   faithfulness={judgment['faithfulness']} "
            f"relevance={judgment['relevance']}"
        )

    n = len(scored)
    flagged = [
        s
        for s in scored
        if s["faithfulness"] <= FLAG_THRESHOLD or s["relevance"] <= FLAG_THRESHOLD
    ]

    summary = {
        "n_scored": n,
        "n_parse_failures": parse_failures,
        "avg_faithfulness": sum(s["faithfulness"] for s in scored) / n if n else None,
        "avg_relevance": sum(s["relevance"] for s in scored) / n if n else None,
        "flagged": flagged,
        "all_scored": scored,
    }
    return summary


def print_report(summary: dict) -> None:
    print("\nGeneration quality:")
    print(f"  scored:          {summary['n_scored']}")
    print(f"  parse failures:  {summary['n_parse_failures']}")
    if summary["avg_faithfulness"] is not None:
        print(f"  avg faithfulness: {summary['avg_faithfulness']:.2f} / 5")
        print(f"  avg relevance:    {summary['avg_relevance']:.2f} / 5")

    if summary["flagged"]:
        print(f"\nFlagged for review (score <= {FLAG_THRESHOLD} on either dimension):")
        for s in summary["flagged"]:
            print(f"  - Q: {s['question']}")
            answer_preview = s["answer"][:150] + ("..." if len(s["answer"]) > 150 else "")
            print(f"    A: {answer_preview}")
            print(f"    faithfulness={s['faithfulness']} ({s['faithfulness_reason']})")
            print(f"    relevance={s['relevance']} ({s['relevance_reason']})")
    else:
        print(f"\nNothing flagged at or below {FLAG_THRESHOLD}/5 on either dimension.")


def main():
    parser = argparse.ArgumentParser(
        description="LLM-as-judge evaluation of generation quality (faithfulness + relevance)."
    )
    parser.add_argument("--testset", type=str, default="eval/testset.json")
    parser.add_argument(
        "--output", type=str, default=None, help="Optional path to write full results as JSON"
    )
    args = parser.parse_args()

    testset_path = Path(args.testset)
    if not testset_path.exists():
        raise SystemExit(
            f"'{testset_path}' not found. Generate one first with "
            "`python -m eval.testset --output eval/testset.json`."
        )
    with open(testset_path, "r", encoding="utf-8") as f:
        testset = json.load(f)
    if not testset:
        raise SystemExit(f"'{testset_path}' is empty.")

    print(f"[generation_eval] loaded {len(testset)} questions, loading pipeline...")
    pipeline = RAGPipeline()

    summary = evaluate(testset, pipeline)
    print_report(summary)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n[generation_eval] wrote results to '{output_path}'")


if __name__ == "__main__":
    main()
