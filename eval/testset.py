"""
Generate an LLM-synthesized evaluation set for retrieval testing.

Rather than pulling questions from a public QA dataset (built on a
totally different corpus than whatever you've ingested), this samples
random chunks straight from your own ingested documents, asks the
local Llama model to write one question that each sampled chunk
directly answers, and records that chunk's chunk_id as ground truth.
The result is a testset that's actually relevant to *your* documents,
and stays relevant no matter what you re-ingest next — just regenerate
it.

Usage:
    python -m eval.testset --n 30 --output eval/testset.json
"""

import argparse
import json
import random
import sys
from pathlib import Path

try:
    from rag_app.graph import generate_once, load_llm
    from rag_app.ingest import load_cached_chunks
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from rag_app.graph import generate_once, load_llm
    from rag_app.ingest import load_cached_chunks

QUESTION_SYSTEM_PROMPT = (
    "You are creating a test question for a retrieval evaluation. Given "
    "a passage, write ONE specific question that this passage directly "
    "and fully answers. The question must be answerable using ONLY this "
    "passage — don't ask something that needs outside context. Don't "
    "refer to \"the passage\", \"the text\", or \"the document\" in your "
    "question — write it as someone would naturally ask it. Reply with "
    "ONLY the question, nothing else."
)

# Reject a generated question if it leaks a meta-reference to "the
# passage" (the model sometimes does this despite the instruction not
# to) — such questions are unrealistic and depend on context a real
# user query wouldn't have.
_BANNED_PHRASES = ("the passage", "the text", "the document", "the excerpt", "this passage")

# do_sample=False (greedy decoding, see config.py) means retrying an
# identical prompt reproduces the identical output every time — so
# there's no retry loop here; a chunk that produces an invalid question
# is just skipped, same reasoning as generation_eval.py's judge parsing.
QUESTION_MAX_NEW_TOKENS = 80


def _is_valid_question(question: str) -> bool:
    if not question or len(question) > 300:
        return False
    lowered = question.lower()
    if any(phrase in lowered for phrase in _BANNED_PHRASES):
        return False
    return True


def _sample_chunks(min_chars: int, n: int, seed: int) -> list:
    chunks = load_cached_chunks()
    # Very short chunks (stray headers, table fragments) make poor,
    # under-specified questions — skip them.
    candidates = [c for c in chunks if len(c.page_content) >= min_chars]
    if not candidates:
        raise ValueError(
            f"No chunks with at least {min_chars} characters found. "
            "Lower --min-chars or re-ingest with bigger chunks."
        )
    rng = random.Random(seed)
    return rng.sample(candidates, min(n, len(candidates)))


def generate_testset(n: int, min_chars: int = 200, seed: int = 42) -> list:
    sampled = _sample_chunks(min_chars=min_chars, n=n, seed=seed)
    print(f"[testset] sampled {len(sampled)} chunks, generating questions...")

    tokenizer, model = load_llm()

    entries = []
    skipped = 0
    for i, chunk in enumerate(sampled, start=1):
        messages = [
            {"role": "system", "content": QUESTION_SYSTEM_PROMPT},
            {"role": "user", "content": chunk.page_content},
        ]
        candidate = generate_once(
            tokenizer, model, messages, max_new_tokens=QUESTION_MAX_NEW_TOKENS
        )
        question = candidate if _is_valid_question(candidate) else None

        if question is None:
            skipped += 1
            continue

        entries.append(
            {
                "question": question,
                "chunk_id": chunk.metadata.get("chunk_id"),
                "source": chunk.metadata.get("source"),
                "page": chunk.metadata.get("page"),
            }
        )
        print(f"[testset] ({i}/{len(sampled)}) {question}")

    if skipped:
        print(f"[testset] skipped {skipped} chunks (question generation produced an invalid result)")
    print(f"[testset] generated {len(entries)} question/answer pairs")
    return entries


def main():
    parser = argparse.ArgumentParser(
        description="Generate an LLM-synthesized retrieval eval testset from your ingested corpus."
    )
    parser.add_argument("--n", type=int, default=30, help="Number of questions to generate")
    parser.add_argument(
        "--output", type=str, default="eval/testset.json", help="Where to write the testset JSON"
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=200,
        help="Skip chunks shorter than this (too little content for a good question)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for chunk sampling")
    args = parser.parse_args()

    entries = generate_testset(n=args.n, min_chars=args.min_chars, seed=args.seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)
    print(f"[testset] wrote {len(entries)} entries to '{output_path}'")


if __name__ == "__main__":
    main()
