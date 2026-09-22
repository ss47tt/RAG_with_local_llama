"""
Evaluate retrieval quality against the testset from eval/testset.py, at
every stage of the pipeline: dense-only, sparse-only, RRF-fused, and
the final reranked+MMR'd set actually handed to the LLM. Reports Hit
Rate@k, MRR, and nDCG@k per stage, so you can tell whether e.g.
reranking is actually helping, rather than eyeballing single questions.

Deliberately does NOT load the big generation model or run query
rewriting — testset questions are self-contained by construction (each
was generated from a single chunk with no conversation history), so
evaluating raw retrieval against them in isolation is both faster and
a cleaner signal than routing through the full pipeline.

Usage:
    python -m eval.retrieval_eval --testset eval/testset.json
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

try:
    from rag_app.config import settings
    from rag_app.graph import fuse_rrf, rerank_and_diversify
    from rag_app.ingest import get_vectorstore, load_cached_chunks
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from rag_app.config import settings
    from rag_app.graph import fuse_rrf, rerank_and_diversify
    from rag_app.ingest import get_vectorstore, load_cached_chunks


class RetrievalResources:
    """Everything retrieval needs, minus the generation model — evaluating
    retrieval quality doesn't require ever calling the LLM, so skipping
    it here makes the eval loop much faster to iterate on."""

    def __init__(self):
        # Built once and passed to get_vectorstore so the embedding
        # model isn't loaded twice (see the equivalent fix/comment in
        # RAGPipeline.__init__ in rag_app/graph.py).
        self.embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model_id)
        self.vectorstore = get_vectorstore(embeddings=self.embeddings)

        chunks = load_cached_chunks()
        self.bm25_retriever = BM25Retriever.from_documents(chunks)
        self.bm25_retriever.k = settings.sparse_top_k

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cross_encoder = CrossEncoder(settings.cross_encoder_model_id, device=device)


def _rank_of(chunk_id: str, docs: list) -> int | None:
    """1-indexed rank of chunk_id within docs, or None if absent."""
    for i, doc in enumerate(docs, start=1):
        if doc.metadata.get("chunk_id") == chunk_id:
            return i
    return None


def _hit(rank) -> float:
    return 1.0 if rank is not None else 0.0


def _reciprocal_rank(rank) -> float:
    return 1.0 / rank if rank is not None else 0.0


def _ndcg(rank) -> float:
    # Single relevant document per question, so IDCG (ideal DCG, i.e.
    # the relevant doc at rank 1) is always 1/log2(2) = 1 — nDCG
    # reduces to plain DCG at the found rank.
    return 1.0 / math.log2(rank + 1) if rank is not None else 0.0


def evaluate(testset: list, resources: RetrievalResources) -> dict:
    stage_k = {
        "dense": settings.dense_top_k,
        "sparse": settings.sparse_top_k,
        "fused": settings.rrf_top_n,
        "final": settings.top_k,
    }
    per_stage_ranks: dict = {stage: [] for stage in stage_k}

    for i, entry in enumerate(testset, start=1):
        question = entry["question"]
        expected_id = entry["chunk_id"]

        dense_docs = resources.vectorstore.as_retriever(
            search_kwargs={"k": settings.dense_top_k}
        ).invoke(question)
        sparse_docs = resources.bm25_retriever.invoke(question)
        fused_docs = fuse_rrf(dense_docs, sparse_docs)
        final_docs = rerank_and_diversify(
            question, fused_docs, resources.cross_encoder, resources.embeddings
        )

        stage_docs = {
            "dense": dense_docs,
            "sparse": sparse_docs,
            "fused": fused_docs,
            "final": final_docs,
        }
        for stage, docs in stage_docs.items():
            per_stage_ranks[stage].append(_rank_of(expected_id, docs))

        print(f"[retrieval_eval] ({i}/{len(testset)}) {question[:70]}")

    results = {}
    for stage, ranks in per_stage_ranks.items():
        n = len(ranks)
        results[stage] = {
            "k": stage_k[stage],
            "hit_rate": sum(_hit(r) for r in ranks) / n if n else None,
            "mrr": sum(_reciprocal_rank(r) for r in ranks) / n if n else None,
            "ndcg": sum(_ndcg(r) for r in ranks) / n if n else None,
        }
    return results


def print_report(results: dict) -> None:
    print("\nRetrieval quality by stage:")
    print(f"{'stage':<8} {'k':>4} {'hit_rate':>10} {'mrr':>8} {'ndcg':>8}")
    for stage, m in results.items():
        if m["hit_rate"] is None:
            print(f"{stage:<8} {m['k']:>4} {'n/a':>10} {'n/a':>8} {'n/a':>8}")
        else:
            print(
                f"{stage:<8} {m['k']:>4} {m['hit_rate']:>10.3f} {m['mrr']:>8.3f} {m['ndcg']:>8.3f}"
            )


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality against a testset.")
    parser.add_argument("--testset", type=str, default="eval/testset.json")
    parser.add_argument(
        "--output", type=str, default=None, help="Optional path to write results as JSON"
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

    print(f"[retrieval_eval] loaded {len(testset)} questions, loading retrieval resources...")
    resources = RetrievalResources()

    results = evaluate(testset, resources)
    print_report(results)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\n[retrieval_eval] wrote results to '{output_path}'")


if __name__ == "__main__":
    main()
