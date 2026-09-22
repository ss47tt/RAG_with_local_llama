"""
Central configuration for the RAG app.

All tunables live here so the ingestion pipeline and the LangGraph
pipeline stay in sync (e.g. which embedding model was used to build
the Chroma index must match the one used at query time).
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # --- Generation model -------------------------------------------------
    # Gated model on the Hub — you must `huggingface-cli login` (or set
    # HF_TOKEN) with an account that has accepted Meta's license first.
    llm_model_id: str = "meta-llama/Llama-3.2-3B-Instruct"
    max_new_tokens: int = 512
    do_sample: bool = False
    # "auto" lets transformers pick the best available device/dtype
    # (GPU + bf16/fp16 if present, otherwise CPU + fp32).
    device_map: str = "auto"

    # --- Embedding model -----------------------------------------------
    embedding_model_id: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Chunking ---------------------------------------------------------
    chunk_size: int = 1200
    chunk_overlap: int = 200

    # --- Retrieval: hybrid dense + sparse -----------------------------
    # Each retriever pulls this many candidates *before* fusion/reranking.
    # Wider than the final top_k so RRF and the reranker have real signal
    # to work with.
    dense_top_k: int = 15
    sparse_top_k: int = 15
    # Reciprocal Rank Fusion constant. 60 is the standard default from the
    # original RRF paper — higher values flatten the influence of rank.
    rrf_k: int = 60
    # How many fused candidates survive to the reranking step.
    rrf_top_n: int = 12

    # --- Reranking ------------------------------------------------------
    cross_encoder_model_id: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # Final number of chunks handed to the LLM after reranking.
    top_k: int = 5
    # MMR (Maximal Marginal Relevance) balances relevance against
    # diversity when picking the final top_k from the reranked
    # candidates: 1.0 = pure relevance (old behavior), 0.0 = pure
    # diversity. 0.7 keeps relevance dominant but leaves room to avoid
    # filling every slot with near-duplicate chunks.
    mmr_lambda: float = 0.7

    # --- Query rewriting ------------------------------------------------
    rewrite_query: bool = True
    rewrite_max_new_tokens: int = 64

    # --- Multi-turn memory ------------------------------------------------
    # How many previous (question, answer) turns are kept and fed back
    # into query rewriting + generation, so follow-ups like "what about
    # the 70B version?" can be resolved against the conversation so far.
    max_history_turns: int = 4

    # --- Generation streaming --------------------------------------------
    stream_to_stdout: bool = True

    # --- Storage ------------------------------------------------------
    persist_dir: str = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
    collection_name: str = "rag_docs"
    # Raw chunked Documents, pickled alongside the Chroma store, so BM25
    # (which needs the actual text corpus, not just embeddings) can be
    # rebuilt at query time without re-ingesting.
    chunks_cache_path: str = os.path.join(
        os.path.dirname(__file__), "..", "chroma_db", "chunks.pkl"
    )


settings = Settings()
