"""
LangGraph RAG pipeline:

    START
      -> rewrite_query        (LLM turns the raw question into a search query,
                                using chat_history to resolve follow-ups)
      -> retrieve_dense        (Chroma similarity search)      \
      -> retrieve_sparse        (BM25 keyword search)            } run independently,
      -> fuse_rrf                (Reciprocal Rank Fusion merges both lists)
      -> rerank                    (cross-encoder scores + MMR picks a diverse top_k)
      -> generate                    (Llama answers, streamed token-by-token,
                                        with chat_history as prior turns)
    END

State flows through a TypedDict so each node only reads/writes the
fields it needs. `documents` holds whatever the *next* node should
treat as "the current candidate set" — dense/sparse/fused are kept
as separate fields so fuse_rrf can see both ranked lists at once.
`chat_history` carries prior (question, answer) turns for the whole
graph run; RAGPipeline owns the actual running history across calls.
"""

import threading
from typing import List, TypedDict

import numpy as np
import torch
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import END, START, StateGraph
from sentence_transformers import CrossEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

try:
    from .config import settings
    from .ingest import get_vectorstore, load_cached_chunks
except ImportError:  # running as a plain script rather than a package
    from config import settings
    from ingest import get_vectorstore, load_cached_chunks

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using the provided "
    "context. Read all the context snippets carefully — the answer is "
    "often present even if phrased differently than the question, or "
    "split across more than one snippet. Synthesize an answer from "
    "whatever relevant information IS there. Only say you don't know if "
    "none of the snippets contain anything relevant to the question. "
    "Cite which snippet you used, e.g. [1], [2]."
)

REWRITE_SYSTEM_PROMPT = (
    "Rewrite the user's LATEST question into a single, self-contained "
    "search query suitable for a document retrieval system. Use the "
    "conversation so far to resolve any pronouns or references (e.g. "
    "\"it\", \"the 70B version\", \"that model\") into concrete terms. "
    "Keep important proper nouns and keywords, and keep it concise. "
    "Reply with ONLY the rewritten query and nothing else."
)


class ChatTurn(TypedDict):
    question: str
    answer: str


class RAGState(TypedDict):
    question: str
    chat_history: List[ChatTurn]
    rewritten_query: str
    dense_docs: List[Document]
    sparse_docs: List[Document]
    fused_docs: List[Document]
    documents: List[Document]
    answer: str


def format_context(docs: List[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[{i}] (source: {source})\n{doc.page_content}")
    return "\n\n".join(parts)


def _history_to_messages(chat_history: List[ChatTurn]) -> list:
    """Turn prior (question, answer) turns into alternating chat
    messages, most recent max_history_turns only."""
    messages = []
    for turn in chat_history[-settings.max_history_turns :]:
        messages.append({"role": "user", "content": turn["question"]})
        messages.append({"role": "assistant", "content": turn["answer"]})
    return messages


def _mmr_select(
    candidates: List[Document],
    relevance_scores: np.ndarray,
    embeddings: np.ndarray,
    k: int,
    lambda_mult: float,
) -> List[Document]:
    """Greedy Maximal Marginal Relevance selection.

    At each step, picks the candidate maximizing
        lambda * relevance - (1 - lambda) * max_similarity_to_already_selected
    so chunks that are both relevant AND different from what's already
    picked win out over a fifth near-duplicate of chunk #1.
    """
    if len(candidates) <= k:
        order = np.argsort(-relevance_scores)
        return [candidates[i] for i in order]

    # Min-max normalize relevance scores to [0, 1] so they're on a
    # comparable scale to cosine similarity (also in [0, 1] here, since
    # embeddings are normalized below).
    lo, hi = relevance_scores.min(), relevance_scores.max()
    norm_relevance = (relevance_scores - lo) / (hi - lo) if hi > lo else np.ones_like(relevance_scores)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normed_embeddings = embeddings / norms

    selected_idx: List[int] = []
    remaining_idx = list(range(len(candidates)))

    while remaining_idx and len(selected_idx) < k:
        best_idx, best_score = None, -np.inf
        for idx in remaining_idx:
            if selected_idx:
                sims = normed_embeddings[idx] @ normed_embeddings[selected_idx].T
                redundancy = float(np.max(sims))
            else:
                redundancy = 0.0
            mmr_score = lambda_mult * norm_relevance[idx] - (1 - lambda_mult) * redundancy
            if mmr_score > best_score:
                best_score, best_idx = mmr_score, idx
        selected_idx.append(best_idx)
        remaining_idx.remove(best_idx)

    return [candidates[i] for i in selected_idx]


def fuse_rrf(dense_docs: List[Document], sparse_docs: List[Document]) -> List[Document]:
    """Reciprocal Rank Fusion: score = sum(1 / (rrf_k + rank)) across
    every ranked list a chunk appears in. Chunks retrieved by BOTH the
    dense and sparse retrievers naturally float to the top.

    Module-level (not a method) so eval scripts can run the exact same
    fusion logic the live pipeline uses, without needing a full
    RAGPipeline (and its generation model) just to check retrieval
    quality.
    """
    scores: dict = {}
    lookup: dict = {}

    for ranked_list in (dense_docs, sparse_docs):
        for rank, doc in enumerate(ranked_list):
            # Fall back to content hash if chunk_id is missing (e.g.
            # older ingested data) so fusion still works.
            cid = doc.metadata.get("chunk_id") or hash(doc.page_content)
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (settings.rrf_k + rank + 1)
            lookup.setdefault(cid, doc)

    ranked_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [lookup[cid] for cid in ranked_ids[: settings.rrf_top_n]]


def rerank_and_diversify(
    query: str,
    docs: List[Document],
    cross_encoder: CrossEncoder,
    embeddings: HuggingFaceEmbeddings,
) -> List[Document]:
    """Cross-encoder scoring + MMR selection down to settings.top_k.

    Module-level for the same reason as fuse_rrf above — an eval
    script can call this directly with its own (lighter-weight)
    cross_encoder/embeddings instances.
    """
    if not docs:
        return []

    pairs = [(query, doc.page_content) for doc in docs]
    relevance_scores = np.array(cross_encoder.predict(pairs))

    # Embed every candidate so MMR can measure how similar each one is
    # to what's already been picked — this is what lets it skip a 4th
    # near-duplicate chunk in favor of something that actually covers
    # new ground.
    doc_embeddings = np.array(embeddings.embed_documents([doc.page_content for doc in docs]))

    return _mmr_select(
        candidates=docs,
        relevance_scores=relevance_scores,
        embeddings=doc_embeddings,
        k=settings.top_k,
        lambda_mult=settings.mmr_lambda,
    )


def load_llm():
    """Load the tokenizer + generation model per config.py's settings.

    Module-level so eval scripts that need occasional LLM calls (e.g.
    testset generation) can reuse this exact loading path — including
    the generation_config quirk-fix below — without duplicating it and
    risking the two copies drifting apart.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        settings.llm_model_id,
        clean_up_tokenization_spaces=False,
    )
    # Newer transformers versions accept `dtype=` instead and print a
    # deprecation notice for `torch_dtype=` — intentionally not switched:
    # requirements.txt only guarantees transformers>=4.45.0, and
    # from_pretrained has historically been permissive about unrecognized
    # kwargs (often silently absorbed rather than raising). An older
    # supported version that doesn't yet recognize `dtype=` could end up
    # silently loading in default fp32 instead of bf16, with no error to
    # indicate anything went wrong. A cosmetic warning is a far smaller
    # cost than that silent regression, so this stays as torch_dtype=
    # until the requirements.txt floor is raised past whatever version
    # introduced dtype=.
    model = AutoModelForCausalLM.from_pretrained(
        settings.llm_model_id,
        device_map=settings.device_map,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    if not settings.do_sample:
        # The checkpoint's generation_config.json ships with sampling
        # defaults (temperature/top_p) baked in. They're meaningless
        # under greedy decoding and transformers warns about them on
        # every call unless we clear them here.
        model.generation_config.temperature = None
        model.generation_config.top_p = None
    return tokenizer, model


def generate_once(tokenizer, model, messages: list, max_new_tokens: int) -> str:
    """Non-streaming single-turn generation. Shared by RAGPipeline._chat's
    non-stream path and by eval scripts that need occasional LLM calls
    without the streaming machinery."""
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=settings.do_sample
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


class RAGPipeline:
    """Wraps every resource the graph depends on (vector store, BM25
    index, tokenizer/model, cross-encoder) plus the compiled graph
    itself, so they're all built once and reused across queries."""

    def __init__(self):
        # Reused for MMR's diversity term AND for the vector store below
        # (same model that built the Chroma index, so chunk embeddings
        # are directly comparable) — built once and passed to
        # get_vectorstore so the embedding model isn't loaded twice.
        self.embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model_id)
        self.vectorstore = get_vectorstore(embeddings=self.embeddings)

        chunks = load_cached_chunks()
        self.bm25_retriever = BM25Retriever.from_documents(chunks)
        self.bm25_retriever.k = settings.sparse_top_k

        self.tokenizer, self.model = load_llm()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cross_encoder = CrossEncoder(settings.cross_encoder_model_id, device=device)

        self.app = self._build_graph()

        # Running conversation for this session. RAGPipeline owns this
        # rather than the caller, so main.py doesn't need to thread it
        # through manually — call .reset_history() to start fresh.
        self.chat_history: List[ChatTurn] = []

    # -- low-level chat helper --------------------------------------------
    def _chat(self, messages: list, max_new_tokens: int, stream: bool = False) -> str:
        """Run one turn of chat through the local model.

        With stream=True, tokens are printed to stdout as they're
        generated (via a background thread + TextIteratorStreamer) and
        the full text is also returned once generation finishes.
        """
        if not stream:
            return generate_once(self.tokenizer, self.model, messages, max_new_tokens)

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=settings.do_sample,
        )

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        gen_kwargs["streamer"] = streamer

        # model.generate runs in a background thread so we can read tokens
        # off `streamer` as they're produced. Without this wrapper, an
        # exception in that thread (OOM, bad kwarg, etc.) would just kill
        # the thread silently — the main loop would see the streamer end
        # early and return a truncated answer with no indication anything
        # went wrong.
        error_box: list = []

        def _run_generate():
            try:
                self.model.generate(**gen_kwargs)
            except Exception as exc:  # noqa: BLE001 - re-raised below
                error_box.append(exc)

        thread = threading.Thread(target=_run_generate)
        thread.start()

        collected = []
        for token_text in streamer:
            print(token_text, end="", flush=True)
            collected.append(token_text)
        thread.join()
        print()  # newline once the stream finishes

        if error_box:
            raise error_box[0]

        return "".join(collected).strip()

    # -- nodes -----------------------------------------------------------
    def _rewrite_query(self, state: RAGState) -> dict:
        if not settings.rewrite_query:
            return {"rewritten_query": state["question"]}

        messages = [{"role": "system", "content": REWRITE_SYSTEM_PROMPT}]
        messages.extend(_history_to_messages(state["chat_history"]))
        messages.append({"role": "user", "content": state["question"]})

        rewritten = self._chat(
            messages, max_new_tokens=settings.rewrite_max_new_tokens, stream=False
        )
        # Guard against a degenerate/empty rewrite — fall back to the
        # original question rather than searching with garbage.
        if not rewritten or len(rewritten) > 300:
            rewritten = state["question"]
        return {"rewritten_query": rewritten}

    def _retrieve_dense(self, state: RAGState) -> dict:
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": settings.dense_top_k}
        )
        docs = retriever.invoke(state["rewritten_query"])
        return {"dense_docs": docs}

    def _retrieve_sparse(self, state: RAGState) -> dict:
        docs = self.bm25_retriever.invoke(state["rewritten_query"])
        return {"sparse_docs": docs}

    def _fuse_rrf(self, state: RAGState) -> dict:
        return {"fused_docs": fuse_rrf(state["dense_docs"], state["sparse_docs"])}

    def _rerank(self, state: RAGState) -> dict:
        documents = rerank_and_diversify(
            state["rewritten_query"], state["fused_docs"], self.cross_encoder, self.embeddings
        )
        return {"documents": documents}

    def _generate(self, state: RAGState) -> dict:
        context = format_context(state["documents"])
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(_history_to_messages(state["chat_history"]))
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question: {state['question']}\n\n"
                    "Answer using the context above."
                ),
            }
        )
        answer = self._chat(
            messages,
            max_new_tokens=settings.max_new_tokens,
            stream=settings.stream_to_stdout,
        )
        return {"answer": answer}

    # -- graph -------------------------------------------------------------
    def _build_graph(self):
        g = StateGraph(RAGState)

        g.add_node("rewrite_query", self._rewrite_query)
        g.add_node("retrieve_dense", self._retrieve_dense)
        g.add_node("retrieve_sparse", self._retrieve_sparse)
        g.add_node("fuse_rrf", self._fuse_rrf)
        g.add_node("rerank", self._rerank)
        g.add_node("generate", self._generate)

        g.add_edge(START, "rewrite_query")
        # Dense and sparse retrieval both depend only on the rewritten
        # query, so they run as independent branches that both feed fusion.
        # LangGraph waits for every incoming edge before running a node
        # with multiple predecessors, so fuse_rrf runs once both are done.
        g.add_edge("rewrite_query", "retrieve_dense")
        g.add_edge("rewrite_query", "retrieve_sparse")
        g.add_edge("retrieve_dense", "fuse_rrf")
        g.add_edge("retrieve_sparse", "fuse_rrf")
        g.add_edge("fuse_rrf", "rerank")
        g.add_edge("rerank", "generate")
        g.add_edge("generate", END)

        return g.compile()

    def query(self, question: str) -> RAGState:
        result = self.app.invoke(
            {
                "question": question,
                "chat_history": self.chat_history,
                "rewritten_query": "",
                "dense_docs": [],
                "sparse_docs": [],
                "fused_docs": [],
                "documents": [],
                "answer": "",
            }
        )
        self.chat_history.append({"question": question, "answer": result["answer"]})
        # Keep the stored history bounded too, not just what's fed into
        # any single prompt, so it doesn't grow unbounded over a long
        # session.
        self.chat_history = self.chat_history[-settings.max_history_turns :]
        return result

    def reset_history(self) -> None:
        """Clear the conversation so the next query starts fresh, with
        no prior turns influencing rewriting or generation."""
        self.chat_history = []
