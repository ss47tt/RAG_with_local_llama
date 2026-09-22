# Local RAG App — Llama-3.2-3B-Instruct + LangGraph

A minimal, local-first Retrieval-Augmented Generation app:

- **Vector store:** Chroma (persisted on disk)
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
- **Generation model:** `meta-llama/Llama-3.2-3B-Instruct` (run locally via `transformers`)
- **Orchestration:** LangGraph — a 6-node graph: `rewrite_query` → `retrieve_dense` + `retrieve_sparse` → `fuse_rrf` → `rerank` (cross-encoder + MMR) → `generate` (streamed, multi-turn aware)

## Project layout

```
rag_app/
├── requirements.txt
├── README.md
└── rag_app/
    ├── __init__.py
    ├── config.py    # all tunables (models, chunk size, top_k, paths)
    ├── ingest.py     # load docs -> chunk -> embed -> persist to Chroma
    ├── graph.py       # the LangGraph pipeline + local LLM loading
    └── main.py         # interactive CLI
```

## 1. Install

```bash
pip install -r requirements.txt
```

GPU strongly recommended (bf16). CPU works but will be slow.

## 2. Get access to Llama-3.2-3B-Instruct

This is a gated model on the Hugging Face Hub:

1. Request access at https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
2. Log in locally: `huggingface-cli login` (or set the `HF_TOKEN` env var)

## 3. Ingest your documents

Put `.txt`, `.md`, or `.pdf` files in a folder, then:

```bash
python -m rag_app.ingest ./my_docs
```

This chunks the documents (`chunk_size=1200`, `chunk_overlap=200` by
default, see `config.py`) and writes embeddings into `./chroma_db`.

## 4. Chat

```bash
python -m rag_app.main
```

This loads the model once, then answers questions from the terminal,
grounding every answer in the top-`k` retrieved chunks and listing the
sources it used.

## How the graph works

```
START -> rewrite_query -> retrieve_dense  \
                        -> retrieve_sparse  } -> fuse_rrf -> rerank -> generate -> END
```

- **`rewrite_query`**: asks the LLM to turn the raw question into a
  self-contained search query (resolves pronouns, keeps keywords).
  Runs with a short `max_new_tokens` budget since it's not user-facing.
- **`retrieve_dense`**: Chroma similarity search over embeddings — good
  at semantic/paraphrase matches.
- **`retrieve_sparse`**: BM25 keyword search over the same chunks —
  good at exact terms, acronyms, names embeddings sometimes miss.
  These two run as independent branches and both feed into fusion.
- **`fuse_rrf`**: merges the two ranked lists with Reciprocal Rank
  Fusion — chunks that rank well in *both* lists float to the top,
  without needing to normalize dense cosine scores against BM25 scores.
- **`rerank`**: a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
  scores each fused candidate directly against the *original* question,
  which is more accurate than embedding similarity alone. Those scores
  then feed **Maximal Marginal Relevance (MMR)**, which picks the final
  `top_k` balancing relevance against diversity — so if 4 of the top 6
  candidates are near-duplicate paragraphs, MMR won't let them crowd
  out a genuinely different (but still relevant) chunk.
- **`generate`**: Llama answers using the reranked context — plus the
  last `max_history_turns` prior (question, answer) turns, so
  follow-ups like "what about the 70B version?" work — streamed
  token-by-token to stdout as it's generated.

`RAGState` (a `TypedDict`) carries all of this between nodes — add a
field and a node if you want to extend the pipeline further.
`RAGPipeline` owns the running `chat_history` across calls to `.query()`;
call `.reset_history()` to start a fresh conversation (the CLI's
`reset`/`clear` command does this for you).

## Chunking

- **PDFs**: loaded page-by-page (`PyMuPDFLoader` with `mode="page"`),
  so the splitter never merges text across a page boundary — a chunk
  stays within one page's content.
- **Markdown**: split along its header hierarchy first
  (`MarkdownHeaderTextSplitter`, keeping `#`/`##`/`###`/`####` intact
  in each chunk's text), and only falls back to character-splitting a
  section if it's still too big for `chunk_size`. This keeps a section
  together with its heading instead of cutting through the middle of it.
- **All chunkers** prefer sentence boundaries (`". "`) over breaking
  mid-word when a paragraph/line break isn't available.

## Customizing

Everything tunable lives in `config.py`:

- `embedding_model_id` — swap embedding model (must re-run ingest if you change this)
- `chunk_size` / `chunk_overlap` — chunking granularity
- `dense_top_k` / `sparse_top_k` — candidates pulled by each retriever before fusion
- `rrf_k` / `rrf_top_n` — fusion constant and how many fused candidates survive to reranking
- `cross_encoder_model_id` / `top_k` — reranking model and final chunk count sent to the LLM
- `mmr_lambda` — MMR's relevance/diversity tradeoff (1.0 = pure relevance, 0.0 = pure diversity)
- `rewrite_query` / `rewrite_max_new_tokens` — toggle and budget for query rewriting
- `max_history_turns` — how many prior (question, answer) turns feed into rewriting + generation
- `stream_to_stdout` — stream tokens live vs. return the full answer at once
- `do_sample` / `max_new_tokens` — generation behavior
- `persist_dir` / `collection_name` / `chunks_cache_path` — where the Chroma index and BM25 corpus cache live

## Extending further

- **Per-source filters**: pass `filter=...` into `as_retriever` /
  `BM25Retriever` search kwargs if you want to scope retrieval to a
  subset of ingested files.
- **Async/parallel branches**: `retrieve_dense` and `retrieve_sparse`
  are already independent nodes; swapping `app.invoke` for
  `app.ainvoke` with async node functions would let them run
  concurrently instead of sequentially.
