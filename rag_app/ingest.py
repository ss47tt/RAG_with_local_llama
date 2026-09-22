"""
Ingestion: load documents from a folder, split them, embed them with a
sentence-transformers model, and persist them into a local Chroma
vector store.

Usage:
    python -m rag_app.ingest /path/to/docs
"""

import argparse
import hashlib
import pickle
import shutil
import sys
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyMuPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

try:
    from .config import settings
except ImportError:  # running as a plain script rather than a package
    from config import settings

# Adding ". " as a separator (ahead of the default's plain " ") means the
# splitter prefers to break at a sentence boundary before falling back to
# breaking mid-sentence, which cuts down on chunks that end mid-thought.
TEXT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

LOADER_MAP = {
    ".txt": lambda path: TextLoader(path, autodetect_encoding=True),
    # sort=True asks PyMuPDF to reorder text spans into reading order,
    # which matters a lot for 2-column academic PDFs (without it, text
    # from both columns gets interleaved line-by-line into garbage).
    # mode="page" (the default) keeps each page as its own Document, so
    # the splitter below never merges text across a page boundary.
    ".pdf": lambda path: PyMuPDFLoader(path, sort=True),
}

# Markdown headers to split on, in order. Headers are stripped into
# metadata (h1/h2/...) by the splitter below, then explicitly
# re-attached as readable text to every resulting chunk in
# load_and_chunk_markdown — see that function for why it's done that
# way instead of just leaving headers inline in the source text.
MARKDOWN_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")]


def load_documents(source_dir: str):
    """Load every .txt/.pdf file under source_dir. Markdown is handled
    separately by load_and_chunk_markdown, since it needs header-aware
    splitting rather than the generic loader+splitter path."""
    docs = []
    for suffix, loader_cls in LOADER_MAP.items():
        loader = DirectoryLoader(
            source_dir,
            glob=f"**/*{suffix}",
            loader_cls=loader_cls,
            show_progress=True,
            use_multithreading=True,
            # Without this, DirectoryLoader aborts the ENTIRE suffix batch
            # the first time any single file fails to load — one corrupt
            # PDF or oddly-encoded .txt file would silently drop every
            # other file of that type from ingestion. silent_errors=True
            # logs the failure and keeps going with the rest.
            silent_errors=True,
        )
        try:
            docs.extend(loader.load())
        except Exception as exc:  # noqa: BLE001 - surface but keep going
            print(f"[ingest] skipped some {suffix} files: {exc}", file=sys.stderr)
    return docs


def _header_prefix(metadata: dict) -> str:
    """Turn {'h1': 'Title', 'h2': 'Section A'} into 'Title > Section A'."""
    parts = [metadata[key] for key in ("h1", "h2", "h3", "h4") if key in metadata]
    return " > ".join(parts)


def load_and_chunk_markdown(source_dir: str) -> list:
    """Header-aware chunking for markdown files.

    A plain character splitter doesn't know a markdown file has
    structure, so it can (and does) cut a chunk right through the
    middle of a section, losing the heading that gave it context. This
    instead: (1) splits each file along its header hierarchy first, so
    a section stays together, then (2) explicitly re-attaches that
    section's header path to every resulting chunk's *text* — including
    each piece of a section too big to keep whole — so a chunk is
    self-contained no matter where a fallback character split happens
    to land. (Just keeping headers inline in the original text and
    letting the character splitter fall through isn't enough: it can —
    and did, in testing — carve the header line off into its own
    near-empty chunk, orphaning it from the body it was labeling.)
    """
    # strip_headers=True: headers come back as metadata (h1/h2/...)
    # rather than inline text, so we control exactly how and where
    # they're re-attached below, instead of leaving it to chance.
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=MARKDOWN_HEADERS, strip_headers=True
    )
    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=TEXT_SEPARATORS,
    )

    chunks = []
    md_paths = sorted(Path(source_dir).rglob("*.md"))
    for path in md_paths:
        text = path.read_text(encoding="utf-8", errors="ignore")
        sections = header_splitter.split_text(text)

        for section in sections:
            section.metadata["source"] = str(path)
            prefix = _header_prefix(section.metadata)
            full_content = f"{prefix}\n\n{section.page_content}" if prefix else section.page_content

            if len(full_content) <= settings.chunk_size:
                section.page_content = full_content
                chunks.append(section)
            else:
                # Split the body ALONE (no header text in it yet), then
                # prepend the header prefix to every resulting piece —
                # guarantees each sub-chunk is self-contained, rather
                # than hoping the character splitter happens to keep
                # header and body together.
                for body_piece in size_splitter.split_text(section.page_content):
                    piece_content = f"{prefix}\n\n{body_piece}" if prefix else body_piece
                    chunks.append(
                        Document(page_content=piece_content, metadata=dict(section.metadata))
                    )

    if md_paths:
        print(f"[ingest] header-split {len(md_paths)} markdown files into {len(chunks)} chunks")
    return chunks


def build_vectorstore(source_dir: str) -> Chroma:
    # Wipe any existing store first. Without this, running ingest.py more
    # than once against the same persist_dir keeps *appending* to the
    # collection instead of replacing it — every chunk gets duplicated,
    # duplicates dominate similarity search, and retrieval quality
    # collapses (the same handful of chunks win every query).
    persist_path = Path(settings.persist_dir)
    if persist_path.exists():
        shutil.rmtree(persist_path)
        print(f"[ingest] cleared existing store at '{persist_path}'")

    docs = load_documents(source_dir)
    md_chunks = load_and_chunk_markdown(source_dir)
    if not docs and not md_chunks:
        raise ValueError(
            f"No .txt/.md/.pdf documents found under '{source_dir}'. "
            "Add some files and try again."
        )
    if docs:
        print(f"[ingest] loaded {len(docs)} raw .txt/.pdf documents")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=TEXT_SEPARATORS,
    )
    generic_chunks = splitter.split_documents(docs) if docs else []
    print(f"[ingest] split .txt/.pdf into {len(generic_chunks)} chunks")

    chunks = generic_chunks + md_chunks
    print(f"[ingest] {len(chunks)} chunks total")

    # Give every chunk a stable, content-derived id. Hybrid retrieval
    # (dense + BM25) needs a shared identifier to know that "chunk #7 from
    # the dense search" and "chunk #7 from BM25" are the same chunk, so
    # Reciprocal Rank Fusion can merge the two ranked lists correctly.
    chunk_ids = []
    for chunk in chunks:
        source = chunk.metadata.get("source", "")
        page = chunk.metadata.get("page", "")
        digest = hashlib.sha1(chunk.page_content.encode("utf-8")).hexdigest()[:12]
        chunk_id = f"{source}::{page}::{digest}"
        chunk.metadata["chunk_id"] = chunk_id
        chunk_ids.append(chunk_id)

    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model_id)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=settings.collection_name,
        persist_directory=settings.persist_dir,
        ids=chunk_ids,
    )
    print(f"[ingest] persisted vector store to '{settings.persist_dir}'")

    # BM25 needs the raw chunk text (it's a sparse/lexical index, not an
    # embedding index), so stash the chunks themselves alongside Chroma.
    cache_path = Path(settings.chunks_cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(chunks, f)
    print(f"[ingest] cached {len(chunks)} raw chunks for BM25 at '{cache_path}'")

    return vectorstore


def get_vectorstore(embeddings: HuggingFaceEmbeddings | None = None) -> Chroma:
    """Open an already-built vector store (for querying) without re-ingesting.

    Pass an existing HuggingFaceEmbeddings instance if the caller already
    has one (e.g. RAGPipeline builds one for MMR's diversity term) —
    otherwise this builds its own, which means the embedding model gets
    loaded into memory twice for no reason.
    """
    if embeddings is None:
        embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model_id)
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings,
        persist_directory=settings.persist_dir,
    )


def load_cached_chunks() -> list:
    """Load the raw chunked Documents cached during ingest, for BM25."""
    cache_path = Path(settings.chunks_cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"No cached chunks found at '{cache_path}'. Run "
            "`python -m rag_app.ingest <folder>` first."
        )
    with open(cache_path, "rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG vector store.")
    parser.add_argument("source_dir", type=str, help="Folder containing .txt/.md/.pdf files")
    args = parser.parse_args()

    if not Path(args.source_dir).is_dir():
        raise SystemExit(f"'{args.source_dir}' is not a directory")

    build_vectorstore(args.source_dir)


if __name__ == "__main__":
    main()
