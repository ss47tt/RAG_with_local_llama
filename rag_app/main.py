"""
Interactive CLI for the RAG app.

Prerequisite: run ingestion once first so there is something to
retrieve from:
    python -m rag_app.ingest ./my_docs

Then chat:
    python -m rag_app.main
"""

try:
    from .config import settings
    from .graph import RAGPipeline
except ImportError:  # running as a plain script, e.g. `python main.py`
    from config import settings
    from graph import RAGPipeline


def main():
    print("Loading vector store and Llama-3.2-3B-Instruct (this can take a while on first run)...")
    pipeline = RAGPipeline()
    print("Ready. Type a question, 'reset' to clear conversation memory, or 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if question.lower() in {"reset", "clear"}:
            pipeline.reset_history()
            print("(conversation memory cleared)\n")
            continue
        if not question:
            continue

        try:
            if settings.stream_to_stdout:
                # The generate node prints tokens to stdout as they're
                # produced, so print the prefix first and let it stream
                # straight onto this line.
                print("\nAssistant: ", end="", flush=True)
                result = pipeline.query(question)
            else:
                result = pipeline.query(question)
                print("\nAssistant:", result["answer"])
        except Exception as exc:  # noqa: BLE001 - keep the session alive
            print(f"\n[error] that turn failed: {exc}\n")
            continue

        print("\nSources used:")
        for i, doc in enumerate(result["documents"], start=1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page")
            preview = doc.page_content[:160].replace("\n", " ").strip()
            if page is not None:
                # PyMuPDF pages are 0-indexed; show the human page number.
                print(f"  [{i}] {source} (page {page + 1})")
            else:
                print(f"  [{i}] {source}")
            print(f"      \"{preview}...\"")
        print()


if __name__ == "__main__":
    main()
