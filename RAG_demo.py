"""
Phase 1 demo: ingest → chunk → embed → retrieve.

Run with:
    uv run python main.py

Drop any PDF into data/documents/ before running. The script ingests all PDFs
in that folder, stores their chunks in ChromaDB, then runs a few example queries
so you can see retrieval in action.

On re-runs the ChromaDB collection persists on disk, so chunks accumulate.
Delete data/chroma_db/ to start fresh.
"""

from src.config import DOCUMENTS_DIR
from src.ingestion.chunker import DocumentChunker
from src.ingestion.loader import DocumentLoader
from src.vectorstore.chroma_store import ChromaStore


def main() -> None:
    loader = DocumentLoader()
    chunker = DocumentChunker()
    store = ChromaStore()

    # ── 1. Load ──────────────────────────────────────────────────────────────
    print(f"\nLoading PDFs from {DOCUMENTS_DIR} ...")
    docs = loader.load_directory()

    if not docs:
        print(
            "No PDFs found. Drop a PDF into data/documents/ and re-run.\n"
            "Example: cp ~/Downloads/invoice.pdf data/documents/"
        )
        return

    pages_by_source: dict[str, int] = {}
    for doc in docs:
        src = doc.metadata.get("source", "unknown")
        pages_by_source[src] = pages_by_source.get(src, 0) + 1

    for src, pages in pages_by_source.items():
        print(f"  Loaded {pages} page(s) from {src}")

    # ── 2. Chunk ─────────────────────────────────────────────────────────────
    chunks = chunker.chunk(docs)
    stats = chunker.chunk_stats(chunks)
    print(
        f"\nChunking complete: {stats['count']} chunks "
        f"(avg {stats['avg_chars']} chars, "
        f"min {stats['min_chars']}, max {stats['max_chars']})"
    )

    # ── 3. Store ─────────────────────────────────────────────────────────────
    print(f"\nEmbedding and storing chunks in ChromaDB ...")
    store.add_documents(chunks)
    print(f"Collection now holds {store.count()} chunk(s) total.")

    # ── 4. Retrieve ───────────────────────────────────────────────────────────

    print("\n" + "=" * 60)
    print("RETRIEVAL DEMO")
    print("=" * 60)

    query = input("Enter query or press 'Enter' to exit: ").strip()
    while query != "":
        results = store.similarity_search_with_score(query, k=3)

        if not results:
            print("  (no results)")
            continue

        for rank, (doc, score) in enumerate(results, start=1):
            snippet = doc.page_content.replace("\n", " ").strip()[:120]
            source = doc.metadata.get("source", "?")
            page = doc.metadata.get("page", "?")
            print(f"  [{rank}] score={score:.3f}  [{source} p.{page}]")
            print(f"       \"{snippet}...\"")

        print("-" * 60)
        query = input("Enter query or press 'Enter' to exit: ").strip()


if __name__ == "__main__":
    main()
