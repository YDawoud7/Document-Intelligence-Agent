"""
Document management CLI for the ChromaDB vector store.

Usage:
    uv run python manage.py list
    uv run python manage.py add data/documents/invoice.pdf
    uv run python manage.py add data/documents/          ← whole directory
    uv run python manage.py remove invoice.pdf

Commands:
  list    — Show all ingested documents and their chunk counts.
  add     — Ingest a PDF file or directory. If the file is already in the store,
             its old chunks are removed first so you never get duplicates.
  remove  — Delete all chunks for a given filename from the store.
"""

import argparse
import sys
from pathlib import Path

from src.ingestion.chunker import DocumentChunker
from src.ingestion.loader import DocumentLoader
from src.vectorstore.chroma_store import ChromaStore


def cmd_list(store: ChromaStore) -> None:
    sources = store.list_sources()
    if not sources:
        print("Vector store is empty. Run 'manage.py add <path>' to ingest a document.")
        return
    print(f"\n{'File':<40} {'Chunks':>6}")
    print("─" * 48)
    for src, count in sorted(sources.items()):
        print(f"{src:<40} {count:>6}")
    print("─" * 48)
    print(f"{'Total':<40} {sum(sources.values()):>6}\n")


def cmd_add(store: ChromaStore, path: str) -> None:
    loader = DocumentLoader()
    chunker = DocumentChunker()
    target = Path(path)

    if not target.exists():
        print(f"Error: '{path}' does not exist.")
        sys.exit(1)

    if target.is_dir():
        docs = loader.load_directory(target)
    elif target.suffix.lower() == ".pdf":
        docs = loader.load_pdf(target)
    else:
        print(f"Error: only PDF files are supported (got '{target.suffix}').")
        sys.exit(1)

    if not docs:
        print("No documents loaded — directory may be empty.")
        return

    # Collect unique source filenames in this batch and remove stale chunks first.
    sources_in_batch = {doc.metadata.get("source", target.name) for doc in docs}
    existing = store.list_sources()
    for src in sources_in_batch:
        if src in existing:
            removed = store.delete_document(src)
            print(f"  Replaced {removed} existing chunk(s) for '{src}'")

    chunks = chunker.chunk(docs)
    store.add_documents(chunks)

    # Summary
    from collections import Counter
    per_source = Counter(c.metadata.get("source", "?") for c in chunks)
    for src, n in sorted(per_source.items()):
        print(f"  Added {n} chunk(s) from '{src}'")
    print(f"\nDone. Collection now holds {store.count()} chunk(s) total.")


def cmd_remove(store: ChromaStore, name: str) -> None:
    removed = store.delete_document(name)
    if removed == 0:
        existing = store.list_sources()
        if existing:
            print(f"'{name}' not found. Stored documents:")
            for src in sorted(existing):
                print(f"  {src}")
        else:
            print("Vector store is empty.")
    else:
        print(f"Removed {removed} chunk(s) for '{name}'.")
        print(f"Collection now holds {store.count()} chunk(s) total.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manage documents in the ChromaDB vector store."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List all ingested documents and chunk counts.")

    p_add = sub.add_parser("add", help="Ingest a PDF file or directory.")
    p_add.add_argument("path", help="Path to a PDF file or a directory of PDFs.")

    p_remove = sub.add_parser("remove", help="Remove a document by filename.")
    p_remove.add_argument("name", help="Filename to remove (e.g. invoice.pdf).")

    args = parser.parse_args()
    store = ChromaStore()

    if args.command == "list":
        cmd_list(store)
    elif args.command == "add":
        cmd_add(store, args.path)
    elif args.command == "remove":
        cmd_remove(store, args.name)


if __name__ == "__main__":
    main()
