"""
Document chunking: splits large Documents into smaller, overlapping chunks.

WHY CHUNK?
  - LLMs have finite context windows — you can't feed a 50-page contract as context.
  - Smaller chunks improve retrieval precision: a chunk about "payment terms" won't
    be diluted by unrelated contract clauses.

WHY RecursiveCharacterTextSplitter?
  It splits on a priority list of separators: ["\n\n", "\n", " ", ""].
  This means it tries paragraph breaks first, then line breaks, then words, and only
  falls back to hard character splits as a last resort — preserving natural boundaries
  far better than a naive fixed-size split.

WHY OVERLAP?
  If a key sentence falls right at a chunk boundary, overlap ensures it appears in
  full in at least one of the two surrounding chunks.
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_OVERLAP, CHUNK_SIZE


class DocumentChunker:
    """Splits Document objects into smaller overlapping chunks."""

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            # Keep separators in the output so chunk boundaries are readable.
            keep_separator=True,
        )

    def chunk(self, documents: list[Document]) -> list[Document]:
        """Split documents into chunks, preserving and extending metadata."""
        chunks = self._splitter.split_documents(documents)

        # Tag each chunk with its position so we can later reconstruct order if needed.
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i

        return chunks

    def chunk_stats(self, chunks: list[Document]) -> dict:
        """Return a quick summary useful for debugging chunk quality."""
        lengths = [len(c.page_content) for c in chunks]
        return {
            "count": len(chunks),
            "avg_chars": round(sum(lengths) / len(lengths)) if lengths else 0,
            "min_chars": min(lengths) if lengths else 0,
            "max_chars": max(lengths) if lengths else 0,
        }
