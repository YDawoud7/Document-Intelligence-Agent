"""
Vector store: embeds document chunks and stores them in ChromaDB for retrieval.

HOW RAG RETRIEVAL WORKS (end-to-end):
  1. At index time: each chunk is converted to an embedding vector and stored.
  2. At query time: the question is embedded the same way, then ChromaDB finds
     the k chunks whose vectors are closest (by cosine similarity) to the question
     vector. Those chunks become the "context" passed to the LLM.

EMBEDDINGS (HuggingFaceEmbeddings / sentence-transformers):
  The model maps any text → a fixed-size list of floats (384 for MiniLM-L6).
  Texts with similar meaning land near each other in that 384-dimensional space,
  which is why "invoice total" and "amount due" retrieve the same chunk even though
  they share no words.

CHROMADB PERSISTENCE:
  Passing a persist_directory means ChromaDB writes its SQLite + vector index to
  disk automatically. Re-creating ChromaStore pointing at the same directory loads
  the existing collection — no re-ingestion needed across runs.
"""

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import (
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
)


class ChromaStore:
    """Wraps ChromaDB with sentence-transformer embeddings."""

    def __init__(
        self,
        collection_name: str = COLLECTION_NAME,
        persist_directory: str | None = None,
        embedding_model: str = EMBEDDING_MODEL,
    ) -> None:
        persist_directory = persist_directory or str(CHROMA_PERSIST_DIR)

        # HuggingFaceEmbeddings downloads the model on first use (~80 MB) and
        # caches it locally. Subsequent runs are instant.
        self._embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

        self._store = Chroma(
            collection_name=collection_name,
            embedding_function=self._embeddings,
            persist_directory=persist_directory,
        )

    def add_documents(self, chunks: list[Document]) -> None:
        """Embed chunks and upsert them into ChromaDB."""
        self._store.add_documents(chunks)

    def similarity_search(self, query: str, k: int = 5) -> list[Document]:
        """Return the k most semantically similar chunks to query."""
        return self._store.similarity_search(query, k=k)

    def similarity_search_with_score(
        self, query: str, k: int = 5
    ) -> list[tuple[Document, float]]:
        """Like similarity_search but also returns the cosine distance score.

        Score interpretation: lower = more similar (L2 / Euclidean distance).
          ~0.0 = nearly identical meaning, >1.5 = likely unrelated.
        Practically, scores below ~1.0 indicate a strong semantic match.
        """
        return self._store.similarity_search_with_score(query, k=k)

    def count(self) -> int:
        """Return the number of chunks currently stored in the collection."""
        return self._store._collection.count()
