"""Tests for DocumentChunker.

Uses LangChain's RecursiveCharacterTextSplitter directly — no mocks, no API calls.
"""

from langchain_core.documents import Document

from src.ingestion.chunker import DocumentChunker


class TestChunking:
    def test_short_document_stays_intact(self, short_document):
        chunker = DocumentChunker(chunk_size=1000, chunk_overlap=200)
        chunks = chunker.chunk([short_document])
        assert len(chunks) == 1
        assert chunks[0].page_content == short_document.page_content

    def test_long_document_splits(self, sample_documents):
        chunker = DocumentChunker(chunk_size=200, chunk_overlap=50)
        chunks = chunker.chunk([sample_documents[0]])
        assert len(chunks) > 1

    def test_chunk_index_metadata(self, sample_documents):
        chunker = DocumentChunker(chunk_size=200, chunk_overlap=50)
        chunks = chunker.chunk(sample_documents)
        for i, chunk in enumerate(chunks):
            assert chunk.metadata["chunk_index"] == i

    def test_metadata_preserved(self, sample_documents):
        chunker = DocumentChunker(chunk_size=200, chunk_overlap=50)
        chunks = chunker.chunk([sample_documents[0]])
        for chunk in chunks:
            assert chunk.metadata["source"] == "contract.pdf"
            assert chunk.metadata["page"] == 0

    def test_empty_input(self):
        chunker = DocumentChunker()
        chunks = chunker.chunk([])
        assert chunks == []

    def test_chunks_respect_max_size(self, sample_documents):
        chunk_size = 200
        chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=50)
        chunks = chunker.chunk(sample_documents)
        for chunk in chunks:
            assert len(chunk.page_content) <= chunk_size + 50  # allow small overshoot from separator


class TestChunkStats:
    def test_stats_structure(self, sample_documents):
        chunker = DocumentChunker(chunk_size=200, chunk_overlap=50)
        chunks = chunker.chunk(sample_documents)
        stats = chunker.chunk_stats(chunks)
        assert "count" in stats
        assert "avg_chars" in stats
        assert "min_chars" in stats
        assert "max_chars" in stats
        assert stats["count"] == len(chunks)

    def test_stats_empty_input(self):
        chunker = DocumentChunker()
        stats = chunker.chunk_stats([])
        assert stats["count"] == 0
        assert stats["avg_chars"] == 0

    def test_min_max_consistency(self, sample_documents):
        chunker = DocumentChunker(chunk_size=200, chunk_overlap=50)
        chunks = chunker.chunk(sample_documents)
        stats = chunker.chunk_stats(chunks)
        assert stats["min_chars"] <= stats["avg_chars"] <= stats["max_chars"]
