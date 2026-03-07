"""
Document loading: converts raw files into LangChain Document objects.

A LangChain Document has two fields:
  - page_content (str): the raw text
  - metadata (dict): arbitrary key/value info attached to that text (source file,
    page number, etc.) — this flows through chunking and into ChromaDB so you
    always know which file and page a retrieved chunk came from.
"""

import logging
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_core.documents import Document

from src.config import DOCUMENTS_DIR

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Loads documents from disk into LangChain Document objects."""

    def load_pdf(self, path: str | Path) -> list[Document]:
        """Load a single PDF. Returns one Document per page."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file, got: {path.suffix}")

        try:
            loader = PyPDFLoader(str(path))
            docs = loader.load()
        except Exception as e:
            logger.error(f"Failed to parse PDF '{path.name}': {e}")
            raise RuntimeError(f"Could not read PDF '{path.name}': {e}") from e

        # Normalise metadata: ensure 'source' is just the filename, not a full path.
        for doc in docs:
            doc.metadata["source"] = path.name
            doc.metadata["file_type"] = "pdf"

        return docs

    def load_directory(
        self,
        dir_path: str | Path | None = None,
        glob: str = "**/*.pdf",
    ) -> list[Document]:
        """Load all PDFs under dir_path (defaults to config.DOCUMENTS_DIR)."""
        dir_path = Path(dir_path) if dir_path else DOCUMENTS_DIR

        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        try:
            loader = DirectoryLoader(
                str(dir_path),
                glob=glob,
                loader_cls=PyPDFLoader,
                show_progress=True,
            )
            docs = loader.load()
        except Exception as e:
            logger.error(f"Failed to load documents from '{dir_path}': {e}")
            raise RuntimeError(f"Could not load documents from '{dir_path}': {e}") from e

        for doc in docs:
            source_path = Path(doc.metadata.get("source", ""))
            doc.metadata["source"] = source_path.name
            doc.metadata["file_type"] = "pdf"

        return docs
