from pathlib import Path

# Project root
ROOT_DIR = Path(__file__).parent.parent

# Directories
DOCUMENTS_DIR = ROOT_DIR / "data" / "documents"
CHROMA_PERSIST_DIR = ROOT_DIR / "data" / "chroma_db"

# ChromaDB
COLLECTION_NAME = "enterprise_docs"

# Chunking
# CHUNK_SIZE: max characters per chunk. ~1000 chars ≈ ~200 tokens — fits comfortably in
# a retrieval context without being so large that unrelated content pollutes the result.
CHUNK_SIZE = 1000
# CHUNK_OVERLAP: characters shared between consecutive chunks so a sentence split across
# a boundary still appears in full in at least one chunk.
CHUNK_OVERLAP = 200

# Embeddings
# all-MiniLM-L6-v2: 384-dimensional, fast, ~80MB download, strong semantic quality.
# Runs locally — no API key required.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Agent LLM
# Haiku is fast and cheap — ideal for agentic loops where multiple LLM calls happen
# per query (tool routing + synthesis). Swap for claude-sonnet-4-6 for higher quality.
CLAUDE_MODEL = "claude-haiku-4-5-20251001"

# ── Guardrail limits ─────────────────────────────────────────────────────────
# These caps stop bad input before it reaches expensive APIs.
MAX_QUERY_LENGTH = 10_000          # max chars for agent queries and tool searches
MAX_EXPRESSION_LENGTH = 1_000      # max chars for calculator expressions
MAX_EXTRACT_TEXT_LENGTH = 50_000   # max chars for entity extraction input

# ── Logging ──────────────────────────────────────────────────────────────────
# config.py is imported first by everything, so this runs early.
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
