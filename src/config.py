import os
from pathlib import Path

# Project root
ROOT_DIR = Path(__file__).parent.parent

# Directories — overridable via environment variables for container deployments.
# In Docker, mount volumes and point these at the mount paths:
#   docker run -e DOCUMENTS_DIR=/data/documents -e CHROMA_PERSIST_DIR=/data/chroma_db ...
DOCUMENTS_DIR = Path(os.environ.get("DOCUMENTS_DIR", str(ROOT_DIR / "data" / "documents")))
CHROMA_PERSIST_DIR = Path(os.environ.get("CHROMA_PERSIST_DIR", str(ROOT_DIR / "data" / "chroma_db")))

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
MAX_UPLOAD_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB — prevents OOM on file upload

# ── Agent tuning ──────────────────────────────────────────────────────────────
SEARCH_RESULTS_K = 4          # top-k chunks returned by similarity search
MAX_AGENT_ITERATIONS = 6      # safety cap on the tool-call loop
GROUNDING_THRESHOLD = 0.15    # min phrase overlap fraction to consider an answer grounded

# ── Logging ──────────────────────────────────────────────────────────────────
import logging


def configure_logging(level: str = "INFO") -> None:
    """Configure the root logger. Call once from each CLI entry point.

    Not called at import time so that frameworks (FastAPI, pytest) can manage
    their own logging setup without conflict.
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

# ── Model configurations ──────────────────────────────────────────────────────
# Each entry maps a shorthand name to its provider, model ID, and API key env var.
# DeepSeek uses the OpenAI-compatible API, so it needs a base_url override.
SUPPORTED_MODELS = {
    "haiku": {
        "provider": "anthropic",
        "model_name": "claude-haiku-4-5-20251001",
        "api_key_env": "ANTHROPIC_API_KEY",
    },
    "gpt4o": {
        "provider": "openai",
        "model_name": "gpt-4o",
        "api_key_env": "OPENAI_API_KEY",
    },
    "deepseek": {
        "provider": "deepseek",
        "model_name": "deepseek-chat",
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com/v1",
    },
}

# $ per 1M tokens — last updated 2025-03
MODEL_PRICING = {
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
    "gpt-4o":                    {"input": 2.50, "output": 10.00},
    "deepseek-chat":             {"input": 0.27, "output": 1.10},
}

# Eval output
EVAL_RESULTS_DIR = ROOT_DIR / "eval_results"
