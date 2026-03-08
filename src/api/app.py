"""
FastAPI application for the Document Intelligence Agent.

Endpoints:
    GET  /health                  — system status
    GET  /documents               — list ingested documents
    POST /documents               — upload + ingest a PDF
    DELETE /documents/{filename}  — remove a document
    POST /query                   — query the agent

Startup behaviour (lifespan):
    - Initializes a shared ChromaStore (points at CHROMA_PERSIST_DIR)
    - Builds one AgentExecutor per model whose API key is present in the environment
    - Missing API keys cause that model to be skipped with a warning (graceful degradation)
    - The embedding model (~80 MB) is loaded once and reused across all requests
"""

import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile

load_dotenv()

from src.agent.agent import build_agent, query_agent
from src.api.models import (
    DocumentClearResponse,
    DocumentDeleteResponse,
    DocumentListResponse,
    DocumentUploadResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
)
from src.config import DOCUMENTS_DIR, MAX_UPLOAD_SIZE_BYTES, SUPPORTED_MODELS, configure_logging
from src.ingestion.chunker import DocumentChunker
from src.ingestion.loader import DocumentLoader
from src.observability.token_tracker import TokenTrackingHandler
from src.vectorstore.chroma_store import ChromaStore

configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build agents and shared store once at startup; clean up on shutdown."""
    logger.info("Starting up Document Intelligence Agent API...")

    app.state.store = ChromaStore()
    logger.info(f"ChromaDB ready — {app.state.store.count()} chunks loaded.")

    # Auto-ingest any PDFs in DOCUMENTS_DIR that aren't already in the store.
    # Skips files already indexed so restarts are safe and fast.
    if DOCUMENTS_DIR.exists():
        existing = app.state.store.list_sources()
        loader, chunker = DocumentLoader(), DocumentChunker()
        for pdf in sorted(DOCUMENTS_DIR.glob("*.pdf")):
            if pdf.name in existing:
                logger.info(f"Skipping '{pdf.name}' (already indexed).")
                continue
            try:
                docs = loader.load_pdf(pdf)
                chunks = chunker.chunk(docs)
                app.state.store.add_documents(chunks)
                logger.info(f"Auto-ingested '{pdf.name}': {len(chunks)} chunks.")
            except Exception as e:
                logger.error(f"Failed to ingest '{pdf.name}': {e}")

    app.state.agents: dict = {}
    for key, config in SUPPORTED_MODELS.items():
        env_key = config["api_key_env"]
        if not os.environ.get(env_key):
            logger.warning(f"Skipping model '{key}': {env_key} not set.")
            continue
        try:
            app.state.agents[key] = build_agent(
                provider=config["provider"],
                model_name=config["model_name"],
                store=app.state.store,
            )
            logger.info(f"Agent ready: {key} ({config['model_name']})")
        except Exception as e:
            logger.error(f"Failed to build agent '{key}': {e}")

    if not app.state.agents:
        logger.error("No agents available — check your API keys in .env")

    yield

    logger.info("Shutting down.")


app = FastAPI(
    title="Document Intelligence Agent",
    description=(
        "A production-grade RAG agent that answers questions about your documents. "
        "Upload PDFs, then query them in natural language. "
        "Supports Claude Haiku, GPT-4o, and DeepSeek."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """Check system status: ChromaDB reachability and available models."""
    try:
        doc_count = app.state.store.count()
        store_ok = True
    except Exception:
        doc_count = 0
        store_ok = False

    status = "ok" if (store_ok and app.state.agents) else "degraded"

    return HealthResponse(
        status=status,
        documents_loaded=doc_count,
        models_available=list(app.state.agents.keys()),
    )


# ── Documents ─────────────────────────────────────────────────────────────────

@app.get("/documents", response_model=DocumentListResponse, tags=["Documents"])
def list_documents():
    """List all ingested documents with their chunk counts."""
    sources = app.state.store.list_sources()
    return DocumentListResponse(
        documents=sources,
        total_chunks=sum(sources.values()),
    )


@app.post("/documents", response_model=DocumentUploadResponse, status_code=201, tags=["Documents"])
async def upload_document(file: UploadFile = File(...)):
    """Upload and ingest a PDF. Re-uploading an existing file replaces it."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(contents) > MAX_UPLOAD_SIZE_BYTES:
        mb = MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)
        raise HTTPException(status_code=413, detail=f"File too large (max {mb} MB).")

    # Write to a temp file so DocumentLoader (which expects a path) can read it
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        loader = DocumentLoader()
        chunker = DocumentChunker()
        docs = loader.load_pdf(tmp_path)

        if not docs:
            raise HTTPException(status_code=422, detail="Could not extract text from PDF.")

        # Override source metadata so it reflects the original filename
        for doc in docs:
            doc.metadata["source"] = file.filename

        # Remove stale chunks for this file before re-adding
        existing = app.state.store.list_sources()
        if file.filename in existing:
            app.state.store.delete_document(file.filename)

        chunks = chunker.chunk(docs)
        app.state.store.add_documents(chunks)
        logger.info(f"Ingested '{file.filename}': {len(chunks)} chunks")
    finally:
        os.unlink(tmp_path)

    return DocumentUploadResponse(
        filename=file.filename,
        chunks_added=len(chunks),
        total_chunks=app.state.store.count(),
    )


@app.delete("/documents", response_model=DocumentClearResponse, tags=["Documents"])
def clear_all_documents():
    """Remove every document and all their chunks from the vector store."""
    sources = app.state.store.list_sources()
    total_chunks = 0
    for filename in list(sources.keys()):
        total_chunks += app.state.store.delete_document(filename)
    logger.info(f"Cleared all documents: {len(sources)} files, {total_chunks} chunks removed")
    return DocumentClearResponse(
        documents_removed=len(sources),
        chunks_removed=total_chunks,
    )


@app.delete("/documents/{filename}", response_model=DocumentDeleteResponse, tags=["Documents"])
def delete_document(filename: str):
    """Remove all chunks for a document by filename."""
    removed = app.state.store.delete_document(filename)
    if removed == 0:
        raise HTTPException(
            status_code=404,
            detail=f"'{filename}' not found. Use GET /documents to list available files.",
        )
    logger.info(f"Deleted '{filename}': {removed} chunks removed")
    return DocumentDeleteResponse(
        filename=filename,
        chunks_removed=removed,
        total_chunks=app.state.store.count(),
    )


# ── Query ─────────────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse, tags=["Agent"])
def query(request: QueryRequest):
    """
    Query the agent with a natural language question.

    The agent autonomously selects from four tools:
    - **search_documents** — searches ingested PDFs via semantic retrieval
    - **calculate** — evaluates arithmetic expressions safely
    - **web_search** — fetches live information from the internet
    - **extract_entities** — extracts structured fields from text passages

    Returns the answer plus metadata: confidence warnings, retrieved snippets,
    token usage, and estimated cost.
    """
    agent = app.state.agents.get(request.model)
    if agent is None:
        available = list(app.state.agents.keys())
        raise HTTPException(
            status_code=404,
            detail=f"Model '{request.model}' is not available. Available: {available}",
        )

    tracker = TokenTrackingHandler()
    start = time.time()
    result = query_agent(agent, request.query, callbacks=[tracker])
    latency = round(time.time() - start, 2)

    model_name = SUPPORTED_MODELS[request.model]["model_name"]
    tokens = tracker.total_tokens or None
    cost = round(tracker.estimate_cost(model_name), 6) if tokens else None

    return QueryResponse(
        answer=result["answer"],
        success=result["success"],
        warning=result.get("warning"),
        snippets=result.get("snippets", []),
        latency_s=latency,
        tokens=tokens,
        cost_usd=cost,
    )
