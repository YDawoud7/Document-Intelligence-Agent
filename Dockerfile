FROM python:3.13-slim

WORKDIR /app

# System deps needed by sentence-transformers / tokenizers build
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency files first so this layer is cached until deps change
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Pre-download the embedding model at build time (~80 MB).
# This avoids a slow cold-start download and keeps the container self-contained.
RUN uv run python -c \
    "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy application source
COPY src/ src/
COPY manage.py query.py ./

# Volume mount points for persistent data.
# Operators mount their document directory and ChromaDB here:
#   docker run -v /host/docs:/data/documents -v /host/db:/data/chroma_db ...
VOLUME ["/data/documents", "/data/chroma_db"]

ENV DOCUMENTS_DIR=/data/documents
ENV CHROMA_PERSIST_DIR=/data/chroma_db

EXPOSE 8000

# REST API (default). Override CMD for CLI use:
#   docker run ... doc-agent uv run python query.py
#   docker run ... doc-agent uv run python manage.py list
CMD ["uv", "run", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
