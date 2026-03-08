"""Pydantic request/response models for the Document Intelligence API."""

from typing import Literal

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., description="The question to ask the agent.")
    model: Literal["haiku", "gpt4o", "deepseek"] = Field(
        default="haiku",
        description='Model to use: "haiku" (default), "gpt4o", or "deepseek".',
    )


class QueryResponse(BaseModel):
    answer: str
    success: bool
    warning: str | None = None
    snippets: list[str] = []
    latency_s: float
    tokens: int | None = None
    cost_usd: float | None = None


class DocumentInfo(BaseModel):
    filename: str
    chunks: int


class DocumentListResponse(BaseModel):
    documents: dict[str, int] = Field(
        description="Mapping of filename → chunk count."
    )
    total_chunks: int


class DocumentUploadResponse(BaseModel):
    filename: str
    chunks_added: int
    total_chunks: int


class DocumentDeleteResponse(BaseModel):
    filename: str
    chunks_removed: int
    total_chunks: int


class DocumentClearResponse(BaseModel):
    documents_removed: int
    chunks_removed: int


class HealthResponse(BaseModel):
    status: str = Field(description='"ok" or "degraded"')
    documents_loaded: int
    models_available: list[str]
