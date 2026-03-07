"""
Four LangChain tools for the Document Intelligence Agent.

Each tool is a Python function decorated with @tool. LangChain passes the
function's docstring to Claude as the tool description — Claude reads ONLY the
docstring (not the code) to decide when to call each tool. Clear, specific
descriptions are the most important design decision here.

Tool routing summary:
  search_documents  → any question about document contents
  calculate         → any arithmetic
  web_search        → current events / facts not in documents
  extract_entities  → parse structured fields from a text passage

GUARDRAILS (Phase 3):
  - Every tool validates input length before calling external APIs.
  - Every tool body is wrapped in try/except → returns a user-friendly error string.
  - API calls (Anthropic, Tavily) use tenacity retry with exponential backoff
    so transient network/rate-limit failures are retried automatically.
  - extract_entities checks for prompt injection patterns since user text is
    embedded directly in the LLM prompt.
"""

import ast
import logging
import operator
import re

from anthropic import APIConnectionError, RateLimitError
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import (
    CLAUDE_MODEL,
    MAX_EXPRESSION_LENGTH,
    MAX_EXTRACT_TEXT_LENGTH,
    MAX_QUERY_LENGTH,
)
from src.vectorstore.chroma_store import ChromaStore

logger = logging.getLogger(__name__)

# ── Retry decorator for API calls ────────────────────────────────────────────
# Exponential backoff (2s → 4s → 8s). Only retries transient errors — network
# issues, rate limits, server 5xx. Never retries 4xx client errors (bad API key,
# invalid input). After 3 failures, reraise=True lets the exception propagate to
# the tool's try/except which returns a user-friendly error string.

_api_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(
        (APIConnectionError, RateLimitError, ConnectionError, TimeoutError)
    ),
    before_sleep=lambda info: logger.warning(
        f"Retrying API call (attempt {info.attempt_number})..."
    ),
    reraise=True,
)

# ── Injection detection for extract_entities ─────────────────────────────────
# This is a heuristic tripwire, not a bulletproof defense. Real prompt injection
# defense comes from architecture (separating data from instructions), but pattern
# detection catches the most common attack templates.

_INJECTION_PATTERNS = re.compile(
    r"(ignore\s+(all\s+)?previous\s+instructions"
    r"|you\s+are\s+now"
    r"|system\s*:\s*"
    r"|<\s*/?\s*system\s*>)",
    re.IGNORECASE,
)

# ── Tool 1: Document RAG search ──────────────────────────────────────────────

# Lazy singleton so the HuggingFace model (~80 MB) only loads on first tool call.
_store: ChromaStore | None = None


def _get_store() -> ChromaStore:
    global _store
    if _store is None:
        _store = ChromaStore()
    return _store


@tool
def search_documents(query: str) -> str:
    """Search the enterprise document knowledge base for information relevant to
    the query. Use this tool when answering questions about document contents,
    policies, contracts, invoices, reports, or any internally stored text.
    Returns the top matching passages with their source file and page number."""
    if len(query) > MAX_QUERY_LENGTH:
        return f"Error: query too long ({len(query)} chars, max {MAX_QUERY_LENGTH})."

    try:
        store = _get_store()
        results = store.similarity_search(query, k=4)
        if not results:
            return "No relevant documents found for this query."
        parts: list[str] = []
        for i, doc in enumerate(results, start=1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "?")
            snippet = doc.page_content.strip()
            parts.append(f"[{i}] Source: {source}, Page: {page}\n{snippet}")
        return "\n\n---\n\n".join(parts)
    except Exception as e:
        logger.error(f"search_documents failed: {e}")
        return f"Error searching documents: {type(e).__name__}. Please try again."


# ── Tool 2: Safe calculator ──────────────────────────────────────────────────

# Whitelisted AST node types — anything else is rejected before evaluation.
# This means raw eval() is never called on user-controlled strings.
_SAFE_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Num,  # kept for Python 3.13 compatibility (alias of Constant)
    ast.Add, ast.Sub, ast.Mult, ast.Div,
    ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd,
)

_OPS: dict[type, object] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp):
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression: {type(node).__name__}")


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression and return the numeric result.
    Supports: +, -, *, /, // (floor division), % (modulo), ** (exponentiation),
    and unary negation. Use this tool for any arithmetic computation such as
    totals, percentages, unit conversions, or financial calculations.
    Example inputs: '(1500 * 0.085) + 1500', '2 ** 10', '125 * 3 * 40'."""
    if len(expression) > MAX_EXPRESSION_LENGTH:
        return f"Error: expression too long ({len(expression)} chars, max {MAX_EXPRESSION_LENGTH})."
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        for node in ast.walk(tree):
            if not isinstance(node, _SAFE_NODES):
                raise ValueError(f"Disallowed: {type(node).__name__}")
        result = _safe_eval(tree)
        return str(int(result)) if result == int(result) else str(round(result, 10))
    except ZeroDivisionError:
        return "Error: division by zero"
    except (ValueError, KeyError) as e:
        return f"Error: {e}"
    except SyntaxError:
        return f"Error: invalid expression '{expression}'"
    except RecursionError:
        return "Error: expression too deeply nested"


# ── Tool 3: Web search ───────────────────────────────────────────────────────

@tool
def web_search(query: str) -> str:
    """Search the live web for current information, news, or facts not present
    in the document knowledge base. Use this tool when answering questions about
    recent events, public data, company information, regulatory updates, or any
    topic requiring up-to-date information from the internet.
    Returns a summary of the top search results."""
    if len(query) > MAX_QUERY_LENGTH:
        return f"Error: query too long ({len(query)} chars, max {MAX_QUERY_LENGTH})."

    try:
        tavily = TavilySearch(max_results=3)

        @_api_retry
        def _invoke():
            return tavily.invoke({"query": query})

        response = _invoke()
        results = response.get("results", [])
        if not results:
            return "No web results found."
        parts: list[str] = []
        for i, r in enumerate(results, start=1):
            title = r.get("title", "No title")
            url = r.get("url", "")
            content = r.get("content", "").strip()[:500]
            parts.append(f"[{i}] {title}\n{url}\n{content}")
        return "\n\n---\n\n".join(parts)
    except Exception as e:
        logger.error(f"web_search failed: {e}")
        return f"Error performing web search: {type(e).__name__}. Please try again."


# ── Tool 4: Structured entity extraction ─────────────────────────────────────

class DocumentEntities(BaseModel):
    """Structured fields extracted from document text."""
    people: list[str] = Field(default_factory=list, description="Full names of people mentioned")
    organizations: list[str] = Field(default_factory=list, description="Company or institution names")
    dates: list[str] = Field(default_factory=list, description="Dates or time periods mentioned")
    monetary_amounts: list[str] = Field(default_factory=list, description="Dollar amounts or financial figures")
    locations: list[str] = Field(default_factory=list, description="Cities, states, countries, or addresses")
    key_terms: list[str] = Field(default_factory=list, description="Domain-specific terms or identifiers")


@tool
def extract_entities(text: str) -> str:
    """Extract structured named entities and key fields from a passage of document text.
    Returns a structured breakdown of: people, organizations, dates, monetary amounts,
    locations, and key terms found in the text. Use this tool when you need to identify
    and organize specific information from retrieved document passages, such as parsing
    contract parties, invoice details, or report metadata."""
    if len(text) > MAX_EXTRACT_TEXT_LENGTH:
        return f"Error: text too long ({len(text)} chars, max {MAX_EXTRACT_TEXT_LENGTH})."
    if len(text.strip()) < 10:
        return "Error: text too short for meaningful entity extraction."

    # Injection detection — the text is embedded in a prompt, so check for
    # manipulation attempts. This is a heuristic tripwire, not bulletproof.
    if _INJECTION_PATTERNS.search(text):
        logger.warning("Possible prompt injection detected in extract_entities input")
        return "Error: input text contains patterns that could interfere with processing."

    try:
        llm = ChatAnthropic(model=CLAUDE_MODEL)
        structured_llm = llm.with_structured_output(DocumentEntities)

        # XML tag delimiters reduce injection surface — Claude is trained to
        # treat <document> tags as data boundaries rather than instructions.
        prompt = (
            "Extract all named entities and key structured fields from the text "
            "between the <document> tags. Be thorough and include every instance. "
            "Only extract entities that are explicitly stated in the text.\n\n"
            f"<document>\n{text}\n</document>"
        )

        @_api_retry
        def _invoke():
            return structured_llm.invoke(prompt)

        result: DocumentEntities = _invoke()
        lines: list[str] = ["=== Extracted Entities ==="]
        if result.people:
            lines.append(f"People: {', '.join(result.people)}")
        if result.organizations:
            lines.append(f"Organizations: {', '.join(result.organizations)}")
        if result.dates:
            lines.append(f"Dates: {', '.join(result.dates)}")
        if result.monetary_amounts:
            lines.append(f"Monetary Amounts: {', '.join(result.monetary_amounts)}")
        if result.locations:
            lines.append(f"Locations: {', '.join(result.locations)}")
        if result.key_terms:
            lines.append(f"Key Terms: {', '.join(result.key_terms)}")
        if len(lines) == 1:
            return "No entities found — text may be too short or non-informational."
        return "\n".join(lines)
    except Exception as e:
        logger.error(f"extract_entities failed: {e}")
        return f"Error extracting entities: {type(e).__name__}. Please try again."


# Exported registry — agent.py imports this list
TOOLS = [search_documents, calculate, web_search, extract_entities]
