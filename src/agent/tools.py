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
"""

import ast
import operator

from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field

from src.config import CLAUDE_MODEL
from src.vectorstore.chroma_store import ChromaStore

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


# ── Tool 3: Web search ───────────────────────────────────────────────────────

@tool
def web_search(query: str) -> str:
    """Search the live web for current information, news, or facts not present
    in the document knowledge base. Use this tool when answering questions about
    recent events, public data, company information, regulatory updates, or any
    topic requiring up-to-date information from the internet.
    Returns a summary of the top search results."""
    tavily = TavilySearch(max_results=3)
    results = tavily.invoke({"query": query})
    if not results:
        return "No web results found."
    parts: list[str] = []
    for i, r in enumerate(results, start=1):
        title = r.get("title", "No title")
        url = r.get("url", "")
        content = r.get("content", "").strip()[:500]
        parts.append(f"[{i}] {title}\n{url}\n{content}")
    return "\n\n---\n\n".join(parts)


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
    llm = ChatAnthropic(model=CLAUDE_MODEL)
    structured_llm = llm.with_structured_output(DocumentEntities)
    prompt = (
        "Extract all named entities and key structured fields from the following text. "
        "Be thorough and include every instance.\n\n"
        f"TEXT:\n{text}"
    )
    result: DocumentEntities = structured_llm.invoke(prompt)
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


# Exported registry — agent.py imports this list
TOOLS = [search_documents, calculate, web_search, extract_entities]
