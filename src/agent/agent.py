"""
Agent construction and query execution with production guardrails.

How it works:
  1. ChatAnthropic is the reasoning LLM — it decides which tool to call.
  2. create_tool_calling_agent binds the tools to Claude using its native
     tool-use API (structured JSON calls, not fragile text parsing).
  3. AgentExecutor runs the tool-call loop: invoke tool → feed result back
     to Claude → repeat until Claude emits a final answer.

The prompt has three parts:
  - system: tells Claude its role and what each tool category is for
  - human: the user's question (passed as {input})
  - placeholder agent_scratchpad: where LangChain injects tool call history
    so Claude can see what it has already done each iteration

GUARDRAILS (Phase 3):
  query_agent() is the recommended entry point. It wraps agent.invoke() with:
    1. Input validation — rejects empty or oversized queries before any API call
    2. Error handling — catches exceptions and returns a safe fallback message
    3. Fallback strategy — if the agent fails or gives up, returns retrieved
       document snippets so the user still gets something useful
    4. Low-confidence detection — catches refusal patterns ("I don't know")
       and appends raw snippets for the user to judge
    5. Grounding check — heuristic that flags answers not supported by the
       retrieved documents (catches gross hallucinations)
"""

import logging
import re

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

from src.agent.tools import TOOLS
from src.config import CLAUDE_MODEL, MAX_QUERY_LENGTH

load_dotenv()  # reads ANTHROPIC_API_KEY and TAVILY_API_KEY from .env

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an Enterprise Document Intelligence Assistant with access \
to four tools:

1. search_documents — Search the internal document knowledge base (PDFs, contracts, \
invoices, reports). Always try this first for any question about document contents.
2. calculate — Safe arithmetic evaluator. Use for any numerical computation.
3. web_search — Live internet search. Use for current events, public facts, or \
information not in the documents.
4. extract_entities — Structured entity extraction. Use when you need to parse \
people, organizations, dates, amounts, or locations from a text passage.

For complex questions, use multiple tools in sequence and synthesize the results. \
Always cite the source file and page when referencing document content."""


def build_agent() -> AgentExecutor:
    """Construct and return a ready-to-invoke AgentExecutor.

    Reads ANTHROPIC_API_KEY and TAVILY_API_KEY from the environment (via .env).
    """
    llm = ChatAnthropic(
        model=CLAUDE_MODEL,
        temperature=0,      # deterministic routing — same query, same tool choice
        max_tokens=2048,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
        # LangChain injects the tool call/result history here each iteration.
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm=llm, tools=TOOLS, prompt=prompt)

    return AgentExecutor(
        agent=agent,
        tools=TOOLS,
        verbose=True,               # prints tool calls and intermediate steps
        max_iterations=6,           # safety cap against infinite tool loops
        handle_parsing_errors=True, # feeds malformed tool calls back to Claude to retry
        return_intermediate_steps=True,  # needed for output guardrails
    )


# ── Output guardrails ────────────────────────────────────────────────────────

# Refusal / low-confidence patterns. When the agent says one of these, we treat
# it as "low confidence" and append any retrieved document snippets so the user
# can judge for themselves.
_REFUSAL_PATTERNS = re.compile(
    r"(I don't know|I'm not sure|I cannot|I couldn't find|"
    r"I'm unable to|no information|cannot determine)",
    re.IGNORECASE,
)


def _check_grounding(answer: str, intermediate_steps: list) -> str | None:
    """Check if the agent's answer is grounded in retrieved document chunks.

    HOW IT WORKS (simple heuristic, not bulletproof):
      1. Find all search_documents tool calls in the intermediate steps.
      2. Collect the retrieved text from those calls.
      3. Extract key phrases from the answer (3-word sliding window).
      4. Check how many key phrases appear in the retrieved text.
      5. If fewer than 15% overlap → flag as potentially ungrounded.

    WHY 15%?
      The LLM legitimately paraphrases, so exact string matching is lossy.
      A low threshold avoids false positives while catching the worst case:
      the LLM completely fabricates an answer with zero connection to the
      retrieved documents.

    Returns a warning string if ungrounded, None if grounded or not applicable.
    """
    # Collect retrieved document text from search_documents calls
    retrieved_text = ""
    used_search = False
    for step in intermediate_steps:
        action, observation = step
        if action.tool == "search_documents":
            used_search = True
            retrieved_text += " " + observation

    # Only check grounding for queries that actually used document search
    if not used_search or not retrieved_text.strip():
        return None
    if "No relevant documents found" in retrieved_text:
        return None

    # Extract key phrases from the answer (3-word sliding window)
    words = answer.split()
    if len(words) < 6:
        return None  # too short to meaningfully check

    key_phrases = []
    for i in range(len(words) - 2):
        phrase = " ".join(words[i:i + 3]).lower()
        # Skip phrases that are mostly stop words
        if not re.search(r"[a-z]{4,}", phrase):
            continue
        key_phrases.append(phrase)

    if not key_phrases:
        return None

    # Check overlap
    retrieved_lower = retrieved_text.lower()
    grounded_count = sum(1 for p in key_phrases if p in retrieved_lower)
    grounding_ratio = grounded_count / len(key_phrases)

    if grounding_ratio < 0.15:
        logger.warning(
            f"Low grounding score: {grounding_ratio:.2f} "
            f"({grounded_count}/{len(key_phrases)} phrases)"
        )
        return (
            "**Note:** This answer may not be fully supported by the retrieved "
            "documents. Please verify the claims against the source material."
        )
    return None


# ── Main entry point with full guardrails ─────────────────────────────────────

def query_agent(agent: AgentExecutor, query: str) -> dict:
    """Run a query through the agent with full guardrails.

    Returns a dict with:
      - "answer": the final answer string (may include warnings)
      - "snippets": list of retrieved document snippets (if any)
      - "warning": optional warning string
      - "success": bool

    Guardrail cascade:
      1. Input validation — reject empty / oversized queries
      2. Agent execution — wrapped in try/except
      3. Fallback on failure — return retrieved snippets if the agent crashes
      4. Low-confidence detection — append snippets on refusal patterns
      5. Grounding check — prepend warning if answer is ungrounded
    """
    # ── 1. Input validation ───────────────────────────────────────────────
    if not query or not query.strip():
        return {
            "answer": "Please provide a question.",
            "snippets": [], "warning": None, "success": False,
        }

    query = query.strip()
    if len(query) > MAX_QUERY_LENGTH:
        return {
            "answer": (
                f"Query too long ({len(query)} chars, max {MAX_QUERY_LENGTH}). "
                f"Please shorten your question."
            ),
            "snippets": [], "warning": None, "success": False,
        }

    # ── 2. Run the agent ──────────────────────────────────────────────────
    try:
        result = agent.invoke({"input": query})
    except Exception as e:
        logger.error(f"Agent execution failed: {e}")
        return {
            "answer": "I couldn't process your question due to a technical error. Please try again.",
            "snippets": [],
            "warning": f"Error: {type(e).__name__}",
            "success": False,
        }

    answer = result.get("output", "").strip()
    intermediate_steps = result.get("intermediate_steps", [])

    # ── 3. Collect retrieved snippets (for fallback display) ──────────────
    snippets = []
    for step in intermediate_steps:
        action, observation = step
        if action.tool == "search_documents" and "No relevant documents" not in observation:
            snippets.append(observation)

    # ── 4. Fallback: empty or max-iterations-exceeded output ──────────────
    if not answer or answer.lower().startswith("agent stopped"):
        fallback_msg = "I couldn't find a reliable answer to your question."
        if snippets:
            fallback_msg += "\n\nHere are the most relevant document passages I found:\n\n"
            fallback_msg += "\n\n---\n\n".join(snippets[:3])
        return {
            "answer": fallback_msg,
            "snippets": snippets,
            "warning": "Agent did not produce a final answer",
            "success": False,
        }

    # ── 5. Low-confidence detection ───────────────────────────────────────
    warning = None
    if _REFUSAL_PATTERNS.search(answer):
        warning = "Low confidence answer detected"
        if snippets:
            answer += "\n\nRelevant passages from your documents:\n\n"
            answer += "\n\n---\n\n".join(snippets[:3])

    # ── 6. Grounding check ────────────────────────────────────────────────
    grounding_warning = _check_grounding(answer, intermediate_steps)
    if grounding_warning:
        answer = grounding_warning + "\n\n" + answer
        warning = warning or grounding_warning

    return {
        "answer": answer,
        "snippets": snippets,
        "warning": warning,
        "success": True,
    }
