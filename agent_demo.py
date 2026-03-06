"""
Phase 2 demo: tool-calling agent with 12 diverse test queries.

Run with:
    uv run python agent_demo.py

Prerequisites:
  - .env at project root with ANTHROPIC_API_KEY and TAVILY_API_KEY
  - At least one PDF ingested (run main.py first)

The demo runs all 12 queries sequentially and prints the agent's tool calls
(via verbose=True in AgentExecutor) plus the final answer for each.
"""

from src.agent.agent import build_agent

QUERIES = [
    # ── Document Q&A — search_documents ─────────────────────────────────────
    {
        "label": "Q1 · Doc Q&A: General content",
        "query": "What are the main topics covered in the ingested documents?",
        "expected": ["search_documents"],
    },
    {
        "label": "Q2 · Doc Q&A: Specific lookup",
        "query": "What payment terms or due dates are described in the documents?",
        "expected": ["search_documents"],
    },
    {
        "label": "Q3 · Doc Q&A: Summarization",
        "query": "Summarize any procedures or step-by-step instructions mentioned in the documents.",
        "expected": ["search_documents"],
    },
    # ── Arithmetic — calculate ───────────────────────────────────────────────
    {
        "label": "Q4 · Calculator: Exponentiation",
        "query": "What is 2 to the power of 16?",
        "expected": ["calculate"],
    },
    {
        "label": "Q5 · Calculator: Financial total with tax",
        "query": "A contract is worth $45,000 and includes 8.5% sales tax. What is the total amount?",
        "expected": ["calculate"],
    },
    {
        "label": "Q6 · Calculator: Hourly billing",
        "query": "A vendor charges $125 per hour. A project takes 3 weeks at 40 hours per week. What is the total cost?",
        "expected": ["calculate"],
    },
    # ── Web search — web_search ──────────────────────────────────────────────
    {
        "label": "Q7 · Web: Current events",
        "query": "What are the latest developments in AI document processing in 2025?",
        "expected": ["web_search"],
    },
    {
        "label": "Q8 · Web: Public fact",
        "query": "What is the current federal corporate tax rate in the United States?",
        "expected": ["web_search"],
    },
    # ── Entity extraction — extract_entities ─────────────────────────────────
    {
        "label": "Q9 · Entity extraction: Inline text",
        "query": (
            "Extract all entities from this text: "
            "'On March 15, 2024, Acme Corp signed a $2.5M service agreement with "
            "TechPartners Inc., represented by CEO Sarah Johnson, for cloud services "
            "to be delivered in Austin, Texas by Q4 2024.'"
        ),
        "expected": ["extract_entities"],
    },
    # ── Multi-step: combine tools ────────────────────────────────────────────
    {
        "label": "Q10 · Multi-step: Doc search → calculate",
        "query": (
            "Search the documents for any monetary amounts mentioned. "
            "Then calculate what 15% of the largest amount would be."
        ),
        "expected": ["search_documents", "calculate"],
    },
    {
        "label": "Q11 · Multi-step: Doc search → extract entities",
        "query": (
            "Find the most relevant document passage about any agreement or contract, "
            "then extract all named entities from that passage."
        ),
        "expected": ["search_documents", "extract_entities"],
    },
    {
        "label": "Q12 · Multi-step: Web search → calculate",
        "query": (
            "Look up the current EUR to USD exchange rate, "
            "then calculate how much €75,000 is in US dollars."
        ),
        "expected": ["web_search", "calculate"],
    },
]


def run_demo() -> None:
    print("\n" + "=" * 70)
    print("Document Intelligence Agent: Tool Calling Demo")
    print("=" * 70)

    agent = build_agent()

    for item in QUERIES:
        print(f"\n{'─' * 70}")
        print(f"  {item['label']}")
        print(f"  Expected tools: {item['expected']}")
        print(f"  Query: {item['query']}")
        print(f"{'─' * 70}")

        try:
            result = agent.invoke({"input": item["query"]})
            print(f"\n  ANSWER: {result['output']}")
        except Exception as e:
            print(f"\n  ERROR: {e}")


if __name__ == "__main__":
    run_demo()
