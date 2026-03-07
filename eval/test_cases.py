"""
20 evaluation test cases for the Document Intelligence Agent.

Each test case has:
  - id: unique identifier (Q01–Q20)
  - category: "routing" | "quality" | "edge"
  - query: the question to ask the agent
  - expected_tools: list of tool names the agent should call
  - expected_contains: list of substrings the answer should contain (case-insensitive)
  - description: human-readable summary

Scoring:
  - Tool routing score: 1.0 if all expected_tools were called, 0.0 otherwise
  - Content score: fraction of expected_contains substrings found in the answer
"""

TEST_CASES = [
    # ── Tool routing (8 queries) ─────────────────────────────────────────────
    {
        "id": "Q01",
        "category": "routing",
        "query": "What are the main topics covered in the ingested documents?",
        "expected_tools": ["search_documents"],
        "expected_contains": [],
        "description": "General document question → search_documents",
    },
    {
        "id": "Q02",
        "category": "routing",
        "query": "What payment terms or due dates are described in the documents?",
        "expected_tools": ["search_documents"],
        "expected_contains": [],
        "description": "Specific document detail → search_documents",
    },
    {
        "id": "Q03",
        "category": "routing",
        "query": "What is 7 * 8 + 12?",
        "expected_tools": ["calculate"],
        "expected_contains": ["68"],
        "description": "Simple arithmetic → calculate",
    },
    {
        "id": "Q04",
        "category": "routing",
        "query": "A project costs $85,000 with 9.5% tax. What is the total?",
        "expected_tools": ["calculate"],
        "expected_contains": ["93075"],
        "description": "Financial calculation → calculate",
    },
    {
        "id": "Q05",
        "category": "routing",
        "query": "What are the latest developments in AI document processing in 2025?",
        "expected_tools": ["web_search"],
        "expected_contains": [],
        "description": "Current events → web_search",
    },
    {
        "id": "Q06",
        "category": "routing",
        "query": "What is the current federal corporate tax rate in the United States?",
        "expected_tools": ["web_search"],
        "expected_contains": ["21"],
        "description": "Public fact → web_search",
    },
    {
        "id": "Q07",
        "category": "routing",
        "query": (
            "Extract all entities from this text: "
            "'On March 15, 2024, Acme Corp signed a $2.5M service agreement with "
            "TechPartners Inc., represented by CEO Sarah Johnson, for cloud services "
            "to be delivered in Austin, Texas by Q4 2024.'"
        ),
        "expected_tools": ["extract_entities"],
        "expected_contains": ["Sarah Johnson", "Acme Corp", "Austin"],
        "description": "Entity extraction from inline text → extract_entities",
    },
    {
        "id": "Q08",
        "category": "routing",
        "query": (
            "Find the most relevant document passage about any agreement or contract, "
            "then extract all named entities from that passage."
        ),
        "expected_tools": ["search_documents", "extract_entities"],
        "expected_contains": [],
        "description": "Multi-step: doc search + entity extraction",
    },
    # ── Answer quality (8 queries) ───────────────────────────────────────────
    {
        "id": "Q09",
        "category": "quality",
        "query": "What is 2 to the power of 16?",
        "expected_tools": ["calculate"],
        "expected_contains": ["65536"],
        "description": "Exponentiation with known result",
    },
    {
        "id": "Q10",
        "category": "quality",
        "query": "Calculate 45000 multiplied by 1.085",
        "expected_tools": ["calculate"],
        "expected_contains": ["48825"],
        "description": "Tax calculation with known result",
    },
    {
        "id": "Q11",
        "category": "quality",
        "query": "A vendor charges $125 per hour. A project takes 3 weeks at 40 hours per week. What is the total cost?",
        "expected_tools": ["calculate"],
        "expected_contains": ["15000"],
        "description": "Hourly billing with known result",
    },
    {
        "id": "Q12",
        "category": "quality",
        "query": (
            "Extract entities from: 'On January 10, 2025, GlobalTech Solutions paid "
            "$1.2M to DataVault Inc. for a 3-year data storage contract. The deal was "
            "negotiated by CFO Michael Chen in San Francisco.'"
        ),
        "expected_tools": ["extract_entities"],
        "expected_contains": ["GlobalTech", "DataVault", "Michael Chen", "San Francisco", "1.2M"],
        "description": "Entity extraction with multiple known entities",
    },
    {
        "id": "Q13",
        "category": "quality",
        "query": "Summarize any procedures or step-by-step instructions mentioned in the documents.",
        "expected_tools": ["search_documents"],
        "expected_contains": [],
        "description": "Document summarization",
    },
    {
        "id": "Q14",
        "category": "quality",
        "query": "What organizations or companies are mentioned in the documents?",
        "expected_tools": ["search_documents"],
        "expected_contains": [],
        "description": "Document entity lookup",
    },
    {
        "id": "Q15",
        "category": "quality",
        "query": (
            "Search the documents for any monetary amounts mentioned. "
            "Then calculate what 15% of the largest amount would be."
        ),
        "expected_tools": ["search_documents", "calculate"],
        "expected_contains": [],
        "description": "Multi-step: doc search + calculate",
    },
    {
        "id": "Q16",
        "category": "quality",
        "query": "What is 999 * 999?",
        "expected_tools": ["calculate"],
        "expected_contains": ["998001"],
        "description": "Large multiplication with known result",
    },
    # ── Edge cases (4 queries) ───────────────────────────────────────────────
    {
        "id": "Q17",
        "category": "edge",
        "query": "What is the company's policy on remote work and flexible hours?",
        "expected_tools": ["search_documents"],
        "expected_contains": [],
        "description": "Question about content likely not in documents",
    },
    {
        "id": "Q18",
        "category": "edge",
        "query": "Tell me about any financial figures in the documents and calculate the sum of all amounts.",
        "expected_tools": ["search_documents"],
        "expected_contains": [],
        "description": "Ambiguous multi-tool question",
    },
    {
        "id": "Q19",
        "category": "edge",
        "query": "What is (2**10 + 3**5) * 7?",
        "expected_tools": ["calculate"],
        "expected_contains": ["8869"],
        "description": "Complex arithmetic expression",
    },
    {
        "id": "Q20",
        "category": "edge",
        "query": "How many words are in the phrase 'the quick brown fox jumps over the lazy dog'? Calculate 100 divided by that number.",
        "expected_tools": ["calculate"],
        "expected_contains": [],
        "description": "Multi-hop reasoning requiring counting + calculation",
    },
]
