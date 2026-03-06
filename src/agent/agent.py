"""
Agent construction: wires the four tools + Claude into a LangChain agent.

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
"""

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

from src.agent.tools import TOOLS
from src.config import CLAUDE_MODEL

load_dotenv()  # reads ANTHROPIC_API_KEY and TAVILY_API_KEY from .env

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
    )
