"""Token tracking callback handler for LangChain agents.

HOW IT WORKS:
  LangChain fires on_llm_end after every LLM call. The response contains
  AIMessage objects with usage_metadata (populated by both langchain-anthropic
  and langchain-openai). This handler accumulates input/output token counts
  across the entire agent loop — including inner LLM calls like the one in
  extract_entities.

USAGE:
  handler = TokenTrackingHandler()
  agent = build_agent(provider="anthropic", callbacks=[handler])
  query_agent(agent, "What is 2+2?")
  print(handler.total_tokens, handler.estimate_cost("claude-haiku-4-5-20251001"))
  handler.reset()  # before next query
"""

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from src.config import MODEL_PRICING


class TokenTrackingHandler(BaseCallbackHandler):
    """Accumulates token usage across all LLM calls in an agent invocation."""

    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.llm_calls = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """Extract token counts from usage_metadata on each generation."""
        for gen_list in response.generations:
            for gen in gen_list:
                msg = getattr(gen, "message", None)
                if msg is None:
                    continue
                usage = getattr(msg, "usage_metadata", None)
                if usage:
                    self.input_tokens += usage.get("input_tokens", 0)
                    self.output_tokens += usage.get("output_tokens", 0)
        self.llm_calls += 1

    def estimate_cost(self, model_name: str) -> float:
        """Estimate cost in USD based on hardcoded per-model pricing."""
        pricing = MODEL_PRICING.get(model_name, {"input": 0, "output": 0})
        return (
            self.input_tokens * pricing["input"]
            + self.output_tokens * pricing["output"]
        ) / 1_000_000

    def reset(self) -> None:
        """Reset counters for the next query."""
        self.input_tokens = 0
        self.output_tokens = 0
        self.llm_calls = 0
