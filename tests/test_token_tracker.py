"""Regression tests for TokenTrackingHandler.

Covers the bug where on_llm_end was not accumulating tokens because
usage_metadata was never read (the callback wasn't firing through
langchain_classic AgentExecutor — fixed by passing callbacks at invoke time).
These tests verify the handler itself correctly reads and accumulates tokens
from the LLMResult passed to on_llm_end.
"""

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from src.observability.token_tracker import TokenTrackingHandler


def _make_llm_result(input_tokens: int, output_tokens: int) -> LLMResult:
    """Build a minimal LLMResult with usage_metadata on a ChatGeneration message."""
    msg = AIMessage(
        content="test",
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    )
    return LLMResult(generations=[[ChatGeneration(message=msg)]])


def _make_llm_result_no_usage() -> LLMResult:
    """LLMResult where the generation has no usage_metadata (e.g. cached response)."""
    msg = AIMessage(content="test")  # usage_metadata defaults to None
    return LLMResult(generations=[[ChatGeneration(message=msg)]])


class TestTokenTrackingHandler:
    def test_on_llm_end_accumulates_tokens(self):
        handler = TokenTrackingHandler()
        handler.on_llm_end(_make_llm_result(100, 50))
        assert handler.input_tokens == 100
        assert handler.output_tokens == 50
        assert handler.total_tokens == 150

    def test_on_llm_end_accumulates_across_calls(self):
        handler = TokenTrackingHandler()
        handler.on_llm_end(_make_llm_result(100, 50))
        handler.on_llm_end(_make_llm_result(200, 80))
        assert handler.input_tokens == 300
        assert handler.output_tokens == 130

    def test_on_llm_end_increments_llm_calls(self):
        handler = TokenTrackingHandler()
        handler.on_llm_end(_make_llm_result(10, 10))
        handler.on_llm_end(_make_llm_result(10, 10))
        assert handler.llm_calls == 2

    def test_on_llm_end_no_usage_metadata_is_noop(self):
        """Missing usage_metadata must not raise and must leave counters at zero."""
        handler = TokenTrackingHandler()
        handler.on_llm_end(_make_llm_result_no_usage())
        assert handler.input_tokens == 0
        assert handler.output_tokens == 0
        assert handler.llm_calls == 1  # call was counted even without token data

    def test_reset_clears_all_counters(self):
        handler = TokenTrackingHandler()
        handler.on_llm_end(_make_llm_result(500, 200))
        handler.reset()
        assert handler.input_tokens == 0
        assert handler.output_tokens == 0
        assert handler.llm_calls == 0
        assert handler.total_tokens == 0

    def test_estimate_cost_uses_model_pricing(self):
        handler = TokenTrackingHandler()
        handler.on_llm_end(_make_llm_result(1_000_000, 1_000_000))
        cost = handler.estimate_cost("claude-haiku-4-5-20251001")
        # $0.80/M input + $4.00/M output = $4.80
        assert abs(cost - 4.80) < 0.001

    def test_estimate_cost_unknown_model_is_zero(self):
        handler = TokenTrackingHandler()
        handler.on_llm_end(_make_llm_result(1_000_000, 1_000_000))
        assert handler.estimate_cost("unknown-model-xyz") == 0.0
