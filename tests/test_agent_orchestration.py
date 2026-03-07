"""Tests for query_agent() guardrail cascade with mocked agent.

Uses unittest.mock to simulate agent.invoke() responses — no real API calls.
"""

from unittest.mock import MagicMock

from src.agent.agent import query_agent


def _make_agent(output, intermediate_steps=None):
    """Create a mock agent that returns a predetermined response."""
    agent = MagicMock()
    agent.invoke.return_value = {
        "output": output,
        "intermediate_steps": intermediate_steps or [],
    }
    return agent


def _make_search_step(observation):
    """Create a fake (AgentAction, observation) tuple for search_documents."""
    action = MagicMock()
    action.tool = "search_documents"
    return (action, observation)


def _make_calc_step(observation):
    action = MagicMock()
    action.tool = "calculate"
    return (action, observation)


# ── Fallback on failure ──────────────────────────────────────────────────────

class TestFallbackOnFailure:
    def test_agent_exception_returns_error(self):
        agent = MagicMock()
        agent.invoke.side_effect = RuntimeError("API down")
        result = query_agent(agent, "What is in the contract?")
        assert result["success"] is False
        assert "technical error" in result["answer"].lower()

    def test_empty_output_triggers_fallback(self):
        agent = _make_agent(output="")
        result = query_agent(agent, "What is in the contract?")
        assert result["success"] is False
        assert "couldn't find" in result["answer"].lower()

    def test_agent_stopped_triggers_fallback(self):
        agent = _make_agent(output="Agent stopped due to iteration limit or time limit.")
        result = query_agent(agent, "What is in the contract?")
        assert result["success"] is False

    def test_fallback_includes_snippets_when_available(self):
        steps = [_make_search_step("The contract says payment is net 30 days.")]
        agent = _make_agent(output="", intermediate_steps=steps)
        result = query_agent(agent, "What are the payment terms?")
        assert result["success"] is False
        assert "net 30 days" in result["answer"]
        assert len(result["snippets"]) == 1


# ── Low-confidence detection ─────────────────────────────────────────────────

class TestLowConfidenceDetection:
    def test_refusal_triggers_warning(self):
        steps = [_make_search_step("The contract mentions payment terms.")]
        agent = _make_agent(
            output="I don't know the exact payment terms from the documents.",
            intermediate_steps=steps,
        )
        result = query_agent(agent, "What are the payment terms?")
        assert result["warning"] is not None
        assert "low confidence" in result["warning"].lower()

    def test_confident_answer_no_warning(self):
        steps = [_make_calc_step("42")]
        agent = _make_agent(
            output="The result of the calculation is 42.",
            intermediate_steps=steps,
        )
        result = query_agent(agent, "What is 6 * 7?")
        assert result["success"] is True
        assert result["warning"] is None


# ── Grounding check integration ──────────────────────────────────────────────

class TestGroundingIntegration:
    def test_grounded_answer_no_warning(self):
        retrieved = "The contract states payment terms are net 30 days from invoice date."
        steps = [_make_search_step(retrieved)]
        agent = _make_agent(
            output="The payment terms are net 30 days from the invoice date.",
            intermediate_steps=steps,
        )
        result = query_agent(agent, "What are the payment terms?")
        assert result["success"] is True
        # Should not have grounding warning
        if result["warning"]:
            assert "not fully supported" not in result["warning"].lower()

    def test_ungrounded_answer_gets_warning(self):
        retrieved = "The contract covers cloud computing services in Austin, Texas."
        steps = [_make_search_step(retrieved)]
        agent = _make_agent(
            output=(
                "The company reported quarterly earnings of fifty million dollars "
                "with significant growth in European markets during the fiscal year "
                "and expanded operations across multiple continents."
            ),
            intermediate_steps=steps,
        )
        result = query_agent(agent, "What were the earnings?")
        assert result["warning"] is not None


# ── Success path ─────────────────────────────────────────────────────────────

class TestSuccessPath:
    def test_normal_answer_succeeds(self):
        agent = _make_agent(output="2 to the power of 16 is 65536.")
        result = query_agent(agent, "What is 2^16?")
        assert result["success"] is True
        assert "65536" in result["answer"]
        assert result["warning"] is None

    def test_result_structure(self):
        agent = _make_agent(output="The answer is 42.")
        result = query_agent(agent, "What is the answer?")
        assert "answer" in result
        assert "snippets" in result
        assert "warning" in result
        assert "success" in result
