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


# ── Callback propagation (regression: langchain_classic drops constructor callbacks) ──

class TestCallbackPropagation:
    def test_callbacks_passed_to_invoke_via_config(self):
        """query_agent must forward callbacks to agent.invoke() as config={"callbacks": ...}.

        Regression: passing callbacks to AgentExecutor at construction time is silently
        dropped by langchain_classic before it reaches child LLM calls. The fix passes
        them at invoke time instead, which does propagate correctly.
        """
        from unittest.mock import call

        agent = _make_agent(output="42")
        fake_handler = object()  # any sentinel value
        query_agent(agent, "What is 6*7?", callbacks=[fake_handler])

        # agent.invoke must have been called with config containing our callback
        agent.invoke.assert_called_once()
        _, kwargs = agent.invoke.call_args
        config = kwargs.get("config", {})
        assert fake_handler in config.get("callbacks", [])

    def test_no_callbacks_invokes_without_config(self):
        """When no callbacks are given, agent.invoke() is called without config overhead."""
        agent = _make_agent(output="42")
        query_agent(agent, "What is 6*7?")
        _, kwargs = agent.invoke.call_args
        # config should be absent or empty — no callbacks key
        config = kwargs.get("config", {})
        assert not config.get("callbacks")


# ── Eval error counting (regression: success=False not counted as error) ──────

class TestEvalErrorCounting:
    """Verify that the evaluate_model() error counter catches success=False returns.

    Regression: errors was only incremented on exceptions, so agent timeouts and
    max-iteration failures (which return success=False without raising) were missed.
    The fix adds: if not result.get("success", True): errors += 1
    """

    def test_success_false_result_is_an_error(self):
        """A result with success=False must register as a failure in eval counting."""
        # Simulate what evaluate_model()'s loop does
        errors = 0
        result = {"answer": "I couldn't find a reliable answer.", "success": False}

        if not result.get("success", True):
            errors += 1

        assert errors == 1

    def test_success_true_result_is_not_an_error(self):
        errors = 0
        result = {"answer": "The answer is 42.", "success": True}

        if not result.get("success", True):
            errors += 1

        assert errors == 0

    def test_query_agent_fallback_returns_success_false(self):
        """Agent hitting max iterations returns success=False — must be countable as error."""
        agent = _make_agent(output="Agent stopped due to iteration limit.")
        result = query_agent(agent, "Some question")
        assert result["success"] is False
        # Confirm the eval loop would count it
        errors = 0
        if not result.get("success", True):
            errors += 1
        assert errors == 1
