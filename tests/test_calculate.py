"""Tests for the AST-based safe calculator tool.

All tests are deterministic — no API calls, no mocks.
"""

import pytest

from src.agent.tools import calculate


class TestBasicArithmetic:
    def test_addition(self):
        assert calculate.invoke({"expression": "1 + 1"}) == "2"

    def test_subtraction(self):
        assert calculate.invoke({"expression": "10 - 3"}) == "7"

    def test_multiplication(self):
        assert calculate.invoke({"expression": "6 * 7"}) == "42"

    def test_division(self):
        result = float(calculate.invoke({"expression": "10 / 3"}))
        assert abs(result - 3.3333333333) < 0.0001

    def test_floor_division(self):
        assert calculate.invoke({"expression": "10 // 3"}) == "3"

    def test_modulo(self):
        assert calculate.invoke({"expression": "10 % 3"}) == "1"

    def test_exponentiation(self):
        assert calculate.invoke({"expression": "2 ** 16"}) == "65536"


class TestOperatorPrecedence:
    def test_multiply_before_add(self):
        assert calculate.invoke({"expression": "2 + 3 * 4"}) == "14"

    def test_parentheses(self):
        assert calculate.invoke({"expression": "(2 + 3) * 4"}) == "20"

    def test_nested_parentheses(self):
        assert calculate.invoke({"expression": "((1 + 2) * (3 + 4))"}) == "21"


class TestUnaryOperators:
    def test_negation(self):
        assert calculate.invoke({"expression": "-5 + 3"}) == "-2"

    def test_positive(self):
        assert calculate.invoke({"expression": "+5"}) == "5"


class TestFinancialCalculations:
    def test_tax_calculation(self):
        result = float(calculate.invoke({"expression": "45000 * 1.085"}))
        assert result == 48825.0

    def test_hourly_billing(self):
        assert calculate.invoke({"expression": "125 * 3 * 40"}) == "15000"


class TestErrorHandling:
    def test_division_by_zero(self):
        result = calculate.invoke({"expression": "1 / 0"})
        assert "division by zero" in result.lower()

    def test_invalid_syntax(self):
        result = calculate.invoke({"expression": "hello world"})
        assert result.startswith("Error")

    def test_expression_too_long(self):
        result = calculate.invoke({"expression": "1+" * 2000})
        assert "too long" in result.lower()

    def test_empty_expression(self):
        result = calculate.invoke({"expression": ""})
        assert result.startswith("Error")


class TestSafety:
    """Verify that dangerous constructs are rejected."""

    def test_rejects_function_calls(self):
        result = calculate.invoke({"expression": "__import__('os')"})
        assert result.startswith("Error")

    def test_rejects_attribute_access(self):
        result = calculate.invoke({"expression": "(1).__class__"})
        assert result.startswith("Error")

    def test_rejects_list_comprehension(self):
        result = calculate.invoke({"expression": "[x for x in range(10)]"})
        assert result.startswith("Error")

    def test_rejects_lambda(self):
        result = calculate.invoke({"expression": "lambda: 1"})
        assert result.startswith("Error")
