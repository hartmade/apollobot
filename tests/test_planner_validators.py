"""Tests for Pydantic type coercion validators in planner.py."""

import json

import pytest

from apollobot.agents.planner import (
    AnalysisStep,
    DataRequirement,
    ResearchPlan,
    _coerce_dict,
    _coerce_list,
)
from apollobot.agents import LLMProvider


class TestCoerceList:
    """Tests for _coerce_list helper."""

    def test_list_passthrough(self):
        """A list of strings passes through with stringification."""
        assert _coerce_list(["a", "b", "c"]) == ["a", "b", "c"]

    def test_int_stringification(self):
        """List of ints becomes list of strings."""
        assert _coerce_list([1, 2, 3]) == ["1", "2", "3"]

    def test_comma_split(self):
        """Comma-separated string is split."""
        assert _coerce_list("a, b, c") == ["a", "b", "c"]

    def test_single_string(self):
        """Single string without commas becomes single-element list."""
        assert _coerce_list("hello") == ["hello"]

    def test_int_wrap(self):
        """An integer is wrapped into a single-element list."""
        assert _coerce_list(42) == ["42"]

    def test_empty_string(self):
        """Empty string returns empty list."""
        assert _coerce_list("") == []

    def test_empty_list(self):
        """Empty list returns empty list."""
        assert _coerce_list([]) == []


class TestCoerceDict:
    """Tests for _coerce_dict helper."""

    def test_dict_passthrough(self):
        """A dict passes through unchanged."""
        d = {"key": "value"}
        assert _coerce_dict(d) == d

    def test_string_to_query(self):
        """A string becomes {"query": ...}."""
        assert _coerce_dict("search term") == {"query": "search term"}

    def test_int_to_value(self):
        """An int becomes {"value": ...}."""
        assert _coerce_dict(42) == {"value": 42}

    def test_none(self):
        """None becomes {"value": None}."""
        assert _coerce_dict(None) == {"value": None}


class TestDataRequirementValidators:
    """Tests for DataRequirement field validators."""

    def test_query_params_string_coercion(self):
        """String query_params is coerced to dict."""
        dr = DataRequirement(
            description="test", source_type="mcp_server", query_params="gene expression"
        )
        assert dr.query_params == {"query": "gene expression"}

    def test_query_params_dict_passthrough(self):
        """Dict query_params passes through."""
        dr = DataRequirement(
            description="test", source_type="mcp_server",
            query_params={"gene": "TP53", "organism": "human"},
        )
        assert dr.query_params == {"gene": "TP53", "organism": "human"}

    def test_priority_int_coercion(self):
        """Integer priority is coerced to string."""
        dr = DataRequirement(
            description="test", source_type="download", priority=1
        )
        assert dr.priority == "1"

    def test_priority_string_passthrough(self):
        """String priority passes through."""
        dr = DataRequirement(
            description="test", source_type="download", priority="required"
        )
        assert dr.priority == "required"


class TestAnalysisStepValidators:
    """Tests for AnalysisStep field validators."""

    def test_inputs_from_string(self):
        """Comma-separated inputs string is split."""
        step = AnalysisStep(
            name="test", description="test", method="regression",
            inputs="dataset_a, dataset_b",
        )
        assert step.inputs == ["dataset_a", "dataset_b"]

    def test_parameters_from_string(self):
        """String parameters coerced to dict."""
        step = AnalysisStep(
            name="test", description="test", method="regression",
            parameters="alpha=0.05",
        )
        assert step.parameters == {"query": "alpha=0.05"}

    def test_statistical_tests_from_string(self):
        """Comma-separated statistical_tests string is split."""
        step = AnalysisStep(
            name="test", description="test", method="regression",
            statistical_tests="t-test, anova",
        )
        assert step.statistical_tests == ["t-test", "anova"]

    def test_all_fields_from_lists(self):
        """Normal list values pass through."""
        step = AnalysisStep(
            name="test", description="test", method="regression",
            inputs=["d1"], parameters={"alpha": 0.05},
            statistical_tests=["chi2"],
        )
        assert step.inputs == ["d1"]
        assert step.parameters == {"alpha": 0.05}
        assert step.statistical_tests == ["chi2"]


class TestResearchPlanValidators:
    """Tests for ResearchPlan field validators."""

    def test_risks_from_string(self):
        """String risks field is coerced to list."""
        plan = ResearchPlan(
            mission_id="test", summary="s", approach="a",
            risks="data quality issues",
        )
        assert plan.risks == ["data quality issues"]

    def test_risks_from_list_of_dicts(self):
        """List of dicts for risks is flattened to strings."""
        plan = ResearchPlan(
            mission_id="test", summary="s", approach="a",
            risks=[{"risk": "low power", "mitigation": "increase n"}],
        )
        assert len(plan.risks) == 1
        assert "low power" in plan.risks[0]
        assert "increase n" in plan.risks[0]

    def test_risks_from_list_of_strings(self):
        """List of strings passes through."""
        plan = ResearchPlan(
            mission_id="test", summary="s", approach="a",
            risks=["risk1", "risk2"],
        )
        assert plan.risks == ["risk1", "risk2"]

    def test_float_from_string(self):
        """String float is coerced."""
        plan = ResearchPlan(
            mission_id="test", summary="s", approach="a",
            estimated_compute_cost="3.50",
        )
        assert plan.estimated_compute_cost == 3.50

    def test_float_unparseable(self):
        """Unparseable float defaults to 0.0."""
        plan = ResearchPlan(
            mission_id="test", summary="s", approach="a",
            estimated_compute_cost="not a number",
        )
        assert plan.estimated_compute_cost == 0.0

    def test_float_none(self):
        """None float defaults to 0.0."""
        plan = ResearchPlan(
            mission_id="test", summary="s", approach="a",
            estimated_time_hours=None,
        )
        assert plan.estimated_time_hours == 0.0


class TestExtractJson:
    """Tests for LLMProvider._extract_json static method."""

    def test_plain_json(self):
        """Plain JSON string is parsed."""
        result = LLMProvider._extract_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_think_tags_stripped(self):
        """Think tags are stripped before parsing."""
        text = '<think>reasoning</think>{"key": "value"}'
        result = LLMProvider._extract_json(text)
        assert result == {"key": "value"}

    def test_markdown_fence(self):
        """JSON in markdown code fence is extracted."""
        text = '```json\n{"key": "value"}\n```'
        result = LLMProvider._extract_json(text)
        assert result == {"key": "value"}

    def test_trailing_commas(self):
        """Trailing commas are fixed."""
        text = '{"key": "value",}'
        result = LLMProvider._extract_json(text)
        assert result == {"key": "value"}

    def test_embedded_in_prose(self):
        """JSON embedded in prose is extracted via brace matching."""
        text = 'Here is the result: {"key": "value"} as requested.'
        result = LLMProvider._extract_json(text)
        assert result == {"key": "value"}

    def test_nested_json(self):
        """Nested JSON objects are handled."""
        text = '{"outer": {"inner": 1}}'
        result = LLMProvider._extract_json(text)
        assert result == {"outer": {"inner": 1}}

    def test_think_tags_with_markdown_fence(self):
        """Think tags + markdown fence combo works."""
        text = '<think>let me think</think>\n```json\n{"a": 1}\n```'
        result = LLMProvider._extract_json(text)
        assert result == {"a": 1}

    def test_invalid_raises(self):
        """Invalid JSON raises JSONDecodeError."""
        with pytest.raises(json.JSONDecodeError):
            LLMProvider._extract_json("not json at all")
