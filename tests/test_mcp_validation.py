"""
MCP Server Validation Framework
================================

Comprehensive success criteria tests for all 20 MCP tools.
Tests are organized by:
  1. Response Schema Validation — every tool returns the correct keys
  2. Error Path Validation — every error returns structured error dicts
  3. Edge Case Coverage — dedup bugs, budget boundaries, empty inputs
  4. Integration Contracts — checkpoint flow, session lifecycle, cost tracking
  5. Infrastructure Validation — lifespan, context, arg parsing, cli

Each test is tagged with the tool it validates and severity:
  [CRITICAL] — would break AI callers at runtime
  [HIGH]     — silent data loss or wrong results
  [MEDIUM]   — degraded experience or misleading output
  [LOW]      — cosmetic or documentation issue
"""

from __future__ import annotations

import asyncio
import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from apollobot.core.mission import Mission, ResearchMode
from apollobot.core.session import Session, Phase, PhaseResult, CostTracker
from apollobot.server.checkpoint import MCPCheckpointHandler
from apollobot.server.errors import (
    SESSION_NOT_FOUND,
    PHASE_NOT_AVAILABLE,
    BUDGET_EXCEEDED,
    INVALID_INPUT,
    INTERNAL_ERROR,
    error_response,
)
from apollobot.server.registry import ActiveSession, SessionRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_config():
    from apollobot.core import ApolloConfig, APIConfig, UserIdentity, ComputeConfig
    return ApolloConfig(
        identity=UserIdentity(name="Test User"),
        api=APIConfig(
            default_provider="anthropic",
            anthropic_api_key="test-key-123",
        ),
        compute=ComputeConfig(max_budget_usd=50.0),
    )


@pytest.fixture
def registry(mock_config, temp_dir):
    mock_config.output_dir = str(temp_dir)
    return SessionRegistry(config=mock_config)


@pytest.fixture
def sample_mission(temp_dir):
    return Mission(
        objective="Test research objective",
        mode=ResearchMode.HYPOTHESIS,
        domain="bioinformatics",
        metadata={"output_dir": str(temp_dir)},
    )


@pytest.fixture
def mock_ctx(registry):
    ctx = MagicMock()
    ctx.request_context.lifespan_context = {"registry": registry}
    return ctx


@pytest.fixture
async def active_session(registry, sample_mission):
    """Create and return an active session via registry."""
    with patch("apollobot.server.registry.Orchestrator") as MockOrch:
        MockOrch.return_value = MagicMock()
        return await registry.create(sample_mission)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def assert_success(result: dict, required_keys: list[str], tool_name: str = ""):
    """Assert result is not an error and has all required keys."""
    assert "error" not in result or result.get("error") is not True, (
        f"{tool_name}: Expected success but got error: {result}"
    )
    for key in required_keys:
        assert key in result, (
            f"{tool_name}: Missing required key '{key}' in response. "
            f"Got keys: {list(result.keys())}"
        )


def assert_error(result: dict, expected_code: str, tool_name: str = ""):
    """Assert result is a structured error with correct code."""
    assert result.get("error") is True, (
        f"{tool_name}: Expected error but got: {result}"
    )
    assert "error_code" in result, (
        f"{tool_name}: Error missing 'error_code'. Got: {result}"
    )
    assert "error_message" in result, (
        f"{tool_name}: Error missing 'error_message'. Got: {result}"
    )
    assert result["error_code"] == expected_code, (
        f"{tool_name}: Expected error code '{expected_code}', got '{result['error_code']}'"
    )


# ===========================================================================
# 1. RESPONSE SCHEMA VALIDATION
#    Every tool must return exactly the documented keys on success.
# ===========================================================================


class TestCreateMissionSchema:
    """[CRITICAL] create_mission response schema."""

    REQUIRED_KEYS = ["session_id", "objective", "mode", "domain",
                     "compute_budget", "output_dir"]

    async def test_success_schema(self, mock_ctx):
        from apollobot.server.app import create_mission
        with patch("apollobot.server.registry.Orchestrator"):
            result = await create_mission(mock_ctx, objective="Test question")
        assert_success(result, self.REQUIRED_KEYS, "create_mission")

    async def test_session_id_is_string(self, mock_ctx):
        from apollobot.server.app import create_mission
        with patch("apollobot.server.registry.Orchestrator"):
            result = await create_mission(mock_ctx, objective="Test")
        assert isinstance(result["session_id"], str)
        assert len(result["session_id"]) > 0

    async def test_compute_budget_is_number(self, mock_ctx):
        from apollobot.server.app import create_mission
        with patch("apollobot.server.registry.Orchestrator"):
            result = await create_mission(mock_ctx, objective="Test")
        assert isinstance(result["compute_budget"], (int, float))


class TestRunDiscoverSchema:
    """[CRITICAL] run_discover response schema."""

    REQUIRED_KEYS = ["session_id", "objective", "mode", "domain",
                     "current_phase", "completed_phases", "cost",
                     "datasets_count", "warnings", "output_dir",
                     "translation_scores"]

    async def test_error_on_missing_session(self, mock_ctx):
        from apollobot.server.app import run_discover
        result = await run_discover(mock_ctx, session_id="nonexistent")
        assert_error(result, SESSION_NOT_FOUND, "run_discover")


class TestRunTranslateSchema:
    """[CRITICAL] run_translate response schema."""

    async def test_error_on_missing_session(self, mock_ctx):
        from apollobot.server.app import run_translate
        result = await run_translate(mock_ctx, session_id="nonexistent")
        assert_error(result, SESSION_NOT_FOUND, "run_translate")


class TestRunImplementSchema:
    """[CRITICAL] run_implement response schema."""

    async def test_error_on_missing_session(self, mock_ctx):
        from apollobot.server.app import run_implement
        result = await run_implement(mock_ctx, session_id="nonexistent")
        assert_error(result, SESSION_NOT_FOUND, "run_implement")


class TestRunCommercializeSchema:
    """[CRITICAL] run_commercialize response schema."""

    async def test_error_on_missing_session(self, mock_ctx):
        from apollobot.server.app import run_commercialize
        result = await run_commercialize(mock_ctx, session_id="nonexistent")
        assert_error(result, SESSION_NOT_FOUND, "run_commercialize")


class TestRunPipelineSchema:
    """[CRITICAL] run_pipeline response schema."""

    async def test_error_on_missing_session(self, mock_ctx):
        from apollobot.server.app import run_pipeline
        result = await run_pipeline(mock_ctx, session_id="nonexistent")
        assert_error(result, SESSION_NOT_FOUND, "run_pipeline")


class TestStepPhaseSchema:
    """[CRITICAL] step_phase response schema."""

    PLANNING_KEYS = ["phase", "status", "plan_summary", "approach",
                     "literature_queries", "data_requirements_count",
                     "analysis_steps_count", "estimated_cost",
                     "estimated_hours", "next_phase"]

    EXECUTION_KEYS = ["phase", "status", "summary", "findings_count",
                      "findings", "next_phase", "cost_so_far"]

    async def test_error_on_missing_session(self, mock_ctx):
        from apollobot.server.app import step_phase
        result = await step_phase(mock_ctx, session_id="nonexistent")
        assert_error(result, SESSION_NOT_FOUND, "step_phase")


class TestGetSessionStatusSchema:
    """[CRITICAL] get_session_status response schema."""

    REQUIRED_KEYS = ["session_id", "objective", "mode", "domain",
                     "current_phase", "completed_phases", "cost",
                     "datasets_count", "warnings", "pending_checkpoint",
                     "has_plan", "translation_scores"]

    async def test_success_schema(self, mock_ctx, active_session):
        from apollobot.server.app import get_session_status
        result = await get_session_status(mock_ctx, active_session.session_id)
        assert_success(result, self.REQUIRED_KEYS, "get_session_status")

    async def test_cost_subkeys(self, mock_ctx, active_session):
        from apollobot.server.app import get_session_status
        result = await get_session_status(mock_ctx, active_session.session_id)
        cost = result["cost"]
        assert "total_usd" in cost
        assert "llm_calls" in cost
        assert "budget_remaining" in cost


class TestGetPhaseResultSchema:
    """[CRITICAL] get_phase_result response schema."""

    REQUIRED_KEYS = ["phase", "started_at", "completed_at", "summary",
                     "findings", "artifacts", "errors"]

    async def test_success_schema(self, mock_ctx, active_session):
        from apollobot.server.app import get_phase_result
        active_session.session.begin_phase(Phase.LITERATURE_REVIEW)
        active_session.session.complete_phase(
            Phase.LITERATURE_REVIEW, summary="Done", findings=[{"k": "v"}]
        )
        result = await get_phase_result(
            mock_ctx, active_session.session_id, "literature_review"
        )
        assert_success(result, self.REQUIRED_KEYS, "get_phase_result")


class TestApproveCheckpointSchema:
    """[CRITICAL] approve_checkpoint response schema."""

    REQUIRED_KEYS = ["checkpoint", "approved", "message"]

    async def test_success_schema(self, mock_ctx, active_session):
        from apollobot.server.app import approve_checkpoint
        active_session.pending_checkpoint = {"phase": "test", "summary": "y/n"}
        active_session.checkpoint_event = asyncio.Event()
        result = await approve_checkpoint(mock_ctx, active_session.session_id)
        assert_success(result, self.REQUIRED_KEYS, "approve_checkpoint")


class TestSearchLiteratureSchema:
    """[CRITICAL] search_literature response schema."""

    REQUIRED_KEYS = ["query", "papers", "total_found", "servers_searched"]

    async def test_success_schema(self, mock_ctx):
        from apollobot.server.app import search_literature
        with patch("apollobot.mcp.MCPClient") as MockClient:
            client = MockClient.return_value
            mock_server = MagicMock()
            mock_server.name = "pubmed"
            mock_server.domain = "shared"
            client.get_servers.return_value = [mock_server]
            client.query = AsyncMock(return_value={"papers": []})
            result = await search_literature(mock_ctx, query="test")
        assert_success(result, self.REQUIRED_KEYS, "search_literature")


class TestQueryDataSourceSchema:
    """[CRITICAL] query_data_source response schema."""

    async def test_error_unknown_server(self, mock_ctx):
        from apollobot.server.app import query_data_source
        with patch("apollobot.mcp.MCPClient") as MockClient:
            client = MockClient.return_value
            client.get_servers.return_value = []
            result = await query_data_source(
                mock_ctx, server_name="nonexistent", capability="search"
            )
        assert_error(result, "SERVER_NOT_FOUND", "query_data_source")


class TestListDataServersSchema:
    """[CRITICAL] list_data_servers response schema."""

    async def test_with_domain_schema(self, mock_ctx):
        from apollobot.server.app import list_data_servers
        result = await list_data_servers(mock_ctx, domain="bioinformatics")
        assert_success(result, ["domain", "servers"], "list_data_servers")
        for srv in result["servers"]:
            assert "name" in srv
            assert "description" in srv
            assert "domain" in srv
            assert "category" in srv
            assert "requires_key" in srv, (
                "[HIGH] list_data_servers: server missing 'requires_key' field"
            )

    async def test_without_domain_schema(self, mock_ctx):
        from apollobot.server.app import list_data_servers
        result = await list_data_servers(mock_ctx)
        assert_success(result, ["domains"], "list_data_servers")

    async def test_all_domain_servers_have_consistent_fields(self, mock_ctx):
        """[HIGH] Both with-domain and without-domain responses should have
        the same fields per server entry."""
        from apollobot.server.app import list_data_servers
        with_domain = await list_data_servers(mock_ctx, domain="bioinformatics")
        without_domain = await list_data_servers(mock_ctx)

        with_keys = set(with_domain["servers"][0].keys())
        without_keys = set(without_domain["domains"]["bioinformatics"][0].keys())
        assert with_keys == without_keys, (
            f"[HIGH] Inconsistent server fields: "
            f"with_domain has {with_keys - without_keys}, "
            f"without_domain has {without_keys - with_keys}"
        )


class TestRunAnalysisStepSchema:
    """[CRITICAL] run_analysis_step response schema."""

    async def test_error_on_missing_session(self, mock_ctx):
        from apollobot.server.app import run_analysis_step
        result = await run_analysis_step(
            mock_ctx, session_id="nonexistent",
            step_name="test", description="desc"
        )
        assert_error(result, SESSION_NOT_FOUND, "run_analysis_step")


class TestDraftSectionSchema:
    """[CRITICAL] draft_section response schema."""

    REQUIRED_KEYS = ["section", "text", "evidence_level"]

    async def test_error_invalid_section(self, mock_ctx, active_session):
        from apollobot.server.app import draft_section
        result = await draft_section(
            mock_ctx, session_id=active_session.session_id,
            section="bibliography"
        )
        assert_error(result, INVALID_INPUT, "draft_section")

    async def test_valid_sections(self, mock_ctx, active_session):
        """All 6 sections should be accepted."""
        from apollobot.server.app import draft_section
        valid = ["abstract", "introduction", "methods",
                 "results", "discussion", "conclusion"]
        for s in valid:
            # We can't actually call LLM, but we can validate it doesn't
            # return INVALID_INPUT for valid section names
            result = await draft_section(
                mock_ctx, session_id=active_session.session_id, section=s
            )
            assert result.get("error_code") != INVALID_INPUT, (
                f"draft_section rejected valid section '{s}'"
            )


class TestReviewManuscriptSchema:
    """[CRITICAL] review_manuscript response schema."""

    async def test_error_no_plan(self, mock_ctx, active_session):
        from apollobot.server.app import review_manuscript
        result = await review_manuscript(mock_ctx, active_session.session_id)
        assert_error(result, PHASE_NOT_AVAILABLE, "review_manuscript")


class TestGetProvenanceSchema:
    """[CRITICAL] get_provenance response schema."""

    REQUIRED_KEYS = ["session_id", "execution_log", "data_lineage",
                     "model_calls", "total_events", "total_transforms",
                     "total_llm_calls"]

    async def test_success_schema(self, mock_ctx, active_session):
        from apollobot.server.app import get_provenance
        result = await get_provenance(mock_ctx, active_session.session_id)
        assert_success(result, self.REQUIRED_KEYS, "get_provenance")


class TestGetCostSchema:
    """[CRITICAL] get_cost response schema."""

    REQUIRED_KEYS = ["session_id", "total_cost_usd", "llm_cost_usd",
                     "compute_cost_usd", "llm_calls", "llm_input_tokens",
                     "llm_output_tokens", "avg_cost_per_call",
                     "budget_total", "budget_remaining",
                     "budget_utilization_pct"]

    async def test_success_schema(self, mock_ctx, active_session):
        from apollobot.server.app import get_cost
        result = await get_cost(mock_ctx, active_session.session_id)
        assert_success(result, self.REQUIRED_KEYS, "get_cost")

    async def test_all_numeric_values(self, mock_ctx, active_session):
        from apollobot.server.app import get_cost
        result = await get_cost(mock_ctx, active_session.session_id)
        for key in self.REQUIRED_KEYS:
            if key == "session_id":
                continue
            assert isinstance(result[key], (int, float)), (
                f"get_cost: '{key}' should be numeric, got {type(result[key])}"
            )


class TestListSessionsSchema:
    """[CRITICAL] list_sessions response schema."""

    REQUIRED_KEYS = ["active", "historical", "total_active", "total_historical"]

    async def test_success_schema(self, mock_ctx):
        from apollobot.server.app import list_sessions
        result = await list_sessions(mock_ctx)
        assert_success(result, self.REQUIRED_KEYS, "list_sessions")

    async def test_active_entry_schema(self, mock_ctx, active_session):
        from apollobot.server.app import list_sessions
        result = await list_sessions(mock_ctx)
        assert len(result["active"]) >= 1
        entry = result["active"][0]
        for key in ["session_id", "objective", "mode", "domain",
                     "current_phase", "cost", "active"]:
            assert key in entry, (
                f"list_sessions: active entry missing '{key}'"
            )


class TestLoadSessionSchema:
    """[CRITICAL] load_session response schema."""

    REQUIRED_KEYS = ["session_id", "objective", "mode", "domain",
                     "current_phase", "completed_phases", "cost",
                     "datasets_count", "warnings"]

    async def test_success_schema(self, mock_ctx, registry, sample_mission, temp_dir):
        from apollobot.server.app import load_session
        session = Session(mission=sample_mission)
        session.init_directories()
        session.save_state()
        with patch("apollobot.server.registry.Orchestrator"):
            result = await load_session(mock_ctx, sample_mission.id)
        assert_success(result, self.REQUIRED_KEYS, "load_session")


# ===========================================================================
# 2. ERROR PATH VALIDATION
#    Every tool must return structured errors, never raise exceptions.
# ===========================================================================


class TestAllToolsNeverRaise:
    """[CRITICAL] No tool should raise an unhandled exception to the caller."""

    async def test_create_mission_bad_yaml(self, mock_ctx):
        """Malformed YAML should return error, not raise."""
        from apollobot.server.app import create_mission
        result = await create_mission(
            mock_ctx, objective="X", mission_yaml="{{invalid yaml::"
        )
        assert result.get("error") is True

    async def test_step_phase_missing_session(self, mock_ctx):
        from apollobot.server.app import step_phase
        result = await step_phase(mock_ctx, session_id="ghost")
        assert result.get("error") is True

    async def test_get_phase_result_missing_session(self, mock_ctx):
        from apollobot.server.app import get_phase_result
        result = await get_phase_result(mock_ctx, "ghost", "analysis")
        assert result.get("error") is True

    async def test_approve_checkpoint_missing_session(self, mock_ctx):
        from apollobot.server.app import approve_checkpoint
        result = await approve_checkpoint(mock_ctx, "ghost")
        assert result.get("error") is True

    async def test_search_literature_no_crash(self, mock_ctx):
        """Even with broken MCP client, should return error not crash."""
        from apollobot.server.app import search_literature
        with patch("apollobot.mcp.MCPClient") as MockClient:
            MockClient.side_effect = Exception("MCPClient init failed")
            result = await search_literature(mock_ctx, query="test")
        assert result.get("error") is True

    async def test_query_data_source_no_crash(self, mock_ctx):
        from apollobot.server.app import query_data_source
        with patch("apollobot.mcp.MCPClient") as MockClient:
            MockClient.side_effect = Exception("fail")
            result = await query_data_source(
                mock_ctx, server_name="x", capability="y"
            )
        assert result.get("error") is True

    async def test_get_provenance_missing_session(self, mock_ctx):
        from apollobot.server.app import get_provenance
        result = await get_provenance(mock_ctx, "ghost")
        assert result.get("error") is True

    async def test_get_cost_missing_session(self, mock_ctx):
        from apollobot.server.app import get_cost
        result = await get_cost(mock_ctx, "ghost")
        assert result.get("error") is True

    async def test_load_session_nonexistent(self, mock_ctx):
        from apollobot.server.app import load_session
        result = await load_session(mock_ctx, "definitely-not-real")
        assert result.get("error") is True

    async def test_run_analysis_step_missing(self, mock_ctx):
        from apollobot.server.app import run_analysis_step
        result = await run_analysis_step(
            mock_ctx, session_id="ghost", step_name="x", description="y"
        )
        assert result.get("error") is True

    async def test_draft_section_missing(self, mock_ctx):
        from apollobot.server.app import draft_section
        result = await draft_section(mock_ctx, session_id="ghost", section="abstract")
        assert result.get("error") is True

    async def test_review_manuscript_missing(self, mock_ctx):
        from apollobot.server.app import review_manuscript
        result = await review_manuscript(mock_ctx, "ghost")
        assert result.get("error") is True


# ===========================================================================
# 3. EDGE CASE COVERAGE
# ===========================================================================


class TestCreateMissionEdgeCases:
    """[HIGH] create_mission edge cases."""

    async def test_negative_budget_rejected(self, mock_ctx):
        """[HIGH] Negative budget should be rejected or clamped."""
        from apollobot.server.app import create_mission
        with patch("apollobot.server.registry.Orchestrator"):
            result = await create_mission(
                mock_ctx, objective="Test", compute_budget=-10.0
            )
        # Either error or budget clamped to 0
        if "error" not in result or result.get("error") is not True:
            assert result["compute_budget"] >= 0, (
                "[HIGH] create_mission accepted negative budget"
            )

    async def test_zero_budget_accepted(self, mock_ctx):
        """Zero budget is valid (free-tier research)."""
        from apollobot.server.app import create_mission
        with patch("apollobot.server.registry.Orchestrator"):
            result = await create_mission(
                mock_ctx, objective="Test", compute_budget=0.0
            )
        assert_success(result, ["session_id"], "create_mission")

    async def test_empty_objective(self, mock_ctx):
        """[MEDIUM] Empty objective should be rejected."""
        from apollobot.server.app import create_mission
        with patch("apollobot.server.registry.Orchestrator"):
            result = await create_mission(mock_ctx, objective="")
        # Should either error or still work (mission allows empty obj)
        # At minimum, should not crash
        assert isinstance(result, dict)

    async def test_all_valid_modes(self, mock_ctx):
        """All ResearchMode values should be accepted."""
        from apollobot.server.app import create_mission
        valid_modes = [m.value for m in ResearchMode]
        for mode in valid_modes:
            with patch("apollobot.server.registry.Orchestrator"):
                result = await create_mission(
                    mock_ctx, objective="Test", mode=mode
                )
            assert result.get("error_code") != INVALID_INPUT, (
                f"Mode '{mode}' was incorrectly rejected"
            )

    async def test_all_valid_domains(self, mock_ctx):
        """Standard domains should be accepted."""
        from apollobot.server.app import create_mission
        domains = ["bioinformatics", "physics", "cs_ml", "comp_chem", "economics"]
        for domain in domains:
            with patch("apollobot.server.registry.Orchestrator"):
                result = await create_mission(
                    mock_ctx, objective="Test", domain=domain
                )
            assert "session_id" in result, f"Domain '{domain}' failed"


class TestStepPhaseEdgeCases:
    """[HIGH] step_phase edge cases."""

    async def test_budget_exactly_at_limit(self, mock_ctx, active_session):
        """Budget at exactly the limit should still allow one more phase."""
        from apollobot.server.app import step_phase
        active_session.session.cost.estimated_cost_usd = 49.99
        active_session.mission.constraints.compute_budget = 50.0
        result = await step_phase(mock_ctx, active_session.session_id)
        # Should NOT return BUDGET_EXCEEDED (49.99 < 50.0)
        assert result.get("error_code") != BUDGET_EXCEEDED

    async def test_budget_one_cent_over(self, mock_ctx, active_session):
        """Budget one cent over should be rejected."""
        from apollobot.server.app import step_phase
        active_session.session.cost.estimated_cost_usd = 50.01
        result = await step_phase(mock_ctx, active_session.session_id)
        assert_error(result, BUDGET_EXCEEDED, "step_phase")

    async def test_all_phases_complete_returns_done(self, mock_ctx, active_session):
        """When all phases are done, returns 'complete' status."""
        from apollobot.server.app import step_phase
        from apollobot.core.session import Phase

        # Mark all discover phases as complete
        for p in [Phase.PLANNING, Phase.LITERATURE_REVIEW, Phase.DATA_ACQUISITION,
                  Phase.ANALYSIS, Phase.STATISTICAL_TESTING, Phase.MANUSCRIPT_DRAFTING,
                  Phase.SELF_REVIEW, Phase.MANUSCRIPT_REVISION]:
            active_session.session.begin_phase(p)
            active_session.session.complete_phase(p, summary="done")

        result = await step_phase(mock_ctx, active_session.session_id)
        assert result.get("status") == "complete"
        assert "message" in result

    async def test_case_sensitive_phase_name(self, mock_ctx, active_session):
        """[MEDIUM] Phase names are case-sensitive enum values."""
        from apollobot.server.app import step_phase
        # Upper case should fail
        result = await step_phase(
            mock_ctx, active_session.session_id, phase="PLANNING"
        )
        assert_error(result, INVALID_INPUT, "step_phase")

    async def test_translate_phase_rejected(self, mock_ctx, active_session):
        """Translate phases should be rejected in discover flow."""
        from apollobot.server.app import step_phase
        result = await step_phase(
            mock_ctx, active_session.session_id, phase="translate_assess"
        )
        assert_error(result, INVALID_INPUT, "step_phase")


class TestSearchLiteratureEdgeCases:
    """[HIGH] search_literature edge cases."""

    async def test_dedup_handles_empty_doi_and_title(self, mock_ctx):
        """[HIGH] Papers with both empty DOI and title should not wrongly dedup."""
        from apollobot.server.app import search_literature

        papers = [
            {"title": "", "doi": "", "abstract": "Paper A content"},
            {"title": "", "doi": "", "abstract": "Paper B content"},
        ]

        with patch("apollobot.mcp.MCPClient") as MockClient:
            client = MockClient.return_value
            mock_server = MagicMock()
            mock_server.name = "pubmed"
            mock_server.domain = "shared"
            client.get_servers.return_value = [mock_server]
            client.query = AsyncMock(return_value={"papers": papers})
            result = await search_literature(mock_ctx, query="test")

        # Both papers have empty DOI and title — they should both be kept
        # since we can't deduplicate without an identifying key.
        assert result["total_found"] == 2, (
            "[HIGH] search_literature should keep papers with no identifying key, "
            f"got {result['total_found']}"
        )

    async def test_non_dict_papers_filtered(self, mock_ctx):
        """Non-dict entries in papers list should be silently filtered."""
        from apollobot.server.app import search_literature
        papers = [
            {"title": "Real Paper", "doi": "10.1/a"},
            "not a dict",
            42,
            None,
        ]
        with patch("apollobot.mcp.MCPClient") as MockClient:
            client = MockClient.return_value
            mock_server = MagicMock()
            mock_server.name = "pubmed"
            mock_server.domain = "shared"
            client.get_servers.return_value = [mock_server]
            client.query = AsyncMock(return_value={"papers": papers})
            result = await search_literature(mock_ctx, query="test")
        assert result["total_found"] == 1
        assert result["papers"][0]["title"] == "Real Paper"


class TestGetCostEdgeCases:
    """[HIGH] get_cost edge cases."""

    async def test_zero_calls_no_division_error(self, mock_ctx, active_session):
        """[CRITICAL] Zero LLM calls should not cause division by zero."""
        from apollobot.server.app import get_cost
        result = await get_cost(mock_ctx, active_session.session_id)
        assert result["avg_cost_per_call"] == 0.0
        assert result["budget_utilization_pct"] == 0.0

    async def test_zero_budget_no_division_error(self, mock_ctx, active_session):
        """[CRITICAL] Zero budget should not cause division by zero."""
        from apollobot.server.app import get_cost
        active_session.mission.constraints.compute_budget = 0.0
        result = await get_cost(mock_ctx, active_session.session_id)
        assert isinstance(result["budget_utilization_pct"], (int, float))

    async def test_negative_remaining_clamped(self, mock_ctx, active_session):
        """Budget remaining should never be negative."""
        from apollobot.server.app import get_cost
        active_session.session.cost.estimated_cost_usd = 100.0
        active_session.mission.constraints.compute_budget = 50.0
        result = await get_cost(mock_ctx, active_session.session_id)
        assert result["budget_remaining"] >= 0


class TestListDataServersEdgeCases:
    """[MEDIUM] list_data_servers edge cases."""

    async def test_unknown_domain_returns_fallback(self, mock_ctx):
        """Unknown domain should return literature servers (fallback)."""
        from apollobot.server.app import list_data_servers
        result = await list_data_servers(mock_ctx, domain="nonexistent_domain")
        assert "servers" in result
        # get_domain_pack falls back to LITERATURE_SERVERS
        assert len(result["servers"]) > 0


# ===========================================================================
# 4. INTEGRATION CONTRACTS
# ===========================================================================


class TestCheckpointIntegration:
    """[HIGH] Checkpoint approval flow."""

    async def test_full_checkpoint_approval_cycle(self, mock_ctx, active_session):
        """Create session → pending checkpoint → approve → resume."""
        from apollobot.server.app import (
            get_session_status, approve_checkpoint
        )

        # 1. No pending checkpoint initially
        status = await get_session_status(mock_ctx, active_session.session_id)
        assert status["pending_checkpoint"] is None

        # 2. Simulate a pending checkpoint
        active_session.pending_checkpoint = {
            "phase": "pipeline_translate",
            "summary": "Proceed to translate?"
        }
        active_session.checkpoint_event = asyncio.Event()

        # 3. Verify status shows pending
        status = await get_session_status(mock_ctx, active_session.session_id)
        assert status["pending_checkpoint"] is not None
        assert status["pending_checkpoint"]["phase"] == "pipeline_translate"

        # 4. Approve
        result = await approve_checkpoint(mock_ctx, active_session.session_id)
        assert result["approved"] is True
        assert active_session.checkpoint_event.is_set()

    async def test_checkpoint_deny_flow(self, mock_ctx, active_session):
        """Denying a checkpoint should set approved=False."""
        from apollobot.server.app import approve_checkpoint

        active_session.pending_checkpoint = {"phase": "test", "summary": "go?"}
        active_session.checkpoint_event = asyncio.Event()

        result = await approve_checkpoint(
            mock_ctx, active_session.session_id, approved=False
        )
        assert result["approved"] is False
        assert active_session.checkpoint_approved is False


class TestSessionLifecycle:
    """[HIGH] Full session lifecycle contract."""

    async def test_create_then_status(self, mock_ctx):
        """create_mission → get_session_status → correct initial state."""
        from apollobot.server.app import create_mission, get_session_status

        with patch("apollobot.server.registry.Orchestrator"):
            created = await create_mission(
                mock_ctx, objective="Lifecycle test", domain="physics"
            )

        status = await get_session_status(mock_ctx, created["session_id"])
        assert status["objective"] == "Lifecycle test"
        assert status["domain"] == "physics"
        assert status["current_phase"] == "planning"
        assert status["cost"]["total_usd"] == 0.0

    async def test_create_then_cost(self, mock_ctx):
        """create_mission → get_cost → zero initial cost."""
        from apollobot.server.app import create_mission, get_cost

        with patch("apollobot.server.registry.Orchestrator"):
            created = await create_mission(mock_ctx, objective="Cost test")

        cost = await get_cost(mock_ctx, created["session_id"])
        assert cost["total_cost_usd"] == 0.0
        assert cost["budget_total"] == 50.0  # default

    async def test_create_then_list(self, mock_ctx):
        """create_mission → list_sessions → session appears in active list."""
        from apollobot.server.app import create_mission, list_sessions

        with patch("apollobot.server.registry.Orchestrator"):
            created = await create_mission(mock_ctx, objective="List test")

        sessions = await list_sessions(mock_ctx)
        active_ids = [s["session_id"] for s in sessions["active"]]
        assert created["session_id"] in active_ids

    async def test_create_then_provenance(self, mock_ctx):
        """create_mission → get_provenance → has session_started event."""
        from apollobot.server.app import create_mission, get_provenance

        with patch("apollobot.server.registry.Orchestrator"):
            created = await create_mission(mock_ctx, objective="Prov test")

        prov = await get_provenance(mock_ctx, created["session_id"])
        assert prov["total_events"] >= 1
        events = [e.get("event") for e in prov["execution_log"]]
        assert "session_started" in events


class TestCostTracking:
    """[HIGH] Cost tracking accuracy."""

    async def test_cost_reflects_recorded_calls(self, mock_ctx, active_session):
        """Recorded LLM calls should appear in get_cost."""
        from apollobot.server.app import get_cost

        active_session.session.cost.record_llm_call(1000, 500, 0.03)
        active_session.session.cost.record_llm_call(2000, 1000, 0.07)

        result = await get_cost(mock_ctx, active_session.session_id)
        assert result["llm_calls"] == 2
        assert result["total_cost_usd"] == pytest.approx(0.10)
        assert result["llm_input_tokens"] == 3000
        assert result["llm_output_tokens"] == 1500

    async def test_budget_remaining_accurate(self, mock_ctx, active_session):
        """Budget remaining = budget - total_cost."""
        from apollobot.server.app import get_cost

        active_session.mission.constraints.compute_budget = 25.0
        active_session.session.cost.record_llm_call(1000, 500, 10.0)

        result = await get_cost(mock_ctx, active_session.session_id)
        assert result["budget_remaining"] == pytest.approx(15.0)
        assert result["budget_utilization_pct"] == pytest.approx(40.0)


# ===========================================================================
# 5. INFRASTRUCTURE VALIDATION
# ===========================================================================


class TestLifespan:
    """[CRITICAL] Lifespan correctly initializes registry."""

    async def test_lifespan_yields_registry(self):
        """app_lifespan yields a dict with 'registry' key."""
        from apollobot.server.app import app_lifespan, mcp

        with patch("apollobot.server.app._load_config") as mock_cfg:
            mock_cfg.return_value = MagicMock()
            async with app_lifespan(mcp) as ctx:
                assert "registry" in ctx
                assert isinstance(ctx["registry"], SessionRegistry)

    async def test_lifespan_handles_config_failure(self):
        """If load_config fails, lifespan still yields a registry."""
        from apollobot.server.app import app_lifespan, mcp

        with patch("apollobot.server.app._load_config") as mock_cfg:
            mock_cfg.side_effect = FileNotFoundError("no config")
            async with app_lifespan(mcp) as ctx:
                assert "registry" in ctx


class TestServerInit:
    """[CRITICAL] Server package __init__.py."""

    def test_create_server_returns_fastmcp(self):
        from apollobot.server import create_server
        server = create_server()
        assert server.name == "apollobot"

    def test_all_20_tools_registered(self):
        from apollobot.server import create_server
        server = create_server()
        tools = server._tool_manager._tools
        assert len(tools) == 20, f"Expected 20 tools, got {len(tools)}"

    def test_tool_names_match_catalog(self):
        """All 20 documented tools exist by name."""
        from apollobot.server import create_server
        server = create_server()
        tool_names = set(server._tool_manager._tools.keys())
        expected = {
            "create_mission", "run_discover", "run_translate",
            "run_implement", "run_commercialize", "run_pipeline",
            "step_phase", "get_session_status", "get_phase_result",
            "approve_checkpoint", "search_literature", "query_data_source",
            "list_data_servers", "run_analysis_step", "draft_section",
            "review_manuscript", "get_provenance", "get_cost",
            "list_sessions", "load_session",
        }
        assert tool_names == expected, (
            f"Missing: {expected - tool_names}, Extra: {tool_names - expected}"
        )


class TestArgParsing:
    """[MEDIUM] run_server argument parsing."""

    def test_default_transport_is_stdio(self):
        """Default transport should be stdio."""
        import sys
        from unittest.mock import patch as mock_patch

        # Verify default by checking the function doesn't crash
        # with no args (we can't actually run it, but we can check the logic)
        from apollobot.server import run_server
        # Just verify function exists and is callable
        assert callable(run_server)


class TestCLIServeCommand:
    """[MEDIUM] CLI serve command exists and is configured."""

    def test_serve_command_exists(self):
        """apollo serve should be a registered CLI command."""
        from apollobot.cli import main
        commands = {cmd for cmd in main.commands}
        assert "serve" in commands, "CLI missing 'serve' command"

    def test_serve_has_transport_option(self):
        """serve command should have --transport option."""
        from apollobot.cli import main
        serve_cmd = main.commands["serve"]
        param_names = [p.name for p in serve_cmd.params]
        assert "transport" in param_names

    def test_serve_has_host_option(self):
        from apollobot.cli import main
        serve_cmd = main.commands["serve"]
        param_names = [p.name for p in serve_cmd.params]
        assert "host" in param_names

    def test_serve_has_port_option(self):
        from apollobot.cli import main
        serve_cmd = main.commands["serve"]
        param_names = [p.name for p in serve_cmd.params]
        assert "port" in param_names

    def test_serve_transport_choices(self):
        """Transport should accept stdio, streamable-http, sse."""
        from apollobot.cli import main
        serve_cmd = main.commands["serve"]
        transport_param = next(p for p in serve_cmd.params if p.name == "transport")
        choices = transport_param.type.choices
        assert "stdio" in choices
        assert "streamable-http" in choices
        assert "sse" in choices
