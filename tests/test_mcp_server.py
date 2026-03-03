"""Tests for the ApolloBot MCP server."""

import asyncio
import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from apollobot.core.mission import Mission, ResearchMode
from apollobot.core.session import Session, Phase, CostTracker
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
    """Create a mock ApolloConfig."""
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
    """Create a SessionRegistry with temp output dir."""
    mock_config.output_dir = str(temp_dir)
    return SessionRegistry(config=mock_config)


@pytest.fixture
def sample_mission(temp_dir):
    """Create a sample mission."""
    return Mission(
        objective="Test research objective",
        mode=ResearchMode.HYPOTHESIS,
        domain="bioinformatics",
        metadata={"output_dir": str(temp_dir)},
    )


@pytest.fixture
def mock_orchestrator():
    """Create a mock Orchestrator."""
    orch = MagicMock()
    orch.llm = MagicMock()
    orch.mcp = MagicMock()
    orch.checkpoint = MagicMock()
    orch.config = MagicMock()
    orch.router = MagicMock()
    orch.interactive = False
    return orch


@pytest.fixture
def mock_ctx(registry):
    """Create a mock MCP context with registry."""
    ctx = MagicMock()
    ctx.request_context.lifespan_context = {"registry": registry}
    return ctx


# ===========================================================================
# Error helpers
# ===========================================================================


class TestErrorResponse:
    """Tests for structured error responses."""

    def test_error_response_structure(self):
        """Error response has correct keys."""
        err = error_response(SESSION_NOT_FOUND, "Not found")
        assert err["error"] is True
        assert err["error_code"] == SESSION_NOT_FOUND
        assert err["error_message"] == "Not found"

    def test_error_codes_are_strings(self):
        """All error codes are uppercase strings."""
        codes = [SESSION_NOT_FOUND, PHASE_NOT_AVAILABLE, BUDGET_EXCEEDED,
                 INVALID_INPUT, INTERNAL_ERROR]
        for code in codes:
            assert isinstance(code, str)
            assert code == code.upper()


# ===========================================================================
# SessionRegistry
# ===========================================================================


class TestSessionRegistry:
    """Tests for the SessionRegistry."""

    async def test_create_session(self, registry, sample_mission):
        """Creating a session returns an ActiveSession."""
        with patch("apollobot.server.registry.Orchestrator") as MockOrch:
            MockOrch.return_value = MagicMock()
            active = await registry.create(sample_mission)

        assert active.session_id == sample_mission.id
        assert active.mission == sample_mission
        assert isinstance(active.session, Session)
        assert active.plan is None

    async def test_get_session(self, registry, sample_mission):
        """Getting a created session returns the same instance."""
        with patch("apollobot.server.registry.Orchestrator") as MockOrch:
            MockOrch.return_value = MagicMock()
            created = await registry.create(sample_mission)
            retrieved = await registry.get(sample_mission.id)

        assert retrieved is created

    async def test_get_missing_session_raises(self, registry):
        """Getting a non-existent session raises KeyError."""
        with pytest.raises(KeyError, match="Session not found"):
            await registry.get("nonexistent-id")

    async def test_list_active_sessions(self, registry, sample_mission):
        """List active returns summary dicts."""
        with patch("apollobot.server.registry.Orchestrator") as MockOrch:
            MockOrch.return_value = MagicMock()
            await registry.create(sample_mission)

        active_list = await registry.list_active()
        assert len(active_list) == 1
        assert active_list[0]["session_id"] == sample_mission.id
        assert active_list[0]["objective"] == sample_mission.objective
        assert active_list[0]["active"] is True

    async def test_remove_session(self, registry, sample_mission):
        """Removing a session makes it inaccessible."""
        with patch("apollobot.server.registry.Orchestrator") as MockOrch:
            MockOrch.return_value = MagicMock()
            await registry.create(sample_mission)
            await registry.remove(sample_mission.id)

        with pytest.raises(KeyError):
            await registry.get(sample_mission.id)

    async def test_load_from_disk(self, registry, sample_mission, temp_dir):
        """Loading a saved session from disk works."""
        # First create and save a session to disk
        session = Session(mission=sample_mission)
        session.init_directories()
        session.save_state()

        with patch("apollobot.server.registry.Orchestrator") as MockOrch:
            MockOrch.return_value = MagicMock()
            active = await registry.load_from_disk(sample_mission.id)

        assert active.session_id == sample_mission.id
        assert active.session.mission.objective == sample_mission.objective

    def test_list_historical_empty(self, registry, temp_dir):
        """List historical returns empty when no sessions on disk."""
        results = registry.list_historical()
        assert results == []

    def test_list_historical_finds_sessions(self, registry, sample_mission, temp_dir):
        """List historical finds saved sessions."""
        session = Session(mission=sample_mission)
        session.init_directories()
        session.save_state()

        results = registry.list_historical()
        assert len(results) == 1
        assert results[0]["session_id"] == sample_mission.id
        assert results[0]["active"] is False


# ===========================================================================
# MCPCheckpointHandler
# ===========================================================================


class TestMCPCheckpointHandler:
    """Tests for the MCP checkpoint handler."""

    async def test_auto_approve_without_session(self):
        """Without a session getter, auto-approves."""
        handler = MCPCheckpointHandler()
        result = await handler.request_approval("test_phase", "test summary")
        assert result is True

    async def test_pending_checkpoint_stored(self, sample_mission, temp_dir):
        """Pending checkpoint is stored on ActiveSession."""
        from apollobot.core.provenance import ProvenanceEngine

        session = Session(mission=sample_mission)
        session.init_directories()
        provenance = ProvenanceEngine(session.session_dir)

        active = ActiveSession(
            session_id=sample_mission.id,
            mission=sample_mission,
            session=session,
            orchestrator=MagicMock(),
            provenance=provenance,
        )

        handler = MCPCheckpointHandler(get_active_session=lambda: active)

        # Start the approval request in background
        async def request_and_check():
            return await handler.request_approval("analysis", "Approve analysis?")

        task = asyncio.create_task(request_and_check())

        # Give it a moment to set up the pending checkpoint
        await asyncio.sleep(0.05)
        assert active.pending_checkpoint is not None
        assert active.pending_checkpoint["phase"] == "analysis"

        # Approve it
        active.checkpoint_approved = True
        active.checkpoint_event.set()

        result = await task
        assert result is True
        assert active.pending_checkpoint is None

    async def test_checkpoint_denied(self, sample_mission, temp_dir):
        """Denying a checkpoint returns False."""
        from apollobot.core.provenance import ProvenanceEngine

        session = Session(mission=sample_mission)
        session.init_directories()
        provenance = ProvenanceEngine(session.session_dir)

        active = ActiveSession(
            session_id=sample_mission.id,
            mission=sample_mission,
            session=session,
            orchestrator=MagicMock(),
            provenance=provenance,
        )

        handler = MCPCheckpointHandler(get_active_session=lambda: active)

        async def request_and_check():
            return await handler.request_approval("analysis", "Approve?")

        task = asyncio.create_task(request_and_check())
        await asyncio.sleep(0.05)

        active.checkpoint_approved = False
        active.checkpoint_event.set()

        result = await task
        assert result is False

    async def test_notify_logs(self):
        """Notify doesn't raise."""
        handler = MCPCheckpointHandler()
        await handler.notify("test", "test summary")


# ===========================================================================
# Tool Tests (unit-level with mocked dependencies)
# ===========================================================================


class TestCreateMissionTool:
    """Tests for the create_mission tool."""

    async def test_create_mission_valid(self, mock_ctx):
        """Creating a mission with valid params succeeds."""
        from apollobot.server.app import create_mission

        with patch("apollobot.server.registry.Orchestrator") as MockOrch:
            MockOrch.return_value = MagicMock()
            result = await create_mission(
                mock_ctx,
                objective="Does X cause Y?",
                mode="hypothesis",
                domain="bioinformatics",
            )

        assert "session_id" in result
        assert result["objective"] == "Does X cause Y?"
        assert result["mode"] == "hypothesis"
        assert result["domain"] == "bioinformatics"
        assert "error" not in result

    async def test_create_mission_invalid_mode(self, mock_ctx):
        """Creating a mission with invalid mode returns error."""
        from apollobot.server.app import create_mission

        result = await create_mission(
            mock_ctx,
            objective="Test",
            mode="invalid_mode",
        )

        assert result["error"] is True
        assert result["error_code"] == INVALID_INPUT

    async def test_create_mission_from_yaml(self, mock_ctx):
        """Creating a mission from YAML string works."""
        from apollobot.server.app import create_mission

        yaml_str = """
objective: "YAML-based research"
mode: hypothesis
domain: physics
"""
        with patch("apollobot.server.registry.Orchestrator") as MockOrch:
            MockOrch.return_value = MagicMock()
            result = await create_mission(
                mock_ctx,
                objective="ignored",
                mission_yaml=yaml_str,
            )

        assert result["objective"] == "YAML-based research"
        assert result["domain"] == "physics"

    async def test_create_mission_custom_budget(self, mock_ctx):
        """Custom compute budget is set."""
        from apollobot.server.app import create_mission

        with patch("apollobot.server.registry.Orchestrator") as MockOrch:
            MockOrch.return_value = MagicMock()
            result = await create_mission(
                mock_ctx,
                objective="Test",
                compute_budget=100.0,
            )

        assert result["compute_budget"] == 100.0


class TestGetSessionStatusTool:
    """Tests for get_session_status tool."""

    async def test_status_valid_session(self, mock_ctx, registry, sample_mission):
        """Status returns correct fields for an active session."""
        from apollobot.server.app import get_session_status

        with patch("apollobot.server.registry.Orchestrator") as MockOrch:
            MockOrch.return_value = MagicMock()
            await registry.create(sample_mission)

        result = await get_session_status(mock_ctx, sample_mission.id)

        assert result["session_id"] == sample_mission.id
        assert result["current_phase"] == "planning"
        assert "cost" in result
        assert result["cost"]["total_usd"] == 0.0
        assert "warnings" in result
        assert result["pending_checkpoint"] is None

    async def test_status_missing_session(self, mock_ctx):
        """Status for non-existent session returns error."""
        from apollobot.server.app import get_session_status

        result = await get_session_status(mock_ctx, "nonexistent")

        assert result["error"] is True
        assert result["error_code"] == SESSION_NOT_FOUND


class TestGetPhaseResultTool:
    """Tests for get_phase_result tool."""

    async def test_phase_result_exists(self, mock_ctx, registry, sample_mission):
        """Getting a completed phase result returns data."""
        from apollobot.server.app import get_phase_result

        with patch("apollobot.server.registry.Orchestrator") as MockOrch:
            MockOrch.return_value = MagicMock()
            active = await registry.create(sample_mission)

        # Simulate a completed phase
        active.session.begin_phase(Phase.LITERATURE_REVIEW)
        active.session.complete_phase(
            Phase.LITERATURE_REVIEW,
            summary="Found 50 papers",
            findings=[{"type": "paper", "count": 50}],
        )

        result = await get_phase_result(mock_ctx, sample_mission.id, "literature_review")

        assert result["phase"] == "literature_review"
        assert result["summary"] == "Found 50 papers"
        assert len(result["findings"]) == 1

    async def test_phase_result_missing(self, mock_ctx, registry, sample_mission):
        """Getting a phase that hasn't run returns error."""
        from apollobot.server.app import get_phase_result

        with patch("apollobot.server.registry.Orchestrator") as MockOrch:
            MockOrch.return_value = MagicMock()
            await registry.create(sample_mission)

        result = await get_phase_result(mock_ctx, sample_mission.id, "analysis")

        assert result["error"] is True
        assert result["error_code"] == PHASE_NOT_AVAILABLE


class TestApproveCheckpointTool:
    """Tests for the approve_checkpoint tool."""

    async def test_approve_no_pending(self, mock_ctx, registry, sample_mission):
        """Approving with no pending checkpoint returns error."""
        from apollobot.server.app import approve_checkpoint

        with patch("apollobot.server.registry.Orchestrator") as MockOrch:
            MockOrch.return_value = MagicMock()
            await registry.create(sample_mission)

        result = await approve_checkpoint(mock_ctx, sample_mission.id)

        assert result["error"] is True
        assert result["error_code"] == PHASE_NOT_AVAILABLE

    async def test_approve_with_pending(self, mock_ctx, registry, sample_mission):
        """Approving a pending checkpoint succeeds."""
        from apollobot.server.app import approve_checkpoint

        with patch("apollobot.server.registry.Orchestrator") as MockOrch:
            MockOrch.return_value = MagicMock()
            active = await registry.create(sample_mission)

        # Simulate a pending checkpoint
        active.pending_checkpoint = {"phase": "analysis", "summary": "Continue?"}
        active.checkpoint_event = asyncio.Event()

        result = await approve_checkpoint(mock_ctx, sample_mission.id, approved=True)

        assert result["approved"] is True
        assert active.checkpoint_event.is_set()
        assert active.checkpoint_approved is True


class TestListDataServersTool:
    """Tests for list_data_servers tool."""

    async def test_list_all_domains(self, mock_ctx):
        """Listing all domains returns domain map."""
        from apollobot.server.app import list_data_servers

        result = await list_data_servers(mock_ctx)

        assert "domains" in result
        assert "bioinformatics" in result["domains"]

    async def test_list_specific_domain(self, mock_ctx):
        """Listing a specific domain returns servers."""
        from apollobot.server.app import list_data_servers

        result = await list_data_servers(mock_ctx, domain="bioinformatics")

        assert result["domain"] == "bioinformatics"
        assert "servers" in result
        assert len(result["servers"]) > 0
        assert "name" in result["servers"][0]
        assert "description" in result["servers"][0]


class TestGetCostTool:
    """Tests for get_cost tool."""

    async def test_cost_initial(self, mock_ctx, registry, sample_mission):
        """Initial cost is zero."""
        from apollobot.server.app import get_cost

        with patch("apollobot.server.registry.Orchestrator") as MockOrch:
            MockOrch.return_value = MagicMock()
            await registry.create(sample_mission)

        result = await get_cost(mock_ctx, sample_mission.id)

        assert result["total_cost_usd"] == 0.0
        assert result["llm_calls"] == 0
        assert result["budget_total"] == 50.0
        assert result["budget_remaining"] == 50.0

    async def test_cost_after_calls(self, mock_ctx, registry, sample_mission):
        """Cost reflects recorded LLM calls."""
        from apollobot.server.app import get_cost

        with patch("apollobot.server.registry.Orchestrator") as MockOrch:
            MockOrch.return_value = MagicMock()
            active = await registry.create(sample_mission)

        active.session.cost.record_llm_call(1000, 500, 0.05)
        active.session.cost.record_llm_call(2000, 1000, 0.10)

        result = await get_cost(mock_ctx, sample_mission.id)

        assert result["total_cost_usd"] == pytest.approx(0.15)
        assert result["llm_calls"] == 2
        assert result["budget_remaining"] == pytest.approx(49.85)


class TestListSessionsTool:
    """Tests for list_sessions tool."""

    async def test_list_empty(self, mock_ctx):
        """Listing when no sessions exist returns empty lists."""
        from apollobot.server.app import list_sessions

        result = await list_sessions(mock_ctx)

        assert result["total_active"] == 0

    async def test_list_with_active(self, mock_ctx, registry, sample_mission):
        """Listing with an active session returns it."""
        from apollobot.server.app import list_sessions

        with patch("apollobot.server.registry.Orchestrator") as MockOrch:
            MockOrch.return_value = MagicMock()
            await registry.create(sample_mission)

        result = await list_sessions(mock_ctx)

        assert result["total_active"] == 1
        assert result["active"][0]["session_id"] == sample_mission.id


class TestGetProvenanceTool:
    """Tests for get_provenance tool."""

    async def test_provenance_initial(self, mock_ctx, registry, sample_mission):
        """Initial provenance has session_started event."""
        from apollobot.server.app import get_provenance

        with patch("apollobot.server.registry.Orchestrator") as MockOrch:
            MockOrch.return_value = MagicMock()
            await registry.create(sample_mission)

        result = await get_provenance(mock_ctx, sample_mission.id)

        assert result["session_id"] == sample_mission.id
        assert result["total_events"] >= 1  # At least session_started


class TestStepPhaseTool:
    """Tests for step_phase tool."""

    async def test_step_budget_exceeded(self, mock_ctx, registry, sample_mission):
        """Step phase returns error when budget is exceeded."""
        from apollobot.server.app import step_phase

        with patch("apollobot.server.registry.Orchestrator") as MockOrch:
            MockOrch.return_value = MagicMock()
            active = await registry.create(sample_mission)

        # Exceed the budget
        active.session.cost.estimated_cost_usd = 999.0

        result = await step_phase(mock_ctx, sample_mission.id)

        assert result["error"] is True
        assert result["error_code"] == BUDGET_EXCEEDED

    async def test_step_invalid_phase(self, mock_ctx, registry, sample_mission):
        """Step with invalid phase name returns error."""
        from apollobot.server.app import step_phase

        with patch("apollobot.server.registry.Orchestrator") as MockOrch:
            MockOrch.return_value = MagicMock()
            await registry.create(sample_mission)

        result = await step_phase(mock_ctx, sample_mission.id, phase="not_a_phase")

        assert result["error"] is True
        assert result["error_code"] == INVALID_INPUT

    async def test_step_needs_plan_for_non_planning(self, mock_ctx, registry, sample_mission):
        """Stepping to literature_review without a plan returns error."""
        from apollobot.server.app import step_phase

        with patch("apollobot.server.registry.Orchestrator") as MockOrch:
            MockOrch.return_value = MagicMock()
            active = await registry.create(sample_mission)

        # Mark planning as done but don't set a plan
        active.session.begin_phase(Phase.PLANNING)
        active.session.complete_phase(Phase.PLANNING, summary="Planned")

        result = await step_phase(mock_ctx, sample_mission.id, phase="literature_review")

        assert result["error"] is True
        assert result["error_code"] == PHASE_NOT_AVAILABLE


class TestSearchLiteratureTool:
    """Tests for search_literature tool."""

    async def test_search_returns_results(self, mock_ctx):
        """Search with mocked MCP client returns papers."""
        from apollobot.server.app import search_literature

        mock_papers = [
            {"title": "Paper A", "doi": "10.1/a", "year": "2024"},
            {"title": "Paper B", "doi": "10.1/b", "year": "2023"},
        ]

        with patch("apollobot.mcp.MCPClient") as MockClient:
            client = MockClient.return_value
            mock_server = MagicMock()
            mock_server.name = "pubmed"
            mock_server.domain = "shared"
            client.get_servers.return_value = [mock_server]
            client.query = AsyncMock(return_value={"papers": mock_papers})

            result = await search_literature(
                mock_ctx, query="gut microbiome", domain="bioinformatics"
            )

        assert result["total_found"] == 2
        assert len(result["papers"]) == 2
        assert result["papers"][0]["title"] == "Paper A"


class TestLoadSessionTool:
    """Tests for load_session tool."""

    async def test_load_nonexistent(self, mock_ctx):
        """Loading a non-existent session returns error."""
        from apollobot.server.app import load_session

        result = await load_session(mock_ctx, "nonexistent-session-xyz")

        assert result["error"] is True

    async def test_load_existing(self, mock_ctx, registry, sample_mission, temp_dir):
        """Loading an existing session from disk works."""
        from apollobot.server.app import load_session

        # Save a session to disk
        session = Session(mission=sample_mission)
        session.init_directories()
        session.save_state()

        with patch("apollobot.server.registry.Orchestrator") as MockOrch:
            MockOrch.return_value = MagicMock()
            result = await load_session(mock_ctx, sample_mission.id)

        assert result["session_id"] == sample_mission.id
        assert "error" not in result
