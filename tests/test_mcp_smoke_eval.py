"""
MCP Server — Smoke & Limited Evaluation Tests
===============================================

Smoke tests (marked @pytest.mark.smoke):
  Fast sanity checks that verify the server boots, all 20 tools are
  importable and callable, and basic I/O contracts hold.  Designed to
  run in <2 seconds with no external dependencies.

Limited evaluation tests (marked @pytest.mark.evaluation):
  Deeper functional flows that exercise realistic multi-step scenarios
  with mocked LLM/API backends.  Tests verify session lifecycle,
  checkpoint orchestration, cost tracking accuracy, concurrent sessions,
  search dedup, and error recovery.

Run individually:
  pytest tests/test_mcp_smoke_eval.py -m smoke -v
  pytest tests/test_mcp_smoke_eval.py -m evaluation -v
"""

from __future__ import annotations

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
# Shared fixtures
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
    with patch("apollobot.server.registry.Orchestrator") as MockOrch:
        MockOrch.return_value = MagicMock()
        return await registry.create(sample_mission)


# ===========================================================================
# SMOKE TESTS — fast "does it boot?" checks
# ===========================================================================


@pytest.mark.smoke
class TestSmoke_Imports:
    """All server modules import without error."""

    def test_import_server_package(self):
        from apollobot.server import create_server, run_server
        assert callable(create_server)
        assert callable(run_server)

    def test_import_app_module(self):
        from apollobot.server.app import mcp, app_lifespan
        assert mcp is not None
        assert callable(app_lifespan)

    def test_import_registry(self):
        from apollobot.server.registry import SessionRegistry, ActiveSession
        assert callable(SessionRegistry)

    def test_import_checkpoint(self):
        from apollobot.server.checkpoint import MCPCheckpointHandler
        assert callable(MCPCheckpointHandler)

    def test_import_errors(self):
        from apollobot.server.errors import error_response
        err = error_response("TEST", "test msg")
        assert err["error"] is True
        assert err["error_code"] == "TEST"


@pytest.mark.smoke
class TestSmoke_ServerBoot:
    """Server can be constructed and has the correct shape."""

    def test_server_name(self):
        from apollobot.server import create_server
        server = create_server()
        assert server.name == "apollobot"

    def test_server_has_instructions(self):
        from apollobot.server import create_server
        server = create_server()
        assert "ApolloBot" in (server.instructions or "")

    def test_exactly_20_tools(self):
        from apollobot.server import create_server
        server = create_server()
        tools = server._tool_manager._tools
        assert len(tools) == 20, f"Expected 20 tools, got {len(tools)}: {list(tools.keys())}"

    def test_all_tool_names_present(self):
        from apollobot.server import create_server
        server = create_server()
        names = set(server._tool_manager._tools.keys())
        expected = {
            "create_mission", "run_discover", "run_translate",
            "run_implement", "run_commercialize", "run_pipeline",
            "step_phase", "get_session_status", "get_phase_result",
            "approve_checkpoint", "search_literature", "query_data_source",
            "list_data_servers", "run_analysis_step", "draft_section",
            "review_manuscript", "get_provenance", "get_cost",
            "list_sessions", "load_session",
        }
        missing = expected - names
        extra = names - expected
        assert names == expected, f"Missing: {missing}, Extra: {extra}"

    def test_lifespan_creates_registry(self):
        from apollobot.server.app import app_lifespan, mcp
        async def _check():
            with patch("apollobot.server.app._load_config") as mock_cfg:
                mock_cfg.return_value = MagicMock()
                async with app_lifespan(mcp) as ctx:
                    assert "registry" in ctx
                    assert isinstance(ctx["registry"], SessionRegistry)
        asyncio.get_event_loop().run_until_complete(_check())


@pytest.mark.smoke
class TestSmoke_CLIServe:
    """CLI 'serve' command exists with correct options."""

    def test_serve_command_registered(self):
        from apollobot.cli import main
        assert "serve" in main.commands

    def test_serve_transport_choices(self):
        from apollobot.cli import main
        serve_cmd = main.commands["serve"]
        transport = next(p for p in serve_cmd.params if p.name == "transport")
        assert set(transport.type.choices) == {"stdio", "streamable-http", "sse"}

    def test_serve_default_transport_is_stdio(self):
        from apollobot.cli import main
        serve_cmd = main.commands["serve"]
        transport = next(p for p in serve_cmd.params if p.name == "transport")
        assert transport.default == "stdio"

    def test_serve_has_host_and_port(self):
        from apollobot.cli import main
        serve_cmd = main.commands["serve"]
        param_names = {p.name for p in serve_cmd.params}
        assert "host" in param_names
        assert "port" in param_names


@pytest.mark.smoke
class TestSmoke_EveryToolCallable:
    """Every tool can be invoked with minimal args and returns a dict
    (success or structured error), never crashes."""

    async def test_create_mission(self, mock_ctx):
        from apollobot.server.app import create_mission
        with patch("apollobot.server.registry.Orchestrator"):
            result = await create_mission(mock_ctx, objective="Smoke test")
        assert isinstance(result, dict)
        assert "session_id" in result

    async def test_run_discover(self, mock_ctx):
        from apollobot.server.app import run_discover
        result = await run_discover(mock_ctx, session_id="no-such-id")
        assert isinstance(result, dict)
        assert result["error"] is True

    async def test_run_translate(self, mock_ctx):
        from apollobot.server.app import run_translate
        result = await run_translate(mock_ctx, session_id="no-such-id")
        assert isinstance(result, dict)
        assert result["error"] is True

    async def test_run_implement(self, mock_ctx):
        from apollobot.server.app import run_implement
        result = await run_implement(mock_ctx, session_id="no-such-id")
        assert isinstance(result, dict)
        assert result["error"] is True

    async def test_run_commercialize(self, mock_ctx):
        from apollobot.server.app import run_commercialize
        result = await run_commercialize(mock_ctx, session_id="no-such-id")
        assert isinstance(result, dict)
        assert result["error"] is True

    async def test_run_pipeline(self, mock_ctx):
        from apollobot.server.app import run_pipeline
        result = await run_pipeline(mock_ctx, session_id="no-such-id")
        assert isinstance(result, dict)
        assert result["error"] is True

    async def test_step_phase(self, mock_ctx):
        from apollobot.server.app import step_phase
        result = await step_phase(mock_ctx, session_id="no-such-id")
        assert isinstance(result, dict)
        assert result["error"] is True

    async def test_get_session_status(self, mock_ctx):
        from apollobot.server.app import get_session_status
        result = await get_session_status(mock_ctx, session_id="no-such-id")
        assert isinstance(result, dict)
        assert result["error"] is True

    async def test_get_phase_result(self, mock_ctx):
        from apollobot.server.app import get_phase_result
        result = await get_phase_result(mock_ctx, "no-such-id", "analysis")
        assert isinstance(result, dict)
        assert result["error"] is True

    async def test_approve_checkpoint(self, mock_ctx):
        from apollobot.server.app import approve_checkpoint
        result = await approve_checkpoint(mock_ctx, "no-such-id")
        assert isinstance(result, dict)
        assert result["error"] is True

    async def test_search_literature(self, mock_ctx):
        from apollobot.server.app import search_literature
        with patch("apollobot.mcp.MCPClient") as MockClient:
            client = MockClient.return_value
            client.get_servers.return_value = []
            result = await search_literature(mock_ctx, query="test")
        assert isinstance(result, dict)
        assert "papers" in result

    async def test_query_data_source(self, mock_ctx):
        from apollobot.server.app import query_data_source
        with patch("apollobot.mcp.MCPClient") as MockClient:
            client = MockClient.return_value
            client.get_servers.return_value = []
            result = await query_data_source(mock_ctx, server_name="x", capability="y")
        assert isinstance(result, dict)

    async def test_list_data_servers(self, mock_ctx):
        from apollobot.server.app import list_data_servers
        result = await list_data_servers(mock_ctx)
        assert isinstance(result, dict)
        assert "domains" in result

    async def test_run_analysis_step(self, mock_ctx):
        from apollobot.server.app import run_analysis_step
        result = await run_analysis_step(
            mock_ctx, session_id="no-such-id", step_name="x", description="y"
        )
        assert isinstance(result, dict)
        assert result["error"] is True

    async def test_draft_section(self, mock_ctx):
        from apollobot.server.app import draft_section
        result = await draft_section(mock_ctx, session_id="no-such-id", section="abstract")
        assert isinstance(result, dict)
        assert result["error"] is True

    async def test_review_manuscript(self, mock_ctx):
        from apollobot.server.app import review_manuscript
        result = await review_manuscript(mock_ctx, "no-such-id")
        assert isinstance(result, dict)
        assert result["error"] is True

    async def test_get_provenance(self, mock_ctx):
        from apollobot.server.app import get_provenance
        result = await get_provenance(mock_ctx, "no-such-id")
        assert isinstance(result, dict)
        assert result["error"] is True

    async def test_get_cost(self, mock_ctx):
        from apollobot.server.app import get_cost
        result = await get_cost(mock_ctx, "no-such-id")
        assert isinstance(result, dict)
        assert result["error"] is True

    async def test_list_sessions(self, mock_ctx):
        from apollobot.server.app import list_sessions
        result = await list_sessions(mock_ctx)
        assert isinstance(result, dict)
        assert "total_active" in result

    async def test_load_session(self, mock_ctx):
        from apollobot.server.app import load_session
        result = await load_session(mock_ctx, "no-such-id")
        assert isinstance(result, dict)
        assert result["error"] is True


# ===========================================================================
# LIMITED EVALUATION TESTS — deeper functional scenarios
# ===========================================================================


@pytest.mark.evaluation
class TestEval_FullSessionLifecycle:
    """End-to-end: create → status → cost → provenance → list → phase_result."""

    async def test_lifecycle(self, mock_ctx, registry):
        from apollobot.server.app import (
            create_mission, get_session_status, get_cost,
            get_provenance, list_sessions,
        )

        # 1. Create mission
        with patch("apollobot.server.registry.Orchestrator"):
            created = await create_mission(
                mock_ctx,
                objective="Lifecycle evaluation: gut microbiome and aging",
                mode="hypothesis",
                domain="bioinformatics",
                compute_budget=25.0,
            )
        assert "error" not in created
        sid = created["session_id"]
        assert created["objective"] == "Lifecycle evaluation: gut microbiome and aging"
        assert created["compute_budget"] == 25.0

        # 2. Get status — should be at planning phase
        status = await get_session_status(mock_ctx, sid)
        assert status["current_phase"] == "planning"
        assert status["has_plan"] is False
        assert status["pending_checkpoint"] is None
        assert status["cost"]["total_usd"] == 0.0
        assert len(status["completed_phases"]) == 0

        # 3. Get cost — zero
        cost = await get_cost(mock_ctx, sid)
        assert cost["total_cost_usd"] == 0.0
        assert cost["budget_total"] == 25.0
        assert cost["budget_remaining"] == 25.0
        assert cost["budget_utilization_pct"] == 0.0
        assert cost["llm_calls"] == 0

        # 4. Simulate some LLM cost, re-check
        active = await registry.get(sid)
        active.session.cost.record_llm_call(5000, 2000, 0.12)
        active.session.cost.record_llm_call(3000, 1500, 0.08)

        cost2 = await get_cost(mock_ctx, sid)
        assert cost2["llm_calls"] == 2
        assert cost2["total_cost_usd"] == pytest.approx(0.20)
        assert cost2["llm_input_tokens"] == 8000
        assert cost2["llm_output_tokens"] == 3500
        assert cost2["budget_remaining"] == pytest.approx(24.80)
        assert cost2["budget_utilization_pct"] == pytest.approx(0.80)
        assert cost2["avg_cost_per_call"] == pytest.approx(0.10)

        # 5. Provenance — has session_started event
        prov = await get_provenance(mock_ctx, sid)
        assert prov["total_events"] >= 1
        events = [e.get("event") for e in prov["execution_log"]]
        assert "session_started" in events
        assert prov["session_id"] == sid

        # 6. List sessions — our session appears
        sessions = await list_sessions(mock_ctx)
        assert sessions["total_active"] >= 1
        active_ids = [s["session_id"] for s in sessions["active"]]
        assert sid in active_ids


@pytest.mark.evaluation
class TestEval_PlanningPhaseStepThrough:
    """Step through the planning phase with a mocked LLM planner."""

    async def test_planning_step(self, mock_ctx, registry):
        from apollobot.server.app import create_mission, step_phase, get_session_status

        # Create session
        with patch("apollobot.server.registry.Orchestrator") as MockOrch:
            mock_orch = MagicMock()
            mock_orch.mcp = MagicMock()
            mock_server = MagicMock()
            mock_server.name = "pubmed"
            mock_orch.mcp.get_servers.return_value = [mock_server]
            mock_orch._connect_mcp_servers = AsyncMock()
            MockOrch.return_value = mock_orch

            created = await create_mission(
                mock_ctx, objective="Effect of sleep on memory consolidation",
                mode="hypothesis", domain="bioinformatics",
            )

        sid = created["session_id"]

        # Mock the ResearchPlanner to return a plan
        mock_plan = MagicMock()
        mock_plan.summary = "Investigate sleep duration effects on memory via meta-analysis"
        mock_plan.approach = "Meta-analysis of polysomnography + cognitive assessment datasets"
        mock_plan.literature_queries = ["sleep memory consolidation", "polysomnography cognition"]
        mock_plan.data_requirements = [{"type": "geo", "query": "sleep EEG"}]
        mock_plan.analysis_steps = [
            MagicMock(name="correlation_analysis"),
            MagicMock(name="meta_regression"),
        ]
        mock_plan.estimated_compute_cost = 15.0
        mock_plan.estimated_time_hours = 2.5

        with patch("apollobot.agents.planner.ResearchPlanner") as MockPlanner:
            MockPlanner.return_value.plan = AsyncMock(return_value=mock_plan)
            result = await step_phase(mock_ctx, sid)

        assert result["phase"] == "planning"
        assert result["status"] == "completed"
        assert "sleep" in result["plan_summary"].lower() or "memory" in result["plan_summary"].lower()
        assert result["next_phase"] == "literature_review"
        assert result["analysis_steps_count"] == 2
        assert result["data_requirements_count"] == 1
        assert result["estimated_cost"] == 15.0

        # Verify status updated
        status = await get_session_status(mock_ctx, sid)
        assert status["has_plan"] is True
        assert "planning" in [p if isinstance(p, str) else p.value for p in status["completed_phases"]]


@pytest.mark.evaluation
class TestEval_CheckpointOrchestration:
    """Full checkpoint flow: pending → status shows it → approve → cleared."""

    async def test_checkpoint_approve_flow(self, mock_ctx, active_session):
        from apollobot.server.app import get_session_status, approve_checkpoint

        sid = active_session.session_id

        # 1. Initially no checkpoint
        status = await get_session_status(mock_ctx, sid)
        assert status["pending_checkpoint"] is None

        # 2. Simulate pipeline checkpoint
        active_session.pending_checkpoint = {
            "phase": "pipeline_translate",
            "summary": "Discovery complete — translation score: 8.2. Proceed to Translate?"
        }
        active_session.checkpoint_event = asyncio.Event()

        # 3. Status reflects pending checkpoint
        status2 = await get_session_status(mock_ctx, sid)
        assert status2["pending_checkpoint"] is not None
        assert status2["pending_checkpoint"]["phase"] == "pipeline_translate"
        assert "8.2" in status2["pending_checkpoint"]["summary"]

        # 4. Approve the checkpoint
        result = await approve_checkpoint(mock_ctx, sid, approved=True)
        assert result["approved"] is True
        assert result["checkpoint"]["phase"] == "pipeline_translate"
        assert active_session.checkpoint_event.is_set()
        assert active_session.checkpoint_approved is True

        # 5. pending_checkpoint is cleared by MCPCheckpointHandler.request_approval
        #    (running in a separate coroutine), not by approve_checkpoint itself.
        #    The tool sets the event; the handler clears the checkpoint after waking.
        #    Simulate what the handler does after the event fires:
        active_session.pending_checkpoint = None  # handler does this

        status3 = await get_session_status(mock_ctx, sid)
        assert status3["pending_checkpoint"] is None

    async def test_checkpoint_deny_flow(self, mock_ctx, active_session):
        from apollobot.server.app import approve_checkpoint

        active_session.pending_checkpoint = {
            "phase": "pipeline_implement",
            "summary": "Translate complete. Proceed to Implement?",
        }
        active_session.checkpoint_event = asyncio.Event()

        result = await approve_checkpoint(mock_ctx, active_session.session_id, approved=False)
        assert result["approved"] is False
        assert "denied" in result["message"].lower() or "stop" in result["message"].lower()
        assert active_session.checkpoint_approved is False
        assert active_session.checkpoint_event.is_set()

    async def test_checkpoint_no_pending_returns_error(self, mock_ctx, active_session):
        from apollobot.server.app import approve_checkpoint
        result = await approve_checkpoint(mock_ctx, active_session.session_id)
        assert result["error"] is True
        assert result["error_code"] == PHASE_NOT_AVAILABLE


@pytest.mark.evaluation
class TestEval_LiteratureSearchRealistic:
    """Realistic multi-server search with dedup and filtering."""

    async def test_multi_server_search_with_dedup(self, mock_ctx):
        from apollobot.server.app import search_literature

        pubmed_papers = [
            {"title": "CRISPR gene editing review", "doi": "10.1/crispr1", "year": "2024", "source": "pubmed"},
            {"title": "Base editing advances", "doi": "10.1/base1", "year": "2023", "source": "pubmed"},
            {"title": "Prime editing efficiency", "doi": "10.1/prime1", "year": "2024", "source": "pubmed"},
        ]
        arxiv_papers = [
            {"title": "CRISPR gene editing review", "doi": "10.1/crispr1", "year": "2024", "source": "arxiv"},  # dupe
            {"title": "ML-guided gene editing", "doi": "10.1/ml1", "year": "2025", "source": "arxiv"},
        ]

        call_count = 0

        async def mock_query(server_name, capability, params):
            nonlocal call_count
            call_count += 1
            if server_name == "pubmed":
                return {"papers": pubmed_papers}
            elif server_name == "arxiv":
                return {"papers": arxiv_papers}
            return {"papers": []}

        with patch("apollobot.mcp.MCPClient") as MockClient:
            client = MockClient.return_value
            mock_pubmed = MagicMock()
            mock_pubmed.name = "pubmed"
            mock_pubmed.domain = "shared"
            mock_arxiv = MagicMock()
            mock_arxiv.name = "arxiv"
            mock_arxiv.domain = "shared"
            client.get_servers.return_value = [mock_pubmed, mock_arxiv]
            client.query = mock_query

            result = await search_literature(
                mock_ctx, query="CRISPR gene editing", domain="bioinformatics"
            )

        assert result["query"] == "CRISPR gene editing"
        assert set(result["servers_searched"]) == {"pubmed", "arxiv"}
        # 5 total papers, 1 duplicate (same DOI) → 4 unique
        assert result["total_found"] == 4
        assert len(result["papers"]) == 4
        # Verify the duplicate was removed (only one copy of crispr1)
        dois = [p["doi"] for p in result["papers"]]
        assert dois.count("10.1/crispr1") == 1

    async def test_search_with_server_failure(self, mock_ctx):
        """One server fails, others still return results."""
        from apollobot.server.app import search_literature

        async def mock_query(server_name, capability, params):
            if server_name == "pubmed":
                raise ConnectionError("pubmed unreachable")
            return {"papers": [{"title": "ArXiv paper", "doi": "10.1/a"}]}

        with patch("apollobot.mcp.MCPClient") as MockClient:
            client = MockClient.return_value
            mock_pubmed = MagicMock()
            mock_pubmed.name = "pubmed"
            mock_pubmed.domain = "shared"
            mock_arxiv = MagicMock()
            mock_arxiv.name = "arxiv"
            mock_arxiv.domain = "shared"
            client.get_servers.return_value = [mock_pubmed, mock_arxiv]
            client.query = mock_query

            result = await search_literature(mock_ctx, query="test")

        # Should still succeed with results from arxiv
        assert result["total_found"] == 1
        assert "arxiv" in result["servers_searched"]
        # pubmed failed silently, not in servers_searched
        assert "pubmed" not in result["servers_searched"]

    async def test_search_papers_with_no_identifiers(self, mock_ctx):
        """Papers without DOI or title should all be kept."""
        from apollobot.server.app import search_literature

        papers = [
            {"abstract": "Content A about stem cells"},
            {"abstract": "Content B about gene therapy"},
            {"abstract": "Content C about immunology"},
        ]

        with patch("apollobot.mcp.MCPClient") as MockClient:
            client = MockClient.return_value
            mock_server = MagicMock()
            mock_server.name = "pubmed"
            mock_server.domain = "shared"
            client.get_servers.return_value = [mock_server]
            client.query = AsyncMock(return_value={"papers": papers})

            result = await search_literature(mock_ctx, query="test")

        assert result["total_found"] == 3


@pytest.mark.evaluation
class TestEval_DataServerDiscovery:
    """list_data_servers across all domains and consistency checks."""

    async def test_all_domains_have_servers(self, mock_ctx):
        from apollobot.server.app import list_data_servers

        result = await list_data_servers(mock_ctx)
        assert "domains" in result
        for domain_name, servers in result["domains"].items():
            assert len(servers) > 0, f"Domain '{domain_name}' has no servers"
            for srv in servers:
                assert "name" in srv
                assert "description" in srv
                assert "domain" in srv
                assert "category" in srv
                assert "requires_key" in srv

    async def test_per_domain_matches_all_domains(self, mock_ctx):
        """Per-domain query should return same servers as all-domains query."""
        from apollobot.server.app import list_data_servers

        all_result = await list_data_servers(mock_ctx)
        for domain_name in all_result["domains"]:
            per_domain = await list_data_servers(mock_ctx, domain=domain_name)
            assert per_domain["domain"] == domain_name
            all_names = {s["name"] for s in all_result["domains"][domain_name]}
            per_names = {s["name"] for s in per_domain["servers"]}
            assert all_names == per_names, (
                f"Mismatch for domain '{domain_name}': "
                f"all={all_names}, per={per_names}"
            )

    async def test_known_domains_present(self, mock_ctx):
        from apollobot.server.app import list_data_servers
        result = await list_data_servers(mock_ctx)
        known = {"bioinformatics", "physics", "cs_ml", "comp_chem", "economics"}
        actual = set(result["domains"].keys())
        missing = known - actual
        assert not missing, f"Expected domains missing: {missing}"


@pytest.mark.evaluation
class TestEval_SessionSaveLoadRoundtrip:
    """Save session to disk, then load it back via load_session tool."""

    async def test_roundtrip(self, mock_ctx, registry, sample_mission, temp_dir):
        from apollobot.server.app import (
            create_mission, get_session_status, load_session, list_sessions,
        )

        # Create and populate a session
        with patch("apollobot.server.registry.Orchestrator"):
            created = await create_mission(
                mock_ctx, objective="Roundtrip test", domain="physics",
                compute_budget=30.0,
            )
        sid = created["session_id"]

        # Add some state: complete a phase, record costs
        active = await registry.get(sid)
        active.session.begin_phase(Phase.PLANNING)
        active.session.complete_phase(Phase.PLANNING, summary="Planned research")
        active.session.cost.record_llm_call(1000, 500, 0.05)
        active.session.save_state()

        # Remove from active memory
        await registry.remove(sid)

        # Verify it's gone from active
        sessions = await list_sessions(mock_ctx)
        active_ids = [s["session_id"] for s in sessions["active"]]
        assert sid not in active_ids

        # It should appear in historical
        assert sessions["total_historical"] >= 1
        hist_ids = [s["session_id"] for s in sessions["historical"]]
        assert sid in hist_ids

        # Load it back
        with patch("apollobot.server.registry.Orchestrator"):
            loaded = await load_session(mock_ctx, sid)

        assert loaded["session_id"] == sid
        assert loaded["objective"] == "Roundtrip test"
        assert loaded["domain"] == "physics"

        # Verify it's back in active
        sessions2 = await list_sessions(mock_ctx)
        active_ids2 = [s["session_id"] for s in sessions2["active"]]
        assert sid in active_ids2


@pytest.mark.evaluation
class TestEval_CostTrackingAccuracy:
    """Cost numbers stay accurate across many operations."""

    async def test_incremental_cost_tracking(self, mock_ctx, active_session):
        from apollobot.server.app import get_cost

        sid = active_session.session_id
        cost = active_session.session.cost

        # Record 10 LLM calls with varying costs
        expected_total = 0.0
        expected_input_tokens = 0
        expected_output_tokens = 0
        for i in range(10):
            inp = (i + 1) * 1000
            out = (i + 1) * 500
            usd = (i + 1) * 0.01
            cost.record_llm_call(inp, out, usd)
            expected_total += usd
            expected_input_tokens += inp
            expected_output_tokens += out

        result = await get_cost(mock_ctx, sid)
        assert result["llm_calls"] == 10
        assert result["total_cost_usd"] == pytest.approx(expected_total)
        assert result["llm_input_tokens"] == expected_input_tokens
        assert result["llm_output_tokens"] == expected_output_tokens
        assert result["avg_cost_per_call"] == pytest.approx(expected_total / 10)
        assert result["budget_remaining"] == pytest.approx(50.0 - expected_total)
        assert result["budget_utilization_pct"] == pytest.approx(expected_total / 50.0 * 100)

    async def test_budget_enforcement_boundary(self, mock_ctx, active_session):
        """Step phase should reject when budget is exceeded."""
        from apollobot.server.app import step_phase

        sid = active_session.session_id
        active_session.session.cost.estimated_cost_usd = 50.01

        result = await step_phase(mock_ctx, sid)
        assert result["error"] is True
        assert result["error_code"] == BUDGET_EXCEEDED


@pytest.mark.evaluation
class TestEval_ConcurrentSessions:
    """Multiple sessions can coexist in the registry."""

    async def test_three_concurrent_sessions(self, mock_ctx, registry, temp_dir):
        from apollobot.server.app import create_mission, get_session_status, list_sessions

        sids = []
        domains = ["bioinformatics", "physics", "economics"]
        objectives = [
            "Gene editing efficiency in cancer cells",
            "Dark matter distribution in galaxy clusters",
            "Impact of central bank policy on inflation expectations",
        ]

        # Create 3 sessions
        for obj, dom in zip(objectives, domains):
            with patch("apollobot.server.registry.Orchestrator"):
                result = await create_mission(
                    mock_ctx, objective=obj, domain=dom,
                )
            assert "error" not in result
            sids.append(result["session_id"])

        # Verify all 3 exist
        sessions = await list_sessions(mock_ctx)
        assert sessions["total_active"] >= 3
        listed_ids = {s["session_id"] for s in sessions["active"]}
        for sid in sids:
            assert sid in listed_ids

        # Verify each has correct metadata
        for i, sid in enumerate(sids):
            status = await get_session_status(mock_ctx, sid)
            assert status["objective"] == objectives[i]
            assert status["domain"] == domains[i]

        # Modify cost on one, verify others unaffected
        active = await registry.get(sids[1])
        active.session.cost.record_llm_call(5000, 2000, 1.50)

        from apollobot.server.app import get_cost
        cost0 = await get_cost(mock_ctx, sids[0])
        cost1 = await get_cost(mock_ctx, sids[1])
        cost2 = await get_cost(mock_ctx, sids[2])

        assert cost0["total_cost_usd"] == 0.0
        assert cost1["total_cost_usd"] == pytest.approx(1.50)
        assert cost2["total_cost_usd"] == 0.0


@pytest.mark.evaluation
class TestEval_ErrorRecovery:
    """Tools recover gracefully from various internal failures."""

    async def test_create_mission_malformed_yaml(self, mock_ctx):
        from apollobot.server.app import create_mission
        result = await create_mission(
            mock_ctx, objective="Test", mission_yaml="{{bad yaml::\n  ]]]"
        )
        assert result["error"] is True
        assert result["error_code"] == INTERNAL_ERROR

    async def test_step_phase_invalid_name(self, mock_ctx, active_session):
        from apollobot.server.app import step_phase
        result = await step_phase(mock_ctx, active_session.session_id, phase="not_real")
        assert result["error"] is True
        assert result["error_code"] == INVALID_INPUT

    async def test_step_phase_non_discover_phase(self, mock_ctx, active_session):
        from apollobot.server.app import step_phase
        result = await step_phase(mock_ctx, active_session.session_id, phase="translate_assess")
        assert result["error"] is True
        assert result["error_code"] == INVALID_INPUT

    async def test_get_phase_result_not_run(self, mock_ctx, active_session):
        from apollobot.server.app import get_phase_result
        result = await get_phase_result(mock_ctx, active_session.session_id, "analysis")
        assert result["error"] is True
        assert result["error_code"] == PHASE_NOT_AVAILABLE

    async def test_draft_section_invalid_section(self, mock_ctx, active_session):
        from apollobot.server.app import draft_section
        result = await draft_section(
            mock_ctx, active_session.session_id, section="bibliography"
        )
        assert result["error"] is True
        assert result["error_code"] == INVALID_INPUT

    async def test_review_manuscript_no_plan(self, mock_ctx, active_session):
        from apollobot.server.app import review_manuscript
        result = await review_manuscript(mock_ctx, active_session.session_id)
        assert result["error"] is True
        assert result["error_code"] == PHASE_NOT_AVAILABLE

    async def test_create_mission_negative_budget(self, mock_ctx):
        from apollobot.server.app import create_mission
        with patch("apollobot.server.registry.Orchestrator"):
            result = await create_mission(mock_ctx, objective="Test", compute_budget=-5.0)
        assert result["error"] is True
        assert result["error_code"] == INVALID_INPUT

    async def test_search_literature_mcp_init_crash(self, mock_ctx):
        """MCPClient constructor throws — should return structured error."""
        from apollobot.server.app import search_literature
        with patch("apollobot.mcp.MCPClient") as MockClient:
            MockClient.side_effect = RuntimeError("MCP init explosion")
            result = await search_literature(mock_ctx, query="test")
        assert result["error"] is True
        assert result["error_code"] == INTERNAL_ERROR

    async def test_load_session_nonexistent(self, mock_ctx):
        from apollobot.server.app import load_session
        result = await load_session(mock_ctx, "absolutely-no-such-session-xyz")
        assert result["error"] is True


@pytest.mark.evaluation
class TestEval_PhaseResultRetrieval:
    """Phase results are correctly stored and retrievable."""

    async def test_completed_phase_result(self, mock_ctx, active_session):
        from apollobot.server.app import get_phase_result

        # Complete a phase with findings
        active_session.session.begin_phase(Phase.LITERATURE_REVIEW)
        active_session.session.complete_phase(
            Phase.LITERATURE_REVIEW,
            summary="Reviewed 42 papers on CRISPR delivery",
            findings=[
                {"type": "paper", "title": "AAV delivery", "relevance": 0.9},
                {"type": "paper", "title": "Lipid nanoparticle delivery", "relevance": 0.85},
            ],
        )

        result = await get_phase_result(
            mock_ctx, active_session.session_id, "literature_review"
        )

        assert result["phase"] == "literature_review"
        assert result["summary"] == "Reviewed 42 papers on CRISPR delivery"
        assert len(result["findings"]) == 2
        assert result["findings"][0]["title"] == "AAV delivery"
        assert result["completed_at"] is not None
        assert result["started_at"] is not None
        assert isinstance(result["artifacts"], list)
        assert isinstance(result["errors"], list)

    async def test_multiple_phases_tracked(self, mock_ctx, active_session):
        from apollobot.server.app import get_phase_result

        # Complete two phases
        for phase, summary, findings in [
            (Phase.PLANNING, "Plan complete", []),
            (Phase.LITERATURE_REVIEW, "Lit review done", [{"paper": "A"}]),
        ]:
            active_session.session.begin_phase(phase)
            active_session.session.complete_phase(
                phase, summary=summary, findings=findings
            )

        plan_result = await get_phase_result(
            mock_ctx, active_session.session_id, "planning"
        )
        lit_result = await get_phase_result(
            mock_ctx, active_session.session_id, "literature_review"
        )

        assert plan_result["summary"] == "Plan complete"
        assert lit_result["summary"] == "Lit review done"
        assert len(lit_result["findings"]) == 1


@pytest.mark.evaluation
class TestEval_MCPCheckpointHandler:
    """MCPCheckpointHandler integration with ActiveSession."""

    async def test_handler_stores_and_resolves(self, sample_mission, temp_dir):
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

        # Start approval in background
        async def approve_after_delay():
            await asyncio.sleep(0.05)
            assert active.pending_checkpoint is not None
            assert active.pending_checkpoint["phase"] == "translate"
            active.checkpoint_approved = True
            active.checkpoint_event.set()

        approve_task = asyncio.create_task(approve_after_delay())
        result = await handler.request_approval("translate", "Proceed to translate?")
        await approve_task

        assert result is True
        assert active.pending_checkpoint is None

    async def test_handler_auto_approves_without_session(self):
        handler = MCPCheckpointHandler()
        result = await handler.request_approval("test", "should auto-approve")
        assert result is True

    async def test_handler_notify_doesnt_crash(self):
        handler = MCPCheckpointHandler()
        await handler.notify("analysis", "50% complete")
        # No exception = pass

    async def test_handler_with_progress_callback(self):
        handler = MCPCheckpointHandler()
        calls = []
        handler.add_progress_callback(
            lambda phase, summary: calls.append((phase, summary))
        )
        # notify calls callbacks
        await handler.notify("lit_review", "searching...")
        # Callbacks are awaited, so for sync lambdas this actually errors
        # Let's use an async callback
        calls.clear()
        async_calls = []

        async def async_cb(phase, summary):
            async_calls.append((phase, summary))

        handler._progress_callbacks = [async_cb]
        await handler.notify("analysis", "running stats")
        assert len(async_calls) == 1
        assert async_calls[0] == ("analysis", "running stats")
