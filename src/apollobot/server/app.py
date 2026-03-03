"""
FastMCP server — exposes ApolloBot as an MCP tool provider.

20 tools across 3 tiers:
  Tier 1: Mission-level (black-box execution)
  Tier 2: Phase-level (step-through control)
  Tier 3: Internal access (composable primitives)
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from apollobot.core import load_config as _load_config
from apollobot.server.checkpoint import MCPCheckpointHandler
from apollobot.server.errors import (
    BUDGET_EXCEEDED,
    CONFIG_MISSING,
    INTERNAL_ERROR,
    INVALID_INPUT,
    PHASE_NOT_AVAILABLE,
    SERVER_NOT_FOUND,
    SESSION_NOT_FOUND,
    error_response,
)
from apollobot.server.registry import ActiveSession, SessionRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan — creates SessionRegistry from config
# ---------------------------------------------------------------------------


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[dict[str, Any]]:
    """Create shared resources for the server's lifetime."""
    try:
        config = _load_config()
    except Exception:
        config = None

    registry = SessionRegistry(config=config)
    yield {"registry": registry}


# ---------------------------------------------------------------------------
# Server declaration
# ---------------------------------------------------------------------------

mcp = FastMCP(
    name="apollobot",
    instructions=(
        "ApolloBot — autonomous research engine by Frontier Science. "
        "Use create_mission to start, then run_discover/run_translate/"
        "run_implement/run_commercialize for full execution, or step_phase "
        "for fine-grained control. Use Tier 3 tools (search_literature, "
        "query_data_source, etc.) for composable primitives."
    ),
    lifespan=app_lifespan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_registry(ctx: Any) -> SessionRegistry:
    """Extract the session registry from the MCP context."""
    return ctx.request_context.lifespan_context["registry"]


async def _get_active(ctx: Any, session_id: str) -> ActiveSession | dict[str, Any]:
    """Get active session or return error dict."""
    registry = _get_registry(ctx)
    try:
        return await registry.get(session_id)
    except KeyError:
        return error_response(SESSION_NOT_FOUND, f"Session not found: {session_id}")


def _session_summary(active: ActiveSession) -> dict[str, Any]:
    """Build a summary dict from an active session."""
    s = active.session
    completed_phases = [
        phase for phase, result in s.phase_results.items()
        if result.completed_at
    ]
    return {
        "session_id": active.session_id,
        "objective": active.mission.objective,
        "mode": active.mission.mode.value,
        "domain": active.mission.domain,
        "current_phase": s.current_phase.value,
        "completed_phases": completed_phases,
        "cost": {
            "total_usd": s.cost.total_cost,
            "llm_calls": s.cost.llm_calls,
            "budget_remaining": max(
                0, s.mission.constraints.compute_budget - s.cost.total_cost
            ),
        },
        "datasets_count": len(s.datasets),
        "warnings": s.warnings,
    }


# ===========================================================================
# Tier 1: Mission-Level Tools
# ===========================================================================


@mcp.tool()
async def create_mission(
    ctx: Any,
    objective: str,
    mode: str = "hypothesis",
    domain: str = "bioinformatics",
    mission_yaml: str = "",
    compute_budget: float = 50.0,
    paper_id: str = "",
    dataset_id: str = "",
) -> dict[str, Any]:
    """Create a new research session from an objective or YAML config.

    Does NOT start execution — use run_discover, step_phase, etc. to begin.

    Args:
        objective: Research question or objective string.
        mode: Research mode — hypothesis, exploratory, meta-analysis,
              replication, simulation, translate, implement, commercialize, pipeline.
        domain: Scientific domain — bioinformatics, physics, cs_ml, comp_chem, economics.
        mission_yaml: Optional YAML string with full mission config.
        compute_budget: Max budget in USD (default 50).
        paper_id: Paper ID for replication mode.
        dataset_id: Dataset ID for exploratory mode.

    Returns:
        Dict with session_id and mission summary, or error dict.
    """
    try:
        from apollobot.core.mission import Mission, ResearchMode

        registry = _get_registry(ctx)

        if mission_yaml:
            import yaml
            raw = yaml.safe_load(mission_yaml)
            mission = Mission(**raw)
        else:
            valid_modes = [m.value for m in ResearchMode]
            if mode not in valid_modes:
                return error_response(
                    INVALID_INPUT,
                    f"Invalid mode '{mode}'. Must be one of: {valid_modes}",
                )
            mission = Mission.from_objective(
                objective,
                mode=mode,
                domain=domain,
                paper_id=paper_id,
                dataset_id=dataset_id,
            )

        if compute_budget < 0:
            return error_response(
                INVALID_INPUT,
                f"compute_budget must be >= 0, got {compute_budget}",
            )
        mission.constraints.compute_budget = compute_budget

        # Create checkpoint handler for this session
        checkpoint = MCPCheckpointHandler()
        active = await registry.create(mission, checkpoint_handler=checkpoint)
        checkpoint.set_session_getter(lambda: active)

        return {
            "session_id": active.session_id,
            "objective": mission.objective,
            "mode": mission.mode.value,
            "domain": mission.domain,
            "compute_budget": mission.constraints.compute_budget,
            "output_dir": str(active.session.session_dir),
        }

    except Exception as e:
        logger.exception("create_mission failed")
        return error_response(INTERNAL_ERROR, str(e))


@mcp.tool()
async def run_discover(ctx: Any, session_id: str) -> dict[str, Any]:
    """Run a full Discover (research) session. Long-running (5-30 min).

    The session must already exist (created via create_mission).
    Executes: planning → literature review → data acquisition → analysis →
    statistical testing → manuscript drafting → self-review → revision.

    Args:
        session_id: The session ID returned by create_mission.

    Returns:
        Session summary with phase results, cost, and output paths.
    """
    result = await _get_active(ctx, session_id)
    if isinstance(result, dict):
        return result
    active: ActiveSession = result

    try:
        session = await active.orchestrator.run_discover(active.mission)
        active.session = session
        return {
            **_session_summary(active),
            "output_dir": str(session.session_dir),
            "translation_scores": session.translation_scores,
        }
    except Exception as e:
        logger.exception("run_discover failed")
        return error_response(INTERNAL_ERROR, str(e))


@mcp.tool()
async def run_translate(ctx: Any, session_id: str) -> dict[str, Any]:
    """Run a full Translate session — converts research findings to implementation spec.

    Args:
        session_id: The session ID (must have mode=translate or source_session set).

    Returns:
        Session summary with translation report.
    """
    result = await _get_active(ctx, session_id)
    if isinstance(result, dict):
        return result
    active: ActiveSession = result

    try:
        session = await active.orchestrator.run_translate(active.mission)
        active.session = session
        return {
            **_session_summary(active),
            "translation_report": session.translation_report,
        }
    except Exception as e:
        logger.exception("run_translate failed")
        return error_response(INTERNAL_ERROR, str(e))


@mcp.tool()
async def run_implement(ctx: Any, session_id: str) -> dict[str, Any]:
    """Run a full Implement session — builds production code from translation spec.

    Args:
        session_id: The session ID (must have mode=implement or source_session set).

    Returns:
        Session summary with implementation artifacts.
    """
    result = await _get_active(ctx, session_id)
    if isinstance(result, dict):
        return result
    active: ActiveSession = result

    try:
        session = await active.orchestrator.run_implement(active.mission)
        active.session = session
        return _session_summary(active)
    except Exception as e:
        logger.exception("run_implement failed")
        return error_response(INTERNAL_ERROR, str(e))


@mcp.tool()
async def run_commercialize(ctx: Any, session_id: str) -> dict[str, Any]:
    """Run a full Commercialize session — market analysis and go-to-market plan.

    Args:
        session_id: The session ID (must have mode=commercialize or source_session set).

    Returns:
        Session summary with market analysis.
    """
    result = await _get_active(ctx, session_id)
    if isinstance(result, dict):
        return result
    active: ActiveSession = result

    try:
        session = await active.orchestrator.run_commercialize(active.mission)
        active.session = session
        return _session_summary(active)
    except Exception as e:
        logger.exception("run_commercialize failed")
        return error_response(INTERNAL_ERROR, str(e))


@mcp.tool()
async def run_pipeline(
    ctx: Any,
    session_id: str,
    auto_translate: bool = False,
) -> dict[str, Any]:
    """Run the full pipeline: Discover → Translate → Implement → Commercialize.

    Pauses at checkpoints between modes. Use approve_checkpoint to continue.

    Args:
        session_id: The session ID.
        auto_translate: If True, auto-proceed to Translate when score >= 7.

    Returns:
        Final session summary with all pipeline results.
    """
    result = await _get_active(ctx, session_id)
    if isinstance(result, dict):
        return result
    active: ActiveSession = result

    try:
        session = await active.orchestrator.run_pipeline(
            active.mission, auto_translate=auto_translate
        )
        active.session = session
        return _session_summary(active)
    except Exception as e:
        logger.exception("run_pipeline failed")
        return error_response(INTERNAL_ERROR, str(e))


# ===========================================================================
# Tier 2: Phase-Level Tools
# ===========================================================================


@mcp.tool()
async def step_phase(
    ctx: Any,
    session_id: str,
    phase: str = "",
) -> dict[str, Any]:
    """Execute the next (or specified) research phase.

    On first call for a Discover session, runs the planning phase to generate
    a ResearchPlan. Subsequent calls step through: literature_review →
    data_acquisition → analysis → statistical_testing → manuscript_drafting →
    self_review → manuscript_revision.

    Args:
        session_id: The session ID.
        phase: Optional specific phase to execute. If empty, runs the next phase.

    Returns:
        Phase result with summary, findings, and what comes next.
    """
    result = await _get_active(ctx, session_id)
    if isinstance(result, dict):
        return result
    active: ActiveSession = result

    try:
        from apollobot.agents.executor import ResearchExecutor
        from apollobot.agents.planner import ResearchPlanner
        from apollobot.core.session import Phase
        from apollobot.mcp import MCPServerInfo
        from apollobot.mcp.servers.builtin import get_domain_pack

        session = active.session

        # Budget check
        if not session.check_budget():
            return error_response(
                BUDGET_EXCEEDED,
                f"Budget exceeded: ${session.cost.total_cost:.2f} / "
                f"${session.mission.constraints.compute_budget:.2f}",
            )

        # Define the Discover phase sequence
        discover_phases = [
            Phase.PLANNING,
            Phase.LITERATURE_REVIEW,
            Phase.DATA_ACQUISITION,
            Phase.ANALYSIS,
            Phase.STATISTICAL_TESTING,
            Phase.MANUSCRIPT_DRAFTING,
            Phase.SELF_REVIEW,
            Phase.MANUSCRIPT_REVISION,
        ]

        # Determine which phase to run
        discover_phase_values = {p.value for p in discover_phases}

        if phase:
            try:
                target_phase = Phase(phase)
            except ValueError:
                return error_response(
                    INVALID_INPUT,
                    f"Invalid phase '{phase}'. Valid phases: {[p.value for p in discover_phases]}",
                )
            # Reject phases outside the discover sequence
            if target_phase.value not in discover_phase_values:
                return error_response(
                    INVALID_INPUT,
                    f"Phase '{phase}' is not a Discover phase. "
                    f"Valid phases: {[p.value for p in discover_phases]}",
                )
        else:
            # Find next phase
            completed = {
                p for p, r in session.phase_results.items()
                if r.completed_at
            }

            target_phase = None
            for p in discover_phases:
                if p.value not in completed:
                    target_phase = p
                    break

            if target_phase is None:
                return {
                    "status": "complete",
                    "message": "All phases have been completed.",
                    **_session_summary(active),
                }

        # Planning phase: generate the research plan
        if target_phase == Phase.PLANNING:
            # Connect MCP servers
            await active.orchestrator._connect_mcp_servers(active.mission.domain)
            available_servers = [
                s.name for s in active.orchestrator.mcp.get_servers(active.mission.domain)
            ]

            planner = ResearchPlanner(active.orchestrator.llm, active.provenance)
            plan = await planner.plan(active.mission, available_servers)
            active.plan = plan

            session.begin_phase(Phase.PLANNING)
            session.complete_phase(Phase.PLANNING, summary=plan.summary)
            session.save_state()

            return {
                "phase": "planning",
                "status": "completed",
                "plan_summary": plan.summary,
                "approach": plan.approach,
                "literature_queries": plan.literature_queries,
                "data_requirements_count": len(plan.data_requirements),
                "analysis_steps_count": len(plan.analysis_steps),
                "estimated_cost": plan.estimated_compute_cost,
                "estimated_hours": plan.estimated_time_hours,
                "next_phase": "literature_review",
            }

        # For all other phases, we need a plan
        if not active.plan:
            return error_response(
                PHASE_NOT_AVAILABLE,
                "No research plan exists. Run step_phase with phase='planning' first.",
            )

        # Connect MCP servers if not already done
        if not active.orchestrator.mcp.get_servers():
            await active.orchestrator._connect_mcp_servers(active.mission.domain)

        # Build executor
        executor = ResearchExecutor(
            llm=active.orchestrator.llm,
            mcp=active.orchestrator.mcp,
            provenance=active.provenance,
            checkpoint_handler=active.orchestrator.checkpoint,
        )

        # Map phases to executor methods
        phase_handlers = {
            Phase.LITERATURE_REVIEW: executor._literature_review,
            Phase.DATA_ACQUISITION: executor._acquire_data,
            Phase.ANALYSIS: executor._run_analyses,
            Phase.STATISTICAL_TESTING: executor._statistical_testing,
            Phase.MANUSCRIPT_DRAFTING: executor._draft_manuscript,
            Phase.SELF_REVIEW: executor._self_review,
            Phase.MANUSCRIPT_REVISION: executor._revise_manuscript,
        }

        handler = phase_handlers.get(target_phase)
        if not handler:
            return error_response(
                PHASE_NOT_AVAILABLE,
                f"No handler for phase '{target_phase.value}'.",
            )

        # Execute the phase
        session.begin_phase(target_phase)
        summary, findings = await handler(session, active.plan)
        session.complete_phase(target_phase, summary=summary, findings=findings)
        session.save_state()
        active.provenance.save()

        # Determine next phase
        phase_idx = discover_phases.index(target_phase)
        next_phase = (
            discover_phases[phase_idx + 1].value
            if phase_idx + 1 < len(discover_phases)
            else None
        )

        return {
            "phase": target_phase.value,
            "status": "completed",
            "summary": summary,
            "findings_count": len(findings),
            "findings": findings[:10],  # Limit to first 10 for readability
            "next_phase": next_phase,
            "cost_so_far": session.cost.total_cost,
        }

    except Exception as e:
        logger.exception("step_phase failed")
        return error_response(INTERNAL_ERROR, str(e))


@mcp.tool()
async def get_session_status(ctx: Any, session_id: str) -> dict[str, Any]:
    """Get current status of a session.

    Args:
        session_id: The session ID.

    Returns:
        Full session status including phase, cost, datasets, warnings, and next actions.
    """
    result = await _get_active(ctx, session_id)
    if isinstance(result, dict):
        return result
    active: ActiveSession = result

    summary = _session_summary(active)
    summary["pending_checkpoint"] = active.pending_checkpoint
    summary["has_plan"] = active.plan is not None
    summary["translation_scores"] = active.session.translation_scores
    return summary


@mcp.tool()
async def get_phase_result(
    ctx: Any,
    session_id: str,
    phase: str,
) -> dict[str, Any]:
    """Get detailed result of a completed phase.

    Args:
        session_id: The session ID.
        phase: Phase name (e.g., 'literature_review', 'analysis').

    Returns:
        Phase result with findings, artifacts, errors, and timing.
    """
    result = await _get_active(ctx, session_id)
    if isinstance(result, dict):
        return result
    active: ActiveSession = result

    phase_result = active.session.phase_results.get(phase)
    if not phase_result:
        return error_response(
            PHASE_NOT_AVAILABLE,
            f"No result for phase '{phase}'. Completed phases: "
            f"{list(active.session.phase_results.keys())}",
        )

    return {
        "phase": phase_result.phase.value,
        "started_at": phase_result.started_at,
        "completed_at": phase_result.completed_at,
        "summary": phase_result.summary,
        "findings": phase_result.findings,
        "artifacts": phase_result.artifacts,
        "errors": phase_result.errors,
    }


@mcp.tool()
async def approve_checkpoint(
    ctx: Any,
    session_id: str,
    approved: bool = True,
) -> dict[str, Any]:
    """Approve or deny a pending pipeline checkpoint.

    When run_pipeline or step_phase hits a checkpoint requiring approval,
    it waits. Call this tool to continue.

    Args:
        session_id: The session ID.
        approved: True to approve and continue, False to deny and stop.

    Returns:
        Confirmation of checkpoint resolution.
    """
    result = await _get_active(ctx, session_id)
    if isinstance(result, dict):
        return result
    active: ActiveSession = result

    if not active.pending_checkpoint:
        return error_response(
            PHASE_NOT_AVAILABLE,
            "No pending checkpoint for this session.",
        )

    checkpoint_info = active.pending_checkpoint.copy()
    active.checkpoint_approved = approved
    active.checkpoint_event.set()

    return {
        "checkpoint": checkpoint_info,
        "approved": approved,
        "message": "Checkpoint approved — execution will continue." if approved
                   else "Checkpoint denied — execution will stop.",
    }


# ===========================================================================
# Tier 3: Internal Access Tools
# ===========================================================================


@mcp.tool()
async def search_literature(
    ctx: Any,
    query: str,
    session_id: str = "",
    domain: str = "bioinformatics",
    limit: int = 20,
) -> dict[str, Any]:
    """Search literature across PubMed, arXiv, and Semantic Scholar.

    Can be used standalone (without a session) or within a session context.

    Args:
        query: Search query string.
        session_id: Optional session ID to use its MCP connections.
        domain: Domain for server selection (if no session_id).
        limit: Max results per server.

    Returns:
        Dict with papers list and search metadata.
    """
    try:
        from apollobot.mcp import MCPClient, MCPServerInfo
        from apollobot.mcp.servers.builtin import get_domain_pack

        # Get or create MCP client
        if session_id:
            result = await _get_active(ctx, session_id)
            if isinstance(result, dict):
                return result
            active: ActiveSession = result
            mcp_client = active.orchestrator.mcp
        else:
            mcp_client = MCPClient()
            for srv in get_domain_pack(domain):
                mcp_client.register(MCPServerInfo(
                    name=srv.name,
                    url=srv.url,
                    description=srv.description,
                    domain=srv.domain,
                    api_base=srv.api_base,
                ))

        all_papers: list[dict[str, Any]] = []
        servers_searched = []

        for server in mcp_client.get_servers():
            if server.domain in ("shared", domain) or session_id:
                try:
                    results = await mcp_client.query(
                        server.name,
                        "search",
                        {"query": query, "limit": limit},
                    )
                    papers = results.get("papers", results.get("results", []))
                    all_papers.extend(papers)
                    servers_searched.append(server.name)
                except Exception:
                    pass

        # Deduplicate by DOI or title; keep papers with no identifying key
        seen = set()
        unique = []
        for p in all_papers:
            if not isinstance(p, dict):
                continue
            key = p.get("doi") or p.get("title") or ""
            if not key:
                # No identifying key — keep the paper (can't dedup)
                unique.append(p)
            elif key not in seen:
                seen.add(key)
                unique.append(p)

        return {
            "query": query,
            "papers": unique[:limit],
            "total_found": len(unique),
            "servers_searched": servers_searched,
        }

    except Exception as e:
        logger.exception("search_literature failed")
        return error_response(INTERNAL_ERROR, str(e))


@mcp.tool()
async def query_data_source(
    ctx: Any,
    server_name: str,
    capability: str,
    parameters: dict[str, Any] | None = None,
    session_id: str = "",
    domain: str = "bioinformatics",
) -> dict[str, Any]:
    """Query any MCP data server directly.

    Args:
        server_name: Name of the MCP server (e.g., 'geo', 'pubmed', 'uniprot').
        capability: The capability to invoke (e.g., 'search', 'query', 'download').
        parameters: Query parameters dict.
        session_id: Optional session ID to use its MCP connections.
        domain: Domain for server selection (if no session_id).

    Returns:
        Raw query results from the server.
    """
    try:
        from apollobot.mcp import MCPClient, MCPServerInfo
        from apollobot.mcp.servers.builtin import get_domain_pack

        if session_id:
            result = await _get_active(ctx, session_id)
            if isinstance(result, dict):
                return result
            active: ActiveSession = result
            mcp_client = active.orchestrator.mcp
        else:
            mcp_client = MCPClient()
            for srv in get_domain_pack(domain):
                mcp_client.register(MCPServerInfo(
                    name=srv.name,
                    url=srv.url,
                    description=srv.description,
                    domain=srv.domain,
                    api_base=srv.api_base,
                ))

        # Check server exists
        known = {s.name for s in mcp_client.get_servers()}
        if server_name not in known:
            return error_response(
                SERVER_NOT_FOUND,
                f"Server '{server_name}' not found. Available: {sorted(known)}",
            )

        result = await mcp_client.query(server_name, capability, parameters or {})
        return {"server": server_name, "capability": capability, "result": result}

    except Exception as e:
        logger.exception("query_data_source failed")
        return error_response(INTERNAL_ERROR, str(e))


@mcp.tool()
async def list_data_servers(
    ctx: Any,
    domain: str = "",
) -> dict[str, Any]:
    """List available MCP data servers, optionally filtered by domain.

    Args:
        domain: Filter by domain (bioinformatics, physics, cs_ml, comp_chem, economics).
                Empty string returns all domains.

    Returns:
        List of servers with name, description, domain, and category.
    """
    try:
        from apollobot.mcp.servers.builtin import DOMAIN_PACKS, get_domain_pack

        if domain:
            servers = get_domain_pack(domain)
            return {
                "domain": domain,
                "servers": [
                    {
                        "name": s.name,
                        "description": s.description,
                        "domain": s.domain,
                        "category": s.category,
                        "requires_key": s.requires_key,
                    }
                    for s in servers
                ],
            }

        # Return all domains
        all_servers = {}
        for d, srvs in DOMAIN_PACKS.items():
            all_servers[d] = [
                {
                    "name": s.name,
                    "description": s.description,
                    "domain": s.domain,
                    "category": s.category,
                    "requires_key": s.requires_key,
                }
                for s in srvs
            ]
        return {"domains": all_servers}

    except Exception as e:
        logger.exception("list_data_servers failed")
        return error_response(INTERNAL_ERROR, str(e))


@mcp.tool()
async def run_analysis_step(
    ctx: Any,
    session_id: str,
    step_name: str,
    description: str,
    method: str = "",
    parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute a single analysis step (code generation + execution).

    Generates Python code for the analysis and runs it.

    Args:
        session_id: The session ID.
        step_name: Name for the analysis step.
        description: Description of the analysis to perform.
        method: Statistical/analytical method to use.
        parameters: Additional parameters for the analysis.

    Returns:
        Analysis result with status, stdout, and any errors.
    """
    result = await _get_active(ctx, session_id)
    if isinstance(result, dict):
        return result
    active: ActiveSession = result

    try:
        from apollobot.agents.planner import AnalysisStep

        step = AnalysisStep(
            name=step_name,
            description=description,
            method=method or "custom",
            parameters=parameters or {},
        )

        # We need a minimal plan with just this step
        from apollobot.agents.planner import ResearchPlan
        mini_plan = active.plan or ResearchPlan(
            mission_id=active.session_id,
            analysis_steps=[step],
        )

        # Temporarily replace analysis_steps with just our step
        original_steps = mini_plan.analysis_steps
        mini_plan.analysis_steps = [step]

        from apollobot.agents.executor import ResearchExecutor
        executor = ResearchExecutor(
            llm=active.orchestrator.llm,
            mcp=active.orchestrator.mcp,
            provenance=active.provenance,
        )

        summary, findings = await executor._run_analyses(active.session, mini_plan)
        mini_plan.analysis_steps = original_steps

        return {
            "step": step_name,
            "summary": summary,
            "findings": findings,
        }

    except Exception as e:
        logger.exception("run_analysis_step failed")
        return error_response(INTERNAL_ERROR, str(e))


@mcp.tool()
async def draft_section(
    ctx: Any,
    session_id: str,
    section: str,
) -> dict[str, Any]:
    """Draft one manuscript section with anti-hallucination guardrails.

    Uses the data inventory as ground truth.

    Args:
        session_id: The session ID.
        section: Section name — abstract, introduction, methods, results,
                 discussion, or conclusion.

    Returns:
        The drafted section text and evidence level.
    """
    result = await _get_active(ctx, session_id)
    if isinstance(result, dict):
        return result
    active: ActiveSession = result

    try:
        valid_sections = ["abstract", "introduction", "methods", "results", "discussion", "conclusion"]
        if section not in valid_sections:
            return error_response(
                INVALID_INPUT,
                f"Invalid section '{section}'. Must be one of: {valid_sections}",
            )

        from apollobot.agents.executor import ResearchExecutor
        executor = ResearchExecutor(
            llm=active.orchestrator.llm,
            mcp=active.orchestrator.mcp,
            provenance=active.provenance,
        )

        data_inventory = executor._build_data_inventory(active.session)

        # Determine evidence level
        analysis_result = active.session.phase_results.get("analysis")
        if analysis_result:
            completed = sum(1 for f in analysis_result.findings if f.get("status") == "completed")
            total = len(analysis_result.findings)
            if total > 0 and completed / total >= 0.7:
                evidence_level = "moderate"
            elif completed > 0:
                evidence_level = "weak"
            else:
                evidence_level = "theoretical"
        else:
            evidence_level = "theoretical"

        plan = active.plan
        approach = plan.approach if plan else active.mission.objective

        resp = await active.orchestrator.llm.complete(
            messages=[{"role": "user", "content": (
                f"Write the {section.upper()} section of a scientific paper.\n\n"
                f"Research objective: {active.session.mission.objective}\n"
                f"Domain: {active.session.mission.domain}\n"
                f"Approach: {approach}\n\n"
                f"DATA INVENTORY (ground truth):\n{data_inventory}\n\n"
                f"Evidence level: {evidence_level}\n"
                "Write in clear, precise scientific prose."
            )}],
            system=(
                "You are writing a scientific paper. "
                "NEVER invent data not in the inventory. "
                f"Evidence level: {evidence_level}."
            ),
        )

        active.session.cost.record_llm_call(
            resp.input_tokens, resp.output_tokens, resp.cost_usd
        )

        return {
            "section": section,
            "text": resp.text,
            "evidence_level": evidence_level,
        }

    except Exception as e:
        logger.exception("draft_section failed")
        return error_response(INTERNAL_ERROR, str(e))


@mcp.tool()
async def review_manuscript(ctx: Any, session_id: str) -> dict[str, Any]:
    """Self-review + statistical audit of the manuscript.

    Runs both a qualitative review and a quantitative statistical audit.

    Args:
        session_id: The session ID (must have a manuscript drafted).

    Returns:
        Review text and statistical audit results.
    """
    result = await _get_active(ctx, session_id)
    if isinstance(result, dict):
        return result
    active: ActiveSession = result

    try:
        if not active.plan:
            return error_response(
                PHASE_NOT_AVAILABLE,
                "No research plan exists. Cannot review without a plan.",
            )

        from apollobot.agents.executor import ResearchExecutor
        executor = ResearchExecutor(
            llm=active.orchestrator.llm,
            mcp=active.orchestrator.mcp,
            provenance=active.provenance,
        )

        summary, findings = await executor._self_review(active.session, active.plan)
        return {
            "summary": summary,
            "findings": findings,
            "translation_scores": active.session.translation_scores,
        }

    except Exception as e:
        logger.exception("review_manuscript failed")
        return error_response(INTERNAL_ERROR, str(e))


@mcp.tool()
async def get_provenance(ctx: Any, session_id: str) -> dict[str, Any]:
    """Get full provenance chain for a session.

    Returns execution log, data lineage, and model call records.

    Args:
        session_id: The session ID.

    Returns:
        Provenance data with execution log, data lineage, and model calls.
    """
    result = await _get_active(ctx, session_id)
    if isinstance(result, dict):
        return result
    active: ActiveSession = result

    prov = active.provenance
    return {
        "session_id": session_id,
        "execution_log": prov.execution_log[-50:],  # Last 50 events
        "data_lineage": [entry.model_dump() for entry in prov.data_lineage[-20:]],
        "model_calls": [call.model_dump() for call in prov.model_calls[-20:]],
        "total_events": len(prov.execution_log),
        "total_transforms": len(prov.data_lineage),
        "total_llm_calls": len(prov.model_calls),
    }


@mcp.tool()
async def get_cost(ctx: Any, session_id: str) -> dict[str, Any]:
    """Get cost breakdown for a session.

    Args:
        session_id: The session ID.

    Returns:
        Cost breakdown with total, per-call average, and budget remaining.
    """
    result = await _get_active(ctx, session_id)
    if isinstance(result, dict):
        return result
    active: ActiveSession = result

    cost = active.session.cost
    budget = active.mission.constraints.compute_budget

    return {
        "session_id": session_id,
        "total_cost_usd": cost.total_cost,
        "llm_cost_usd": cost.estimated_cost_usd,
        "compute_cost_usd": cost.compute_cost_usd,
        "llm_calls": cost.llm_calls,
        "llm_input_tokens": cost.llm_input_tokens,
        "llm_output_tokens": cost.llm_output_tokens,
        "avg_cost_per_call": cost.estimated_cost_usd / max(cost.llm_calls, 1),
        "budget_total": budget,
        "budget_remaining": max(0, budget - cost.total_cost),
        "budget_utilization_pct": (cost.total_cost / budget * 100) if budget > 0 else 0,
    }


@mcp.tool()
async def list_sessions(ctx: Any) -> dict[str, Any]:
    """List all sessions — both active (in memory) and historical (on disk).

    Returns:
        Dict with active and historical session lists.
    """
    try:
        registry = _get_registry(ctx)
        active = await registry.list_active()
        historical = registry.list_historical()

        return {
            "active": active,
            "historical": historical,
            "total_active": len(active),
            "total_historical": len(historical),
        }

    except Exception as e:
        logger.exception("list_sessions failed")
        return error_response(INTERNAL_ERROR, str(e))


@mcp.tool()
async def load_session(ctx: Any, session_id: str) -> dict[str, Any]:
    """Load a historical session from disk into active memory.

    This allows you to inspect results, resume work, or run additional
    analyses on a previously completed session.

    Args:
        session_id: The session ID (directory name under sessions dir).

    Returns:
        Session summary after loading.
    """
    try:
        registry = _get_registry(ctx)
        active = await registry.load_from_disk(session_id)
        return _session_summary(active)

    except FileNotFoundError:
        return error_response(SESSION_NOT_FOUND, f"Session '{session_id}' not found on disk.")
    except Exception as e:
        logger.exception("load_session failed")
        return error_response(INTERNAL_ERROR, str(e))
