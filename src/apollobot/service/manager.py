"""Durable investigation manager that adapts ApolloBot to web jobs."""

from __future__ import annotations

import asyncio
import hashlib
import json
import mimetypes
import os
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4, uuid5

from apollobot import __version__
from apollobot.agents import LLMProvider, OpenAIProvider, create_llm
from apollobot.agents.executor import CheckpointHandler
from apollobot.agents.orchestrator import Orchestrator
from apollobot.agents.planner import ResearchPlan, ResearchPlanner
from apollobot.core import ApolloConfig, load_config
from apollobot.core.mission import Constraints, Mission
from apollobot.core.provenance import ProvenanceEngine
from apollobot.core.session import Phase
from apollobot.service.framer import classify_question_safety
from apollobot.service.model_catalog import (
    MODEL_CATALOG_VERSION,
    ModelRoute,
    resolve_model_route,
)
from apollobot.service.models import (
    NODE_BLUEPRINT,
    PHASE_TO_NODE,
    QuestionCheck,
    ServiceEvent,
    utc_now,
)
from apollobot.service.store import ServiceStore


class EventCheckpointHandler(CheckpointHandler):
    def __init__(self, investigation_id: str, manager: InvestigationManager) -> None:
        self.investigation_id = investigation_id
        self.manager = manager

    async def request_approval(self, phase: str, summary: str) -> bool:
        # The platform-level approval happened before the job was enqueued.
        return True

    async def notify(self, phase: str, summary: str) -> None:
        await self.manager.record_phase(self.investigation_id, phase, summary)


class InvestigationManager:
    def __init__(
        self,
        store: ServiceStore,
        config: ApolloConfig | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        self.store = store
        self.config = (config or load_config()).model_copy(deep=True)
        self.output_dir = Path(output_dir or self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.output_dir = str(self.output_dir)
        self.tasks: dict[str, asyncio.Task[None]] = {}
        self._phase_started: set[tuple[str, str]] = set()
        self.max_concurrent_jobs = environment_limit("APOLLOBOT_MAX_CONCURRENT_JOBS", 3)
        self.max_concurrent_plans = environment_limit(
            "APOLLOBOT_MAX_CONCURRENT_PLANS", self.max_concurrent_jobs
        )
        self.planning_timeout_seconds = environment_timeout("APOLLOBOT_PLANNER_TIMEOUT", 40)
        self.max_literature_queries = environment_limit("APOLLOBOT_MAX_LITERATURE_QUERIES", 2)
        self.max_data_requirements = environment_limit("APOLLOBOT_MAX_DATA_REQUIREMENTS", 2)
        self.max_analysis_steps = environment_limit("APOLLOBOT_MAX_ANALYSIS_STEPS", 2)
        self._execution_slots = asyncio.Semaphore(self.max_concurrent_jobs)
        self._planning_slots = asyncio.Semaphore(self.max_concurrent_plans)

    def create(
        self,
        check: QuestionCheck,
        user_id: str | None = None,
        investigation_id: str | None = None,
        model_id: object = None,
        provider_tag: object = None,
    ) -> dict[str, Any]:
        if check.answerability != "investigable":
            raise ValueError("Question must be investigable before an investigation is created")
        if classify_question_safety(check.question):
            raise ValueError("Question requires safety review")
        if investigation_id:
            try:
                investigation_id = str(UUID(investigation_id))
            except ValueError as error:
                raise ValueError("Investigation id must be a UUID") from error
        else:
            investigation_id = str(uuid4())
        route = resolve_model_route(model_id, provider_tag)
        nodes = [node.model_copy(deep=True) for node in NODE_BLUEPRINT]
        payload = {
            "id": investigation_id,
            "user_id": user_id,
            "title": check.title,
            "objective": check.question,
            "domain": check.domain,
            "mode": check.mode,
            "status": "planned",
            "current_node": "define_hypotheses",
            "budget_usd": check.estimate.compute_usd,
            "model_id": route.model_id,
            "model_provider_tag": route.provider_tag,
            "check": check.model_dump(by_alias=True),
        }
        self.store.create_investigation(payload, nodes)
        hypotheses = " ".join(
            f"Question {index + 1}: {hypothesis}"
            for index, hypothesis in enumerate(check.hypotheses[:3])
        )
        self.store.append_message(
            investigation_id,
            "apollobot",
            "direction",
            (f"I propose this research direction: {check.rationale.strip()} {hypotheses}".strip()),
            0,
        )
        self._event(
            investigation_id,
            "investigation.created",
            "complete",
            "Question-level plan created. No tools or compute have run.",
            node_id="frame_question",
            data=self._model_stamp(route),
        )
        return {
            "id": investigation_id,
            "status": "planned",
            "engine": "apollobot",
            "model_id": route.model_id,
            "provider_tag": route.provider_tag,
        }

    def snapshot(self, investigation_id: str, after: int = 0) -> dict[str, Any] | None:
        return self.store.snapshot(investigation_id, after)

    def artifact_path(
        self, investigation_id: str, artifact_id: str
    ) -> tuple[dict[str, Any], Path] | None:
        artifact = self.store.get_artifact(investigation_id, artifact_id)
        if not artifact:
            return None
        session_root = (self.output_dir / investigation_id).resolve()
        path = (session_root / artifact["path"]).resolve()
        if session_root not in path.parents or not path.is_file():
            return None
        return artifact, path

    async def action(
        self, investigation_id: str, action: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        investigation = self.store.get_investigation(investigation_id)
        if not investigation:
            raise KeyError(investigation_id)

        if action == "prepare":
            if investigation["status"] != "planned":
                raise ValueError(
                    f"Cannot prepare an investigation in {investigation['status']} state"
                )
            if investigation_id not in self.tasks or self.tasks[investigation_id].done():
                self.tasks[investigation_id] = asyncio.create_task(
                    self._prepare_guarded(investigation_id),
                    name=f"apollo-plan-{investigation_id}",
                )
            return {"investigation": {"id": investigation_id, "status": "planning"}}

        if action == "revise":
            if investigation["status"] not in {"planned", "awaiting_approval"}:
                raise ValueError("A proposal can only be revised before the experiment is approved")
            feedback = str(payload.get("message") or "").strip()
            if len(feedback) < 8 or len(feedback) > 2000:
                raise ValueError("Revision guidance must be between 8 and 2000 characters")
            messages = self.store.list_messages(investigation_id)
            revision = max((int(item["revision"]) for item in messages), default=0) + 1
            self.store.append_message(
                investigation_id,
                "researcher",
                "experiment_plan",
                feedback,
                revision,
            )
            guidance = [
                str(item["body"])
                for item in self.store.list_messages(investigation_id)
                if item["role"] == "researcher"
            ][-8:]
            mission = self._mission(investigation_id, investigation)
            mission.metadata["researcher_guidance"] = guidance
            mission.metadata["plan_revision"] = revision
            self.store.update_node(investigation_id, "define_hypotheses", "ready")
            self.store.update_node(investigation_id, "design_experiment", "ready")
            self.store.update_node(
                investigation_id,
                "approve_plan",
                "pending",
                "A revised executable plan is being developed for human review.",
            )
            self.store.update_investigation(
                investigation_id,
                status="planning",
                current_node="define_hypotheses",
                mission_json=mission.model_dump_json(),
                plan_json=None,
                error=None,
            )
            self._event(
                investigation_id,
                "plan.revision_requested",
                "planning",
                f"The researcher requested plan revision {revision}; no tools or compute started.",
                node_id="define_hypotheses",
                data={"revision": revision},
            )
            if investigation_id not in self.tasks or self.tasks[investigation_id].done():
                self.tasks[investigation_id] = asyncio.create_task(
                    self._prepare_guarded(investigation_id),
                    name=f"apollo-plan-{investigation_id}-r{revision}",
                )
            return {
                "investigation": {
                    "id": investigation_id,
                    "status": "planning",
                    "revision": revision,
                }
            }

        if action == "approve":
            if investigation["status"] not in {"awaiting_approval", "paused"}:
                raise ValueError(
                    f"Cannot approve an investigation in {investigation['status']} state"
                )
            if "plan" not in investigation:
                raise ValueError("The executable experiment plan has not been prepared")
            if investigation_id not in self.tasks or self.tasks[investigation_id].done():
                self.store.update_node(
                    investigation_id,
                    "approve_plan",
                    "complete",
                    "Plan approved by a human operator.",
                )
                self.store.update_investigation(
                    investigation_id, status="queued", current_node="search_literature"
                )
                run = self._queue_experiment_run(investigation_id, investigation["plan"])
                self._event(
                    investigation_id,
                    "checkpoint.approved",
                    "complete",
                    "Research plan approved; ApolloBot queued.",
                    node_id="approve_plan",
                    data={"run_id": run["id"], "attempt": run["attempt"]},
                )
                self.tasks[investigation_id] = asyncio.create_task(
                    self._run_guarded(investigation_id), name=f"apollo-{investigation_id}"
                )
            return {"investigation": {"id": investigation_id, "status": "queued"}}

        if action == "cancel":
            task = self.tasks.get(investigation_id)
            if task and not task.done():
                task.cancel()
            self.store.update_investigation(
                investigation_id, status="cancelled", completed_at=utc_now()
            )
            self._end_current_run(
                investigation_id,
                "cancelled",
                "The operator cancelled this execution attempt.",
                exit_code=130,
            )
            self.store.update_experiment(investigation_id, status="cancelled")
            self._event(
                investigation_id,
                "investigation.cancelled",
                "cancelled",
                "Investigation cancelled by the operator.",
            )
            return {"investigation": {"id": investigation_id, "status": "cancelled"}}

        if action == "pause":
            if investigation["status"] not in {"queued", "running"}:
                raise ValueError("Only queued or running investigations can be paused")
            task = self.tasks.get(investigation_id)
            if task and not task.done():
                task.cancel()
            current_node = str(investigation.get("current_node") or "search_literature")
            self.store.update_node(
                investigation_id,
                current_node,
                "ready",
                "Execution attempt stopped. Resume restarts the approved plan in a clean run.",
            )
            self.store.update_investigation(
                investigation_id,
                status="paused",
                error="Paused by the operator; resume restarts the approved plan.",
            )
            self._end_current_run(
                investigation_id,
                "cancelled",
                "The operator paused this attempt; a resume creates a new immutable attempt.",
                exit_code=130,
            )
            self.store.update_experiment(investigation_id, status="approved")
            self._event(
                investigation_id,
                "investigation.paused",
                "paused",
                (
                    "Execution stopped by the operator. Captured artifacts remain available; "
                    "resume starts a clean attempt of the approved plan."
                ),
                node_id=current_node,
            )
            return {"investigation": {"id": investigation_id, "status": "paused"}}

        if action == "resume":
            if investigation["status"] != "paused":
                raise ValueError("Only an interrupted investigation can be resumed")
            if "plan" not in investigation:
                raise ValueError("The interrupted investigation has no approved executable plan")
            if investigation_id not in self.tasks or self.tasks[investigation_id].done():
                self.store.update_investigation(investigation_id, status="queued")
                run = self._queue_experiment_run(investigation_id, investigation["plan"])
                self._event(
                    investigation_id,
                    "investigation.resume_requested",
                    "queued",
                    (
                        "The operator explicitly resumed the approved plan after a worker "
                        "interruption."
                    ),
                    data={"run_id": run["id"], "attempt": run["attempt"]},
                )
                self.tasks[investigation_id] = asyncio.create_task(
                    self._run_guarded(investigation_id), name=f"apollo-{investigation_id}"
                )
            return {"investigation": {"id": investigation_id, "status": "queued"}}

        if action == "branch":
            raise ValueError("Branch creation is handled by the Frontier record service")

        raise ValueError(f"Unknown action: {action}")

    async def recover_interrupted(self) -> int:
        """Move crash-interrupted jobs to explicit, truthful restart states."""
        recovered = 0
        for investigation in self.store.list_investigations({"planning", "queued", "running"}):
            investigation_id = investigation["id"]
            if investigation["status"] == "planning" and "plan" not in investigation:
                self.store.update_node(investigation_id, "define_hypotheses", "pending")
                self.store.update_node(investigation_id, "design_experiment", "pending")
                self.store.update_investigation(
                    investigation_id,
                    status="planned",
                    current_node="define_hypotheses",
                    error="Worker restarted before the experiment plan was committed.",
                )
                self._event(
                    investigation_id,
                    "worker.interrupted_planning",
                    "planned",
                    (
                        "The worker restarted before planning finished. Develop the experiment "
                        "again; no tools or compute were approved."
                    ),
                )
            else:
                current_node = str(investigation.get("current_node") or "search_literature")
                self.store.update_node(
                    investigation_id,
                    current_node,
                    "ready",
                    (
                        "Worker interruption recorded. Explicit resume is required before "
                        "execution restarts."
                    ),
                )
                self.store.update_investigation(
                    investigation_id,
                    status="paused",
                    error=(
                        "Worker restarted during approved execution; explicit resume is required."
                    ),
                )
                self._end_current_run(
                    investigation_id,
                    "failed",
                    "The worker restarted before this attempt reached a terminal state.",
                    exit_code=1,
                )
                self.store.update_experiment(investigation_id, status="approved")
                self._event(
                    investigation_id,
                    "worker.interrupted_execution",
                    "paused",
                    (
                        "The worker restarted during execution. The approved plan is preserved "
                        "and will not rerun without explicit approval."
                    ),
                    node_id=current_node,
                )
            recovered += 1
        return recovered

    async def _prepare_guarded(self, investigation_id: str) -> None:
        async with self._planning_slots:
            await self._prepare(investigation_id)

    async def _run_guarded(self, investigation_id: str) -> None:
        async with self._execution_slots:
            await self._run(investigation_id)

    async def _prepare(self, investigation_id: str) -> None:
        investigation = self.store.get_investigation(investigation_id)
        if not investigation:
            return
        mission = self._mission(investigation_id, investigation)
        route = self._model_route(investigation)
        self.store.update_investigation(
            investigation_id,
            status="planning",
            current_node="define_hypotheses",
            mission_json=mission.model_dump_json(),
            error=None,
        )
        self.store.update_node(investigation_id, "define_hypotheses", "running")
        self.store.update_node(investigation_id, "design_experiment", "running")
        self._event(
            investigation_id,
            "plan.started",
            "running",
            "ApolloBot is developing the falsifiable hypotheses and executable experiment.",
            node_id="design_experiment",
            data=self._model_stamp(route),
        )
        try:
            planning_config = self.config.model_copy(deep=True)
            orchestrator = Orchestrator(
                config=planning_config,
                interactive=False,
                llm_factory=lambda: self._llm_for_route(route),
            )
            await orchestrator._connect_mcp_servers(mission.domain)
            available_servers = [
                server.name for server in orchestrator.mcp.get_servers(mission.domain)
            ]
            provenance = ProvenanceEngine(self.output_dir / investigation_id)
            provenance.log_event("model_route_selected", self._model_stamp(route))
            planner = ResearchPlanner(orchestrator.llm, provenance)
            try:
                async with asyncio.timeout(self.planning_timeout_seconds):
                    plan = await planner.plan(mission, available_servers)
                    plan.assert_executable()
            except Exception as error:
                plan = self._fallback_plan(mission, investigation)
                provenance.log_event(
                    "experiment_plan_fallback",
                    {
                        "reason": type(error).__name__,
                        "timeout_seconds": self.planning_timeout_seconds,
                    },
                )
                self._event(
                    investigation_id,
                    "plan.fallback",
                    "complete",
                    (
                        "The model planning pass exceeded its service window, so ApolloBot "
                        "prepared a bounded executable pilot from the accepted question contract."
                    ),
                    node_id="design_experiment",
                    data={"reason": type(error).__name__},
                )
            plan, bounds = self._bound_plan(mission, plan)
            plan.assert_executable()
            if bounds:
                provenance.log_event("experiment_plan_bounded", bounds)
                self._event(
                    investigation_id,
                    "plan.bounded",
                    "complete",
                    "The generated plan was reduced to the declared interactive budget.",
                    node_id="design_experiment",
                    data=bounds,
                )
            provenance.log_event(
                "experiment_plan_prepared",
                {"plan": plan.model_dump(mode="json")},
            )
            provenance.save()
            preregistered_at = utc_now()
            self.store.upsert_experiment(
                investigation_id,
                self._experiment_from_plan(investigation_id, investigation, plan, preregistered_at),
            )
            await self._capture_preregistration(investigation_id, plan)
            self.store.update_node(
                investigation_id,
                "define_hypotheses",
                "complete",
                f"Defined {len(plan.hypotheses)} falsifiable hypotheses.",
            )
            self.store.update_node(
                investigation_id,
                "design_experiment",
                "complete",
                (
                    f"Prepared {len(plan.analysis_steps)} analysis steps and "
                    f"{len(plan.data_requirements)} data requirements."
                ),
            )
            self.store.update_node(
                investigation_id,
                "approve_plan",
                "awaiting_approval",
                "The executable plan is ready for human review.",
            )
            self.store.update_investigation(
                investigation_id,
                status="awaiting_approval",
                current_node="approve_plan",
                plan_json=plan.model_dump_json(),
            )
            revision = max(0, int(mission.metadata.get("plan_revision") or 0))
            self.store.append_message(
                investigation_id,
                "apollobot",
                "experiment_plan",
                (
                    f"I prepared plan revision {revision}. {plan.summary.strip()} "
                    "Review the scope, hypotheses, data, controls, and stopping rules below. "
                    "Nothing will execute until you explicitly approve this preregistration."
                ),
                revision,
            )
            self._event(
                investigation_id,
                "plan.completed",
                "complete",
                "Executable experiment plan prepared and waiting for approval.",
                node_id="design_experiment",
                data={"plan": plan.model_dump(mode="json")},
            )
            self._event(
                investigation_id,
                "checkpoint.requested",
                "awaiting_approval",
                "The executable plan is waiting for human approval.",
                node_id="approve_plan",
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self.store.update_node(
                investigation_id, "design_experiment", "failed", str(error)[:500]
            )
            self.store.update_investigation(
                investigation_id,
                status="failed",
                error=str(error),
                completed_at=utc_now(),
            )
            self._event(
                investigation_id,
                "plan.failed",
                "failed",
                f"Experiment planning failed: {str(error)[:300]}",
                node_id="design_experiment",
            )

    @staticmethod
    def _fallback_plan(mission: Mission, investigation: dict[str, Any]) -> ResearchPlan:
        guidance = [
            str(item).strip()
            for item in mission.metadata.get("researcher_guidance", [])
            if str(item).strip()
        ][-8:]
        guidance_contract = (
            " Apply these researcher-declared scope constraints and document how each was "
            "handled: " + " | ".join(guidance)
            if guidance
            else ""
        )
        hypotheses = mission.hypotheses or [
            f"A bounded computational pilot can produce evidence relevant to: {mission.objective}"
        ]
        plan_hypotheses = [
            {
                "hypothesis": hypothesis,
                "test": (
                    "Run the preregistered pilot and evaluate the declared outcome against "
                    "its captured evidence and robustness checks."
                ),
                "null_hypothesis": (
                    "The pilot does not produce sufficient evidence to support the proposed claim."
                ),
            }
            for hypothesis in hypotheses[:3]
        ]
        title = str(investigation.get("title") or mission.title or mission.objective)
        return ResearchPlan(
            mission_id=mission.id,
            summary=f"Run a bounded, preregistered computational pilot for {title}.",
            approach=(
                "Map the closest public evidence, define an auditable extraction boundary, "
                "execute one deterministic comparison, and report supported, unsupported, "
                "and unresolved outcomes without extending beyond the captured artifacts."
                f"{guidance_contract}"
            ),
            hypotheses=plan_hypotheses,
            literature_queries=[mission.objective[:500]],
            data_requirements=[
                {
                    "description": "Public evidence directly relevant to the accepted question",
                    "source_type": "mcp_server",
                    "priority": "required",
                    "access_mode": "public",
                    "redistribution_allowed": False,
                    "availability_note": (
                        "Use only sources whose access and provenance can be captured."
                    ),
                }
            ],
            analysis_steps=[
                {
                    "name": "bounded_evidence_pilot",
                    "description": (
                        "Construct and execute the smallest comparison that can test the primary "
                        "hypothesis within the accepted budget."
                    ),
                    "method": "preregistered_computational_comparison",
                    "inputs": ["Public evidence directly relevant to the accepted question"],
                    "parameters": {
                        "pilot": True,
                        "captured_provenance_required": True,
                        "robustness_check_required": True,
                        "researcher_guidance": guidance,
                    },
                    "expected_output": "A result table, robustness result, and evidence lineage.",
                    "statistical_tests": ["predeclared primary comparison", "sensitivity check"],
                }
            ],
            statistical_framework=(
                "Report effect direction, uncertainty, and sensitivity; do not promote an "
                "inference when the available evidence is insufficient."
            ),
            expected_outputs=[
                "evidence inventory",
                "executed analysis artifact",
                "robustness assessment",
                "limitations and provenance record",
                *(["researcher guidance compliance note"] if guidance else []),
            ],
            risks=[
                "Public evidence may be insufficient or inaccessible within the pilot window.",
                "A null or inconclusive result remains a valid outcome.",
                "The bounded pilot cannot establish broader novelty on its own.",
            ],
            estimated_compute_cost=min(5.0, mission.constraints.compute_budget),
            estimated_time_hours=min(1.0, max(1 / 60, mission_duration_minutes(mission) / 60)),
        )

    def _bound_plan(
        self, mission: Mission, plan: ResearchPlan
    ) -> tuple[ResearchPlan, dict[str, Any]]:
        """Keep model-generated work inside the service's interactive execution envelope."""
        bounded = plan.model_copy(deep=True)
        before = {
            "literature_queries": len(bounded.literature_queries),
            "data_requirements": len(bounded.data_requirements),
            "analysis_steps": len(bounded.analysis_steps),
            "estimated_compute_cost": bounded.estimated_compute_cost,
            "estimated_time_hours": bounded.estimated_time_hours,
        }
        bounded.literature_queries = bounded.literature_queries[: self.max_literature_queries]
        bounded.data_requirements = bounded.data_requirements[: self.max_data_requirements]
        bounded.analysis_steps = bounded.analysis_steps[: self.max_analysis_steps]
        bounded.estimated_compute_cost = min(
            max(0.0, bounded.estimated_compute_cost),
            max(0.01, mission.constraints.compute_budget),
        )
        bounded.estimated_time_hours = min(
            max(0.0, bounded.estimated_time_hours),
            mission_duration_minutes(mission) / 60,
        )
        after = {
            "literature_queries": len(bounded.literature_queries),
            "data_requirements": len(bounded.data_requirements),
            "analysis_steps": len(bounded.analysis_steps),
            "estimated_compute_cost": bounded.estimated_compute_cost,
            "estimated_time_hours": bounded.estimated_time_hours,
        }
        changes = {
            key: {"before": before[key], "after": after[key]}
            for key in before
            if before[key] != after[key]
        }
        return bounded, changes

    async def record_phase(self, investigation_id: str, phase: str, summary: str) -> None:
        node_key = PHASE_TO_NODE.get(phase)
        if not node_key:
            return
        marker = (investigation_id, phase)
        if summary == f"Starting {phase}" or marker not in self._phase_started:
            self._phase_started.add(marker)
            self.store.update_node(investigation_id, node_key, "running")
            self.store.update_investigation(
                investigation_id, status="running", current_node=node_key
            )
            self._event(
                investigation_id, "node.started", "running", humanize_start(phase), node_id=node_key
            )
            return

        failed = summary.lower().startswith("phase failed")
        self.store.update_node(
            investigation_id, node_key, "failed" if failed else "complete", summary
        )
        self._event(
            investigation_id,
            "node.failed" if failed else "node.completed",
            "failed" if failed else "complete",
            summary,
            node_id=node_key,
        )

    async def _run(self, investigation_id: str) -> None:
        investigation = self.store.get_investigation(investigation_id)
        if not investigation:
            return
        mission = self._mission(investigation_id, investigation)
        plan = ResearchPlan.model_validate(investigation["plan"])
        run = self.store.current_experiment_run(investigation_id)
        if not run or run["status"] != "queued":
            run = self._queue_experiment_run(investigation_id, investigation["plan"])
        environment = self._environment_manifest(plan, investigation)
        environment_digest = self._environment_digest(environment)
        mission.metadata["random_seed"] = run["random_seed"]
        mission.metadata["environment_digest"] = environment_digest
        self.store.update_experiment(investigation_id, status="running")
        self.store.update_experiment_run(
            run["id"],
            status="running",
            environment_digest=environment_digest,
            started_at=utc_now(),
            assertions=[
                {
                    "name": "human_plan_approval",
                    "passed": True,
                    "detail": "Execution began only after the preregistered plan was approved.",
                }
            ],
            metrics={"environment": environment, "attempt": run["attempt"]},
        )
        self.store.update_investigation(
            investigation_id,
            mission_json=mission.model_dump_json(),
            status="running",
            error=None,
        )
        self._event(
            investigation_id,
            "investigation.started",
            "running",
            "ApolloBot started the approved investigation.",
            data=self._model_stamp(self._model_route(investigation)),
        )

        try:
            run_config = self.config.model_copy(deep=True)
            run_config.output_dir = str(
                self.output_dir
                / investigation_id
                / "attempts"
                / f"attempt-{int(run['attempt']):04d}"
            )
            Path(run_config.output_dir).mkdir(parents=True, exist_ok=True)
            route = self._model_route(investigation)
            orchestrator = Orchestrator(
                config=run_config,
                interactive=False,
                llm_factory=lambda: self._llm_for_route(route),
            )
            orchestrator.checkpoint = EventCheckpointHandler(investigation_id, self)
            session = await orchestrator.run_discover(mission, plan=plan)
            artifacts = await self._capture_artifacts(investigation_id, session.session_dir)
            discovery_assessment = assess_discovery(
                getattr(session, "translation_scores", {}), session.hypotheses_status
            )
            related_literature = summarize_related_literature(session.literature_corpus)
            status = "complete" if session.current_phase == Phase.COMPLETE else "failed"
            current_node = (
                "prepare_replication_kit"
                if status == "complete"
                else investigation.get("current_node", "")
            )
            if status == "complete":
                self.store.update_node(
                    investigation_id,
                    "prepare_replication_kit",
                    "complete",
                    "Replication materials and checksums captured.",
                )
            self.store.update_investigation(
                investigation_id,
                status=status,
                current_node=current_node,
                cost_usd=session.cost.total_cost,
                result_json=json.dumps(
                    {
                        "key_findings": session.key_findings,
                        "warnings": session.warnings,
                        "datasets": session.datasets,
                        "literature_count": len(session.literature_corpus),
                        "related_literature": related_literature,
                        "hypotheses_status": session.hypotheses_status,
                        "discovery_assessment": discovery_assessment,
                        "model": self._model_stamp(route),
                        "phase_results": {
                            key: value.model_dump(mode="json")
                            for key, value in session.phase_results.items()
                        },
                    }
                ),
                completed_at=utc_now(),
            )
            code_artifact = next(
                (artifact["id"] for artifact in artifacts if artifact["artifact_type"] == "code"),
                None,
            )
            result_artifact = next(
                (
                    artifact["id"]
                    for artifact in artifacts
                    if artifact["artifact_type"] in {"figure", "manuscript", "replication-kit"}
                ),
                None,
            )
            self.store.update_experiment_run(
                run["id"],
                status=status,
                code_artifact_id=code_artifact,
                result_artifact_id=result_artifact,
                exit_code=0 if status == "complete" else 1,
                assertions=[
                    {
                        "name": "human_plan_approval",
                        "passed": True,
                        "detail": "Execution began only after the preregistered plan was approved.",
                    },
                    {
                        "name": "pipeline_complete",
                        "passed": status == "complete",
                        "detail": f"ApolloBot ended in the {session.current_phase.value} phase.",
                    },
                    {
                        "name": "artifacts_captured",
                        "passed": bool(artifacts),
                        "detail": f"Indexed {len(artifacts)} generated artifacts.",
                    },
                ],
                metrics={
                    "environment": environment,
                    "attempt": run["attempt"],
                    "cost_usd": session.cost.total_cost,
                    "key_findings": session.key_findings,
                    "warnings": session.warnings,
                    "dataset_count": len(session.datasets),
                    "literature_count": len(session.literature_corpus),
                    "hypotheses_status": session.hypotheses_status,
                    "discovery_assessment": discovery_assessment,
                },
                completed_at=utc_now(),
            )
            self.store.update_experiment(investigation_id, status=status)
            self._event(
                investigation_id,
                f"investigation.{status}",
                status,
                "Investigation completed and artifacts were captured."
                if status == "complete"
                else f"Investigation ended in {session.current_phase.value} state.",
            )
        except asyncio.CancelledError:
            current = self.store.get_investigation(investigation_id)
            if current and current["status"] not in {"paused", "cancelled"}:
                current_node = str(current.get("current_node") or "search_literature")
                self.store.update_node(
                    investigation_id,
                    current_node,
                    "ready",
                    "Worker shutdown interrupted this attempt. Explicit resume is required.",
                )
                self.store.update_investigation(
                    investigation_id,
                    status="paused",
                    error="Worker shutdown interrupted execution; explicit resume is required.",
                )
                self._end_current_run(
                    investigation_id,
                    "failed",
                    "Worker shutdown interrupted this attempt before completion.",
                    exit_code=1,
                )
                self.store.update_experiment(investigation_id, status="approved")
                self._event(
                    investigation_id,
                    "worker.interrupted_execution",
                    "paused",
                    (
                        "Worker shutdown interrupted execution. The approved plan is preserved "
                        "and will not rerun without explicit approval."
                    ),
                    node_id=current_node,
                )
            raise
        except Exception as error:
            self.store.update_investigation(
                investigation_id, status="failed", error=str(error), completed_at=utc_now()
            )
            self._end_current_run(
                investigation_id,
                "failed",
                f"Execution failed: {str(error)[:300]}",
                exit_code=1,
            )
            self.store.update_experiment(investigation_id, status="failed")
            self._event(
                investigation_id,
                "investigation.failed",
                "failed",
                f"Investigation failed: {str(error)[:300]}",
            )

    async def _capture_artifacts(
        self, investigation_id: str, session_dir: Path
    ) -> list[dict[str, Any]]:
        artifacts: list[dict[str, Any]] = []
        if not session_dir.exists():
            return artifacts
        investigation_root = (self.output_dir / investigation_id).resolve()
        resolved_session = session_dir.resolve()
        if investigation_root not in resolved_session.parents:
            raise ValueError("Run output escaped its investigation artifact root")
        restricted_paths: set[str] = set()
        redistributable_raw_paths: set[str] = set()
        manifest_valid = False
        access_manifest = session_dir / "data" / "access-manifest.json"
        if access_manifest.is_file():
            try:
                manifest = json.loads(access_manifest.read_text())
                if (
                    not isinstance(manifest, dict)
                    or manifest.get("schema") != "frontier-data-access/v1"
                ):
                    raise ValueError("Unsupported data access manifest")
                datasets = manifest.get("datasets")
                if not isinstance(datasets, list):
                    raise ValueError("Invalid data access manifest datasets")
                for item in datasets:
                    if not isinstance(item, dict):
                        raise ValueError("Invalid data access manifest entry")
                    local_path = item.get("local_path")
                    if not local_path:
                        continue
                    normalized = Path(str(local_path))
                    if normalized.is_absolute() or ".." in normalized.parts:
                        raise ValueError("Data access manifest path escaped the run")
                    relative = normalized.as_posix().removeprefix("./")
                    if item.get("redistribution_allowed") is True:
                        redistributable_raw_paths.add(relative)
                    else:
                        restricted_paths.add(relative)
                manifest_valid = True
            except (OSError, ValueError, TypeError):
                # A missing or malformed rights declaration is not permission.
                # Raw datasets fail closed; generated code, figures, logs, and
                # the manifest itself remain available for reproducibility.
                restricted_paths = set()
                redistributable_raw_paths = set()
        for path in session_dir.rglob("*"):
            if not path.is_file():
                continue
            session_relative = str(path.resolve().relative_to(resolved_session))
            if session_relative in restricted_paths:
                continue
            if session_relative.startswith("data/raw/") and (
                not manifest_valid or session_relative not in redistributable_raw_paths
            ):
                continue
            relative = str(path.resolve().relative_to(investigation_root))
            artifact_id = str(uuid4())
            digest = await asyncio.to_thread(sha256_file, path)
            artifact = {
                "id": artifact_id,
                "artifact_type": artifact_type(relative),
                "label": path.name,
                "path": relative,
                "media_type": mimetypes.guess_type(path.name)[0] or "application/octet-stream",
                "size_bytes": path.stat().st_size,
                "checksum_sha256": digest,
            }
            artifacts.append(self.store.add_artifact(investigation_id, artifact))
        self._event(
            investigation_id,
            "artifacts.captured",
            "complete",
            "Generated files, logs, figures, and replication materials were indexed.",
            data={"artifact_count": len(artifacts)},
        )
        return artifacts

    async def _capture_preregistration(
        self, investigation_id: str, plan: ResearchPlan
    ) -> dict[str, Any]:
        root = self.output_dir / investigation_id
        path = root / "preregistration" / "research-plan.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        content = plan.model_dump_json(indent=2)
        await asyncio.to_thread(path.write_text, content)
        artifact = {
            "id": str(uuid5(UUID(investigation_id), "preregistration-plan")),
            "artifact_type": "provenance",
            "label": "Preregistered research plan",
            "path": str(path.relative_to(root)),
            "media_type": "application/json",
            "size_bytes": path.stat().st_size,
            "checksum_sha256": await asyncio.to_thread(sha256_file, path),
        }
        persisted = self.store.add_artifact(investigation_id, artifact)
        self._event(
            investigation_id,
            "experiment.preregistered",
            "complete",
            "Experiment methods, controls, and decision criteria were preregistered.",
            node_id="design_experiment",
            data={"artifact_id": persisted["id"]},
        )
        return persisted

    def _queue_experiment_run(
        self, investigation_id: str, plan_payload: dict[str, Any]
    ) -> dict[str, Any]:
        plan = ResearchPlan.model_validate(plan_payload)
        plan.assert_executable()
        snapshot = self.store.snapshot(investigation_id)
        if not snapshot:
            raise ValueError("Investigation not found while queuing its experiment")
        if not snapshot["experiments"]:
            self.store.upsert_experiment(
                investigation_id,
                self._experiment_from_plan(
                    investigation_id,
                    snapshot["investigation"],
                    plan,
                    utc_now(),
                ),
            )
        run_id = str(uuid4())
        seed = int(hashlib.sha256(run_id.encode()).hexdigest()[:8], 16)
        environment = self._environment_manifest(plan, snapshot["investigation"])
        self.store.update_experiment(investigation_id, status="approved")
        return self.store.create_experiment_run(
            investigation_id,
            run_id,
            environment_digest=self._environment_digest(environment),
            random_seed=seed,
        )

    def _end_current_run(
        self,
        investigation_id: str,
        status: str,
        detail: str,
        *,
        exit_code: int,
    ) -> None:
        run = self.store.current_experiment_run(investigation_id)
        if not run or run["status"] not in {"queued", "running"}:
            return
        assertions = list(run.get("assertions") or [])
        assertions.append({"name": "attempt_completed", "passed": False, "detail": detail})
        metrics = dict(run.get("metrics") or {})
        metrics["termination_reason"] = detail
        self.store.update_experiment_run(
            run["id"],
            status=status,
            exit_code=exit_code,
            assertions=assertions,
            metrics=metrics,
            completed_at=utc_now(),
        )

    @staticmethod
    def _experiment_from_plan(
        investigation_id: str,
        investigation: dict[str, Any],
        plan: ResearchPlan,
        preregistered_at: str,
    ) -> dict[str, Any]:
        first_hypothesis = plan.hypotheses[0] if plan.hypotheses else {}
        hypothesis = first_hypothesis.get("hypothesis") or investigation["objective"]
        controls: list[dict[str, Any]] = []
        if plan.statistical_framework:
            controls.append(
                {"type": "statistical_framework", "description": plan.statistical_framework}
            )
        for step in plan.analysis_steps:
            for test in step.statistical_tests:
                controls.append(
                    {"type": "statistical_test", "step": step.name, "description": test}
                )
        success_criteria = [
            {
                "hypothesis": item.get("hypothesis", ""),
                "criterion": item.get("test", "A preregistered test supports the hypothesis."),
            }
            for item in plan.hypotheses
        ] or [{"criterion": "The preregistered analysis produces an interpretable outcome."}]
        failure_criteria = [
            {
                "hypothesis": item.get("hypothesis", ""),
                "criterion": item.get(
                    "null_hypothesis", "The primary test does not support the claim."
                ),
            }
            for item in plan.hypotheses
        ]
        failure_criteria.extend({"risk": risk} for risk in plan.risks)
        return {
            "id": str(uuid5(UUID(investigation_id), "primary-experiment")),
            "node_key": "design_experiment",
            "title": plan.summary or investigation["title"],
            "hypothesis": hypothesis,
            "method": {
                "approach": plan.approach,
                "literature_queries": plan.literature_queries,
                "data_requirements": [
                    requirement.model_dump(mode="json") for requirement in plan.data_requirements
                ],
                "analysis_steps": [step.model_dump(mode="json") for step in plan.analysis_steps],
                "expected_outputs": plan.expected_outputs,
            },
            "controls": controls,
            "success_criteria": success_criteria,
            "failure_criteria": failure_criteria,
            "preregistered_at": preregistered_at,
            "status": "draft",
        }

    def _environment_manifest(
        self, plan: ResearchPlan, investigation: dict[str, Any]
    ) -> dict[str, Any]:
        route = self._model_route(investigation)
        return {
            "apollobot_version": __version__,
            "sandbox_mode": os.getenv("APOLLOBOT_SANDBOX_MODE", "container"),
            "sandbox_image": os.getenv(
                "APOLLOBOT_SANDBOX_IMAGE", "frontier-science/apollobot-sandbox:py312"
            ),
            "model_provider": route.billing_provider,
            "model_provider_tag": route.provider_tag,
            "model": route.model_id,
            "model_catalog_version": MODEL_CATALOG_VERSION,
            "plan_sha256": hashlib.sha256(plan.model_dump_json().encode()).hexdigest(),
        }

    @staticmethod
    def _environment_digest(environment: dict[str, Any]) -> str:
        encoded = json.dumps(environment, separators=(",", ":"), sort_keys=True).encode()
        return f"sha256:{hashlib.sha256(encoded).hexdigest()}"

    def _mission(self, investigation_id: str, investigation: dict[str, Any]) -> Mission:
        route = self._model_route(investigation)
        if investigation.get("mission"):
            mission = Mission.model_validate(investigation["mission"])
            mission.metadata.update(
                model_id=route.model_id,
                model_provider_tag=route.provider_tag,
                model_catalog_version=MODEL_CATALOG_VERSION,
            )
            return mission
        check = QuestionCheck.model_validate(investigation["check"])
        return Mission(
            id=investigation_id,
            title=check.title,
            objective=check.question,
            hypotheses=check.hypotheses,
            mode=check.mode,
            domain=check.apollo_domain,
            constraints=Constraints(
                compute_budget=max(0.01, check.estimate.compute_usd),
                time_limit=f"{max(1, check.estimate.duration_minutes)}m",
                data_sources="public_only",
                ethics="observational_only",
            ),
            metadata={
                "output_dir": str(self.output_dir),
                "frontier_investigation_id": investigation_id,
                "sandbox_mode": os.getenv("APOLLOBOT_SANDBOX_MODE", "container"),
                "model_id": route.model_id,
                "model_provider_tag": route.provider_tag,
                "model_catalog_version": MODEL_CATALOG_VERSION,
            },
        )

    @staticmethod
    def _model_route(investigation: dict[str, Any]) -> ModelRoute:
        return resolve_model_route(
            investigation.get("model_id"), investigation.get("model_provider_tag")
        )

    def _llm_for_route(self, route: ModelRoute) -> LLMProvider:
        if (
            os.getenv("APOLLOBOT_ENV", "development").lower() == "acceptance"
            and self.config.api.default_provider == "acceptance"
        ):
            return create_llm("acceptance", self.config.api.get_key())
        api_key = self.config.api.openai_api_key
        if not api_key:
            raise RuntimeError("The OpenRouter API key is not configured")
        return OpenAIProvider(
            api_key=api_key,
            model=route.model_id,
            base_url=os.getenv("OPENAI_BASE_URL") or "https://openrouter.ai/api/v1",
            provider_name=route.billing_provider,
            provider_tag=route.provider_tag,
            data_collection="deny",
            reasoning_effort=route.reasoning_effort,
            reasoning_exclude=True,
            input_cost_per_million=route.input_cost_per_million,
            cached_input_cost_per_million=route.cached_input_cost_per_million,
            output_cost_per_million=route.output_cost_per_million,
            max_tokens=route.max_output_tokens,
        )

    @staticmethod
    def _model_stamp(route: ModelRoute) -> dict[str, Any]:
        return {
            "model_id": route.model_id,
            "provider_tag": route.provider_tag,
            "provider": route.billing_provider,
            "catalog_version": MODEL_CATALOG_VERSION,
        }

    def _event(
        self,
        investigation_id: str,
        event_type: str,
        status: str,
        summary: str,
        node_id: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> ServiceEvent:
        return self.store.append_event(
            ServiceEvent(
                investigation_id=investigation_id,
                node_id=node_id,
                sequence=0,
                event_type=event_type,
                status=status,
                public_summary=summary,
                data=data or {},
            )
        )


def humanize_start(phase: str) -> str:
    labels = {
        "literature_review": "Searching and mapping relevant literature.",
        "data_acquisition": "Evaluating and acquiring candidate datasets.",
        "analysis": "Executing the approved computational analysis.",
        "statistical_testing": "Running statistical and robustness checks.",
        "manuscript_drafting": "Drafting the structured research record.",
        "self_review": "Running adversarial methodological review.",
        "manuscript_revision": "Revising claims against the review findings.",
    }
    return labels.get(phase, f"Starting {phase.replace('_', ' ')}.")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_type(relative: str) -> str:
    parts = set(Path(relative).parts)
    if "figures" in parts:
        return "figure"
    if "analysis" in parts:
        return "code"
    if "data" in parts:
        return "dataset"
    if "provenance" in parts or relative.startswith("preregistration/"):
        return "provenance"
    if "review" in parts:
        return "review"
    if "replication_kit" in parts:
        return "replication-kit"
    if relative.endswith((".tex", ".pdf", ".md")):
        return "manuscript"
    return "file"


def assess_discovery(
    translation_scores: dict[str, Any] | None,
    hypotheses_status: dict[str, Any] | None,
) -> dict[str, Any]:
    """Create conservative, structured discovery triage from completed-run state."""
    scores = translation_scores or {}
    statuses = [str(value).lower() for value in (hypotheses_status or {}).values()]
    novelty = numeric_score(scores.get("novelty"))
    average = numeric_score(scores.get("average"))
    supported = sum(status == "supported" for status in statuses)
    rejected = sum(status == "rejected" for status in statuses)
    inconclusive = len(statuses) - supported - rejected
    if novelty is None:
        breakthrough_status = "not_assessed"
        rationale = "The run produced no calibrated novelty assessment."
    elif novelty >= 8 and supported > 0:
        breakthrough_status = "candidate"
        rationale = (
            "The model-assisted novelty triage was high and at least one preregistered "
            "hypothesis was supported. Independent review and reproduction are still required."
        )
    else:
        breakthrough_status = "not_established"
        rationale = (
            "The completed run does not meet the platform's candidate threshold; null, "
            "incremental, and inconclusive outcomes remain publishable."
        )
    return {
        "breakthrough_status": breakthrough_status,
        "novelty_score": novelty,
        "translation_score": average,
        "hypotheses": {
            "supported": supported,
            "rejected": rejected,
            "inconclusive": inconclusive,
        },
        "rationale": rationale,
        "disclaimer": "Triage only; neither scoring nor publication establishes a breakthrough.",
    }


def summarize_related_literature(corpus: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a small citation-safe literature index without copying source abstracts."""
    related: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in corpus:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or item.get("name") or "").strip()
        if not title:
            continue
        doi = str(item.get("doi") or item.get("DOI") or "").strip()
        key = (doi or title).lower()
        if key in seen:
            continue
        seen.add(key)
        url = str(item.get("url") or item.get("link") or "").strip()
        if url and not url.startswith(("https://", "http://")):
            url = ""
        related.append(
            {
                "title": title[:500],
                "doi": doi[:200] or None,
                "url": url[:2000] or None,
                "year": item.get("year") or item.get("publication_year"),
                "source": str(item.get("source") or item.get("server") or "literature index")[:100],
            }
        )
        if len(related) == 8:
            break
    return related


def numeric_score(value: object) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return min(10.0, max(0.0, score))


def environment_limit(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as error:
        raise RuntimeError(f"{name} must be an integer") from error
    if value < 1 or value > 64:
        raise RuntimeError(f"{name} must be between 1 and 64")
    return value


def environment_timeout(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as error:
        raise RuntimeError(f"{name} must be an integer") from error
    if value < 5 or value > 300:
        raise RuntimeError(f"{name} must be between 5 and 300")
    return value


def mission_duration_minutes(mission: Mission) -> int:
    value = mission.constraints.time_limit.strip().lower()
    try:
        if value.endswith("m"):
            return max(1, int(float(value[:-1])))
        if value.endswith("h"):
            return max(1, int(float(value[:-1]) * 60))
    except ValueError:
        pass
    return 60
