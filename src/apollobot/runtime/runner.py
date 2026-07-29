"""
ResearchRunner — the main continuous execution loop.

Analogous to OpenCat's AgentRunner. Uses a setTimeout-style chain
(asyncio.call_later) where each tick schedules the next one after
completing, enabling adaptive intervals.

Lifecycle:
    1. start() — load memory, register signals, begin scan loop
    2. tick()  — build state -> brain.reason() -> guardrails.check() -> execute
    3. stop()  — graceful shutdown
"""

from __future__ import annotations

import asyncio
import json
import logging
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console

from apollobot.agents import LLMProvider, create_llm
from apollobot.agents.orchestrator import Orchestrator
from apollobot.core import ApolloConfig, load_config
from apollobot.core.mission import Mission, ResearchMode
from apollobot.core.session import Phase, Session
from apollobot.runtime.brain import ResearchBrain
from apollobot.runtime.guardrails import ResearchGuardrails
from apollobot.runtime.config import RuntimeConfig
from apollobot.runtime.events import RunnerEvent, RunnerEventEmitter, RunnerEventType
from apollobot.runtime.storage import RunnerStorage
from apollobot.runtime.types import (
    ActionType,
    BrainAction,
    BrainDecision,
    RunnerState,
    SessionSummary,
)
from apollobot.runtime.health import HealthServer
from apollobot.runtime.metrics import compute_metrics
from apollobot.runtime.notify_bridge import NotifyBridge
from apollobot.runtime.pidfile import PidFile
from apollobot.runtime.trajectory import ResearchTrajectory
from apollobot.runtime.provenance import RuntimeProvenanceLogger
from apollobot.runtime.remote_log import RemoteLogTransport
from apollobot.runtime.watchdog import Watchdog

logger = logging.getLogger(__name__)
console = Console()


class ResearchRunner:
    """
    Continuous execution loop for autonomous research.

    Uses a setTimeout-style chain: each tick schedules the next one
    after completing, enabling adaptive intervals where the brain
    can request longer or shorter waits.
    """

    def __init__(
        self,
        runtime_config: RuntimeConfig,
        apollo_config: ApolloConfig | None = None,
    ) -> None:
        self.runtime_config = runtime_config
        self.apollo_config = apollo_config or load_config()

        # Storage
        self.storage = RunnerStorage(runtime_config.db_path)

        # LLM
        self.llm: LLMProvider = create_llm(
            provider=self.apollo_config.api.default_provider,
            api_key=self.apollo_config.api.get_key(),
        )

        # Brain
        self.brain = ResearchBrain(self.llm, self.storage, runtime_config)

        # Guardrails
        self.guardrails = ResearchGuardrails(runtime_config.guardrails, self.storage)

        # Watchdog
        self.watchdog = Watchdog(runtime_config.watchdog)

        # PID file
        self.pidfile = PidFile()

        # Event emitter
        self.events = RunnerEventEmitter()

        # Health server
        self.health = HealthServer(port=runtime_config.health_port)

        # Provenance
        self.provenance = RuntimeProvenanceLogger()

        # Remote logging (disabled by default — needs URL in config)
        self.remote_log = RemoteLogTransport(
            url=getattr(runtime_config, "remote_log_url", ""),
        )

        # Trajectory analysis (cross-session learning)
        self.trajectory = ResearchTrajectory(self.storage)

        # Notification bridge (runtime events → notification channels)
        self._notify_bridge: NotifyBridge | None = None

        # Orchestrator (for running sessions — reuses existing machinery)
        self.orchestrator = Orchestrator(config=self.apollo_config, interactive=False)

        # State
        self.tick_count = 0
        self.running = False
        self.shutting_down = False
        self.tick_in_progress = False
        self.start_time: float = 0.0
        self._next_tick_handle: asyncio.TimerHandle | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Initialize and begin the scan loop."""
        if not self.pidfile.acquire():
            _running, existing_pid = self.pidfile.is_running()
            msg = f"Another runtime is already running (PID: {existing_pid})"
            console.print(f"[red]{msg}[/red]")
            raise RuntimeError(msg)

        self.running = True
        self.start_time = time.monotonic()

        # Load persistent brain memory
        await self.brain.load_memory()

        # Start health server
        self.health.running = True
        self.health.start_time = self.start_time
        self.health.domain = self.runtime_config.domain
        self.health.on_guardrails_update = self._handle_guardrails_update
        await self.health.start()

        # Register signal handlers
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(self.stop(str(s))))

        # Print startup with metrics
        metrics = compute_metrics(self.storage)
        console.print(f"\n[bold green]ApolloBot Runtime Started[/bold green]")
        console.print(f"  Domain: {self.runtime_config.domain}")
        console.print(f"  Interval: {self.runtime_config.default_interval}s")
        console.print(
            f"  Daily budget: ${self.runtime_config.guardrails.daily_compute_budget_usd:.2f}"
        )
        console.print(f"  Max concurrent: {self.runtime_config.guardrails.max_concurrent_sessions}")
        console.print(f"  Health: http://0.0.0.0:{self.runtime_config.health_port}/health")
        if metrics.total_sessions > 0:
            console.print(
                f"  History: {metrics.completed_sessions} papers, "
                f"reputation {metrics.reputation_score}/100"
            )
        if self.runtime_config.dry_run:
            console.print(f"  [yellow]DRY RUN — no real API calls[/yellow]")
        if self.runtime_config.user_instructions:
            console.print(f"  Instructions: {self.runtime_config.user_instructions[:80]}...")
        console.print()

        # Start notification bridge if notifications configured
        if self.apollo_config.notifications.enabled:
            try:
                router = self.orchestrator.router
                self._notify_bridge = NotifyBridge(self.events, router)
                await self._notify_bridge.start()
            except Exception:
                logger.debug("Notification bridge setup failed", exc_info=True)

        # Start remote logging
        await self.remote_log.start()

        # Log provenance
        self.provenance.log_lifecycle(
            "runtime_started",
            {
                "domain": self.runtime_config.domain,
                "daily_budget": self.runtime_config.guardrails.daily_compute_budget_usd,
                "dry_run": self.runtime_config.dry_run,
            },
        )

        await self.events.emit(
            RunnerEvent(
                RunnerEventType.RUNTIME_STARTED, tick=0, data={"domain": self.runtime_config.domain}
            )
        )

        # First tick immediately
        await self._tick()

    async def stop(self, reason: str = "shutdown") -> None:
        """Graceful shutdown."""
        if self.shutting_down:
            return
        self.shutting_down = True
        self.running = False

        console.print(f"\n[yellow]Runtime stopping: {reason}[/yellow]")

        # Cancel scheduled next tick
        if self._next_tick_handle:
            self._next_tick_handle.cancel()
            self._next_tick_handle = None

        # Wait for in-progress tick
        while self.tick_in_progress:
            await asyncio.sleep(0.1)

        # Stop health server
        self.health.running = False
        await self.health.stop()

        # Print final metrics
        metrics = compute_metrics(self.storage)
        if metrics.total_sessions > 0:
            console.print(
                f"  Papers: {metrics.completed_sessions} | "
                f"Cost: ${metrics.total_cost_usd:.2f} | "
                f"Reputation: {metrics.reputation_score}/100"
            )

        # Stop notification bridge
        if self._notify_bridge:
            await self._notify_bridge.stop()

        # Release PID file
        self.pidfile.release()

        # Stop remote logging
        await self.remote_log.stop()

        # Log provenance
        self.provenance.log_lifecycle(
            "runtime_stopped",
            {
                "reason": reason,
                "tick_count": self.tick_count,
            },
        )

        # Cleanup
        self.storage.close()

        await self.events.emit(
            RunnerEvent(
                RunnerEventType.RUNTIME_STOPPED, tick=self.tick_count, data={"reason": reason}
            )
        )

        console.print(f"[dim]Runtime stopped after {self.tick_count} ticks[/dim]")

    # ------------------------------------------------------------------
    # Main tick loop
    # ------------------------------------------------------------------

    async def _tick(self) -> None:
        """Single scan cycle."""
        if self.tick_in_progress or not self.running:
            return

        self.tick_in_progress = True
        self.tick_count += 1
        next_interval = self.runtime_config.default_interval

        try:
            await self.events.emit(RunnerEvent(RunnerEventType.TICK_START, tick=self.tick_count))

            # Check circuit breaker
            if not self.watchdog.should_attempt():
                console.print(f"[dim]Tick {self.tick_count}: watchdog open, skipping[/dim]")
                await self.events.emit(
                    RunnerEvent(RunnerEventType.WATCHDOG_OPENED, tick=self.tick_count)
                )
                next_interval = int(self.runtime_config.watchdog.cooldown_seconds)
                return

            # Build state
            state = self._build_state()

            # Budget warning at 80%
            budget = self.runtime_config.guardrails.daily_compute_budget_usd
            if budget > 0 and state.daily_cost_usd > 0.8 * budget:
                await self.events.emit(
                    RunnerEvent(
                        RunnerEventType.BUDGET_WARNING,
                        tick=self.tick_count,
                        data={"daily_cost": state.daily_cost_usd, "budget": budget},
                    )
                )

            console.print(
                f"[bold]Tick {self.tick_count}[/bold] | "
                f"sessions: {len(state.active_sessions)} active, "
                f"{state.total_papers} papers | "
                f"${state.daily_cost_usd:.2f} today"
            )

            # Brain reasoning
            decision = await self.brain.reason(state)

            # Log decision to provenance
            self.provenance.log_decision(
                tick=self.tick_count,
                reasoning=decision.reasoning,
                actions=[a.type.value for a in decision.actions],
                next_check_in=decision.next_check_in,
                memory_updates=decision.memory,
            )

            console.print(f"  [dim]Brain: {decision.reasoning[:100]}[/dim]")

            # Process actions
            for action in decision.actions:
                enforcement = self.guardrails.check(action, state)
                if not enforcement.allowed:
                    console.print(
                        f"  [red]Blocked[/red]: {action.type.value} — {enforcement.reason}"
                    )
                    self.brain.record_action_result(
                        self.tick_count, action, "blocked", enforcement.reason
                    )
                    self.provenance.log_enforcement(
                        tick=self.tick_count,
                        action_type=action.type.value,
                        allowed=False,
                        reason=enforcement.reason,
                        objective=action.objective or "",
                        domain=action.domain or "",
                    )
                    await self.events.emit(
                        RunnerEvent(
                            RunnerEventType.ACTION_BLOCKED,
                            tick=self.tick_count,
                            data={"action": action.type.value, "reason": enforcement.reason},
                        )
                    )
                    continue

                await self._execute_action(action)
                await self.events.emit(
                    RunnerEvent(
                        RunnerEventType.ACTION_EXECUTED,
                        tick=self.tick_count,
                        data={"action": action.type.value, "objective": action.objective},
                    )
                )

            # Adaptive interval
            next_interval = max(
                self.runtime_config.min_interval,
                min(self.runtime_config.max_interval, decision.next_check_in),
            )

            self.watchdog.record_success()

            await self.events.emit(RunnerEvent(RunnerEventType.TICK_COMPLETE, tick=self.tick_count))

        except Exception as e:
            logger.exception("Tick %d failed", self.tick_count)
            console.print(f"  [red]Tick failed: {e}[/red]")
            self.watchdog.record_failure()
            next_interval = self.runtime_config.error_interval
            await self.events.emit(
                RunnerEvent(
                    RunnerEventType.TICK_FAILED, tick=self.tick_count, data={"error": str(e)}
                )
            )

        finally:
            self.tick_in_progress = False
            # Update health server state
            self.health.update(
                tick_count=self.tick_count,
                last_tick_time=datetime.now(timezone.utc).isoformat(),
                watchdog_state=self.watchdog.state.value,
                active_sessions=len(self.storage.get_active_sessions()),
                total_papers=len(self.storage.get_completed_sessions(limit=10000)),
                daily_cost=self.storage.daily_spend(),
            )

        # Schedule next tick
        if self.running:
            self._schedule_next_tick(next_interval)

    def _schedule_next_tick(self, interval_seconds: int) -> None:
        """Schedule the next tick using asyncio."""
        if not self.running or self.shutting_down:
            return
        console.print(f"  [dim]Next tick in {interval_seconds}s[/dim]")
        loop = asyncio.get_running_loop()
        self._next_tick_handle = loop.call_later(
            interval_seconds,
            lambda: asyncio.create_task(self._tick()),
        )

    # ------------------------------------------------------------------
    # State assembly
    # ------------------------------------------------------------------

    def _build_state(self) -> RunnerState:
        """Assemble the full state snapshot for the brain."""
        active = self.storage.get_active_sessions()
        completed = self.storage.get_completed_sessions(limit=20)
        failed = self.storage.get_failed_sessions(limit=5)

        total_papers = len(completed)
        total_cost = sum(s.cost_usd for s in completed) + sum(s.cost_usd for s in active)
        daily_cost = self.storage.daily_spend()
        daily_sessions = self.storage.sessions_started_today()

        # Trajectory analysis every 10 ticks (expensive query)
        trajectory_summary = ""
        if self.tick_count % 10 == 1 or self.tick_count == 1:
            try:
                analysis = self.trajectory.analyze(self.runtime_config.guardrails.approved_domains)
                trajectory_summary = self.trajectory.format_for_brain(analysis)
            except Exception:
                logger.debug("Trajectory analysis failed", exc_info=True)

        return RunnerState(
            tick_number=self.tick_count,
            uptime_seconds=time.monotonic() - self.start_time,
            domain=self.runtime_config.domain,
            active_sessions=active,
            completed_sessions=completed,
            failed_sessions=failed,
            total_papers=total_papers,
            total_cost_usd=total_cost,
            daily_cost_usd=daily_cost,
            daily_sessions_started=daily_sessions,
            guardrails_remaining_budget=(
                self.runtime_config.guardrails.daily_compute_budget_usd - daily_cost
            ),
            guardrails_max_concurrent=self.runtime_config.guardrails.max_concurrent_sessions,
            watchdog_state=self.watchdog.state.value,
            memory=self.brain.memory,
            trajectory_summary=trajectory_summary,
        )

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------

    async def _execute_action(self, action: BrainAction) -> None:
        """Route brain decisions to the appropriate executor."""
        console.print(f"  [green]Executing[/green]: {action.type.value}", end="")
        if action.objective:
            console.print(f" — {action.objective[:60]}", end="")
        console.print()

        if action.type == ActionType.START_RESEARCH:
            await self._start_research(action)
        elif action.type == ActionType.SCAN_LITERATURE:
            await self._scan_literature(action)
        elif action.type == ActionType.REVIEW_SESSION:
            await self._review_session(action)
        elif action.type == ActionType.AUTO_REVIEW:
            await self._auto_review(action)
        elif action.type == ActionType.AUTO_SUBMIT:
            await self._auto_submit(action)
        elif action.type == ActionType.IDLE:
            self.brain.record_action_result(
                self.tick_count, action, "completed", "Idle — no action taken"
            )

    async def _start_research(self, action: BrainAction) -> None:
        """Start a new research session via the existing Orchestrator."""
        domain = action.domain or self.runtime_config.domain
        mode = action.mode or "hypothesis"

        mission = Mission.from_objective(
            action.objective,
            mode=mode,
            domain=domain,
        )

        # Register in session tracker
        self.storage.register_session(
            SessionSummary(
                session_id=mission.id,
                objective=mission.objective,
                domain=domain,
                mode=mode,
                phase="planning",
                started_at=datetime.now(timezone.utc).isoformat(),
            )
        )
        await self.events.emit(
            RunnerEvent(
                RunnerEventType.SESSION_STARTED,
                tick=self.tick_count,
                data={"session_id": mission.id, "objective": mission.objective, "domain": domain},
            )
        )

        try:
            if self.runtime_config.dry_run:
                console.print(f"    [yellow]DRY RUN: would start session {mission.id}[/yellow]")
                self.storage.update_session(mission.id, phase="complete")
                self.brain.record_action_result(
                    self.tick_count,
                    action,
                    "completed",
                    f"Dry run — session {mission.id} simulated",
                )
                return

            # Run the session (this is the existing Orchestrator — unchanged)
            session = await self.orchestrator.run(mission)

            # Record results
            cost = session.cost.total_cost
            self.storage.record_spend(cost, "session")
            self.storage.update_session(
                mission.id,
                phase=session.current_phase.value,
                cost_usd=cost,
                completed_at=datetime.now(timezone.utc).isoformat(),
                key_findings=json.dumps(session.key_findings),
                translation_score=session.translation_scores.get("average", 0.0),
            )

            result = "completed" if session.current_phase == Phase.COMPLETE else "failed"
            self.brain.record_action_result(
                self.tick_count,
                action,
                result,
                f"Session {mission.id}: {session.current_phase.value}, "
                f"cost=${cost:.2f}, findings={len(session.key_findings)}",
            )

            console.print(f"    Session {mission.id}: {session.current_phase.value} (${cost:.2f})")

            evt = (
                RunnerEventType.SESSION_COMPLETED
                if result == "completed"
                else RunnerEventType.SESSION_FAILED
            )
            await self.events.emit(
                RunnerEvent(
                    evt,
                    tick=self.tick_count,
                    data={
                        "session_id": mission.id,
                        "phase": session.current_phase.value,
                        "cost": cost,
                    },
                )
            )

        except Exception as e:
            logger.exception("Session %s failed", mission.id)
            self.storage.update_session(mission.id, phase="failed")
            self.brain.record_action_result(self.tick_count, action, "failed", str(e))
            await self.events.emit(
                RunnerEvent(
                    RunnerEventType.SESSION_FAILED,
                    tick=self.tick_count,
                    data={"session_id": mission.id, "error": str(e)},
                )
            )

    async def _scan_literature(self, action: BrainAction) -> None:
        """Quick literature scan without a full session."""
        domain = action.domain or self.runtime_config.domain

        try:
            # Use MCP client to do a quick search
            from apollobot.mcp import MCPClient, MCPServerInfo
            from apollobot.mcp.servers.builtin import get_domain_pack

            mcp = MCPClient()
            servers = get_domain_pack(domain)
            for srv in servers:
                mcp.register(
                    MCPServerInfo(
                        name=srv.name,
                        url=srv.url,
                        description=srv.description,
                        domain=domain,
                        api_base=srv.api_base,
                    )
                )

            results = []
            for server in mcp.get_servers():
                try:
                    result = await mcp.query(
                        server.name,
                        "search",
                        {"query": action.objective, "limit": 10},
                    )
                    papers = result.get("papers", result.get("results", []))
                    results.extend(papers)
                except Exception:
                    pass

            details = f"Found {len(results)} papers"
            if results:
                titles = [p.get("title", "?") for p in results[:3] if isinstance(p, dict)]
                details += f": {'; '.join(t[:50] for t in titles)}"

            self.brain.record_action_result(self.tick_count, action, "completed", details)
            console.print(f"    {details}")

        except Exception as e:
            self.brain.record_action_result(self.tick_count, action, "failed", str(e))

    async def _review_session(self, action: BrainAction) -> None:
        """Re-examine a past session's findings."""
        # For now, just record that we reviewed it — the brain can
        # incorporate the findings via its memory
        self.brain.record_action_result(
            self.tick_count,
            action,
            "completed",
            f"Reviewed session {action.session_id}",
        )

    async def _auto_review(self, action: BrainAction) -> None:
        """Run AI review on a completed session."""
        session_id = action.session_id
        if not session_id:
            self.brain.record_action_result(
                self.tick_count, action, "failed", "No session_id specified"
            )
            return

        try:
            from apollobot.core import APOLLO_SESSIONS_DIR
            from apollobot.agents import create_llm
            from apollobot.review.submission import SubmissionReviewer

            sessions_dir = Path(APOLLO_SESSIONS_DIR)
            session_dir = sessions_dir / session_id

            # Find manuscript
            manuscript_text = ""
            for name in ("manuscript.md", "manuscript.tex"):
                candidate = session_dir / name
                if candidate.exists():
                    manuscript_text = candidate.read_text()
                    break

            if not manuscript_text:
                self.brain.record_action_result(
                    self.tick_count,
                    action,
                    "failed",
                    f"No manuscript found for session {session_id}",
                )
                return

            llm = create_llm(
                self.apollo_config.api.default_provider,
                self.apollo_config.api.get_key(),
            )
            reviewer = SubmissionReviewer(llm)

            prov_path = session_dir / "provenance"
            report = await reviewer.review(
                manuscript_text,
                provenance_path=prov_path if prov_path.exists() else None,
                session_id=session_id,
            )

            details = (
                f"Review complete: {report.recommendation}, confidence={report.confidence:.2f}"
            )
            self.brain.record_action_result(self.tick_count, action, "completed", details)
            self.provenance.log_action_result(
                tick=self.tick_count,
                action_type="auto_review",
                result="completed",
                details=details,
                session_id=session_id,
            )
            console.print(f"    {details}")

        except Exception as e:
            self.brain.record_action_result(self.tick_count, action, "failed", str(e))

    async def _auto_submit(self, action: BrainAction) -> None:
        """Record that unauthenticated legacy auto-submission is intentionally disabled."""
        session_id = action.session_id
        if not session_id:
            self.brain.record_action_result(
                self.tick_count, action, "failed", "No session_id specified"
            )
            return

        details = (
            "Legacy auto-submission is disabled. Publish authenticated managed "
            "investigations through Frontier Science living records."
        )
        self.brain.record_action_result(self.tick_count, action, "failed", details)
        self.provenance.log_action_result(
            tick=self.tick_count,
            action_type="auto_submit",
            result="failed",
            details=details,
            session_id=session_id,
        )
        console.print(f"    {details}")

    # ------------------------------------------------------------------
    # Guardrails runtime updates
    # ------------------------------------------------------------------

    def _handle_guardrails_update(self, updates: dict) -> dict:
        """Handle guardrails config changes from the health server API."""
        applied = {}
        guardrails = self.runtime_config.guardrails
        for key, value in updates.items():
            if hasattr(guardrails, key):
                setattr(guardrails, key, value)
                applied[key] = value
                logger.info("Guardrails updated: %s = %s", key, value)
        if applied:
            # Recreate guardrails enforcer with updated config
            self.guardrails = ResearchGuardrails(guardrails, self.storage)
            self.provenance.log_lifecycle("guardrails_updated", applied)
        return applied


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------


async def run_continuous(
    domain: str = "bioinformatics",
    daily_budget: float = 100.0,
    max_concurrent: int = 3,
    interval: int = 300,
    user_instructions: str = "",
    health_port: int = 8080,
    dry_run: bool = False,
    db_path: str = "",
) -> None:
    """Start the continuous runtime. Blocks until shutdown signal."""
    from apollobot.runtime.config import GuardrailsConfig

    config = RuntimeConfig(
        domain=domain,
        default_interval=interval,
        guardrails=GuardrailsConfig(
            daily_compute_budget_usd=daily_budget,
            max_concurrent_sessions=max_concurrent,
        ),
        user_instructions=user_instructions,
        health_port=health_port,
        dry_run=dry_run,
        db_path=db_path,
    )

    runner = ResearchRunner(config)
    try:
        await runner.start()
        # Keep running until stopped
        while runner.running:
            await asyncio.sleep(1)
    except (KeyboardInterrupt, asyncio.CancelledError):
        await runner.stop("keyboard interrupt")
