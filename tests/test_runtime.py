"""
Tests for the continuous runtime components.

Tests storage, guardrails, watchdog, brain parsing, and runner state assembly
without requiring real LLM API calls.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from apollobot.runtime.config import GuardrailsConfig, RuntimeConfig, WatchdogConfig
from apollobot.runtime.events import RunnerEvent, RunnerEventEmitter, RunnerEventType
from apollobot.runtime.storage import RunnerStorage
from apollobot.runtime.guardrails import ResearchGuardrails
from apollobot.runtime.types import (
    ActionRecord,
    ActionType,
    BrainAction,
    BrainDecision,
    DecisionRecord,
    EnforcementResult,
    RunnerState,
    SessionSummary,
)
from apollobot.runtime.brain import ResearchBrain
from apollobot.runtime.health import HealthServer
from apollobot.runtime.metrics import ResearchMetrics, compute_metrics
from apollobot.runtime.pidfile import PidFile
from apollobot.runtime.watchdog import Watchdog, WatchdogState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary SQLite database."""
    return str(tmp_path / "test_runtime.db")


@pytest.fixture
def storage(tmp_db):
    s = RunnerStorage(tmp_db)
    yield s
    s.close()


@pytest.fixture
def guardrails(storage):
    config = GuardrailsConfig(
        daily_compute_budget_usd=100.0,
        max_concurrent_sessions=2,
        max_sessions_per_day=5,
        approved_domains=["bioinformatics", "physics"],
    )
    return ResearchGuardrails(config, storage)


@pytest.fixture
def base_state():
    return RunnerState(
        tick_number=1,
        uptime_seconds=60.0,
        domain="bioinformatics",
    )


# ---------------------------------------------------------------------------
# Storage tests
# ---------------------------------------------------------------------------


class TestStorage:
    def test_memory_round_trip(self, storage):
        storage.save_memory({"lead_1": "promising", "dead_end": "gene X"})
        memory = storage.load_memory()
        assert memory["lead_1"] == "promising"
        assert memory["dead_end"] == "gene X"

    def test_memory_update(self, storage):
        storage.save_memory({"key": "old"})
        storage.save_memory({"key": "new"})
        memory = storage.load_memory()
        assert memory["key"] == "new"

    def test_action_history(self, storage):
        storage.record_action(
            ActionRecord(
                tick=1,
                action_type="start_research",
                objective="test question",
                result="completed",
            )
        )
        storage.record_action(
            ActionRecord(
                tick=2,
                action_type="idle",
                result="completed",
            )
        )
        actions = storage.recent_actions(limit=10)
        assert len(actions) == 2
        assert actions[0].tick == 1
        assert actions[1].tick == 2

    def test_decision_history(self, storage):
        storage.record_decision(
            DecisionRecord(
                tick=1,
                reasoning="exploring new lead",
                actions=["start_research"],
                next_check_in=300,
            )
        )
        decisions = storage.recent_decisions(limit=5)
        assert len(decisions) == 1
        assert decisions[0].reasoning == "exploring new lead"

    def test_session_registry(self, storage):
        storage.register_session(
            SessionSummary(
                session_id="s-001",
                objective="test question",
                domain="bioinformatics",
                mode="hypothesis",
                phase="planning",
                started_at="2026-01-01T00:00:00",
            )
        )
        active = storage.get_active_sessions()
        assert len(active) == 1
        assert active[0].session_id == "s-001"

        storage.update_session("s-001", phase="complete", cost_usd=5.0)
        active = storage.get_active_sessions()
        assert len(active) == 0
        completed = storage.get_completed_sessions()
        assert len(completed) == 1
        assert completed[0].cost_usd == 5.0

    def test_spend_tracking(self, storage):
        storage.record_spend(10.0, "session")
        storage.record_spend(5.0, "session")
        assert storage.daily_spend() == 15.0

    def test_sessions_started_today(self, storage):
        storage.register_session(
            SessionSummary(
                session_id="s-today",
                objective="q",
                domain="bio",
                mode="h",
                phase="planning",
                started_at="2026-03-06T12:00:00",
            )
        )
        # Count depends on current date, but at least the registration works
        count = storage.sessions_started_today()
        assert isinstance(count, int)


# ---------------------------------------------------------------------------
# Guardrails tests
# ---------------------------------------------------------------------------


class TestGuardrails:
    def test_allows_idle(self, guardrails, base_state):
        action = BrainAction(type=ActionType.IDLE)
        result = guardrails.check(action, base_state)
        assert result.allowed

    def test_allows_valid_research(self, guardrails, base_state):
        action = BrainAction(
            type=ActionType.START_RESEARCH,
            objective="test",
            domain="bioinformatics",
        )
        result = guardrails.check(action, base_state)
        assert result.allowed

    def test_blocks_unapproved_domain(self, guardrails, base_state):
        action = BrainAction(
            type=ActionType.START_RESEARCH,
            objective="test",
            domain="astrology",
        )
        result = guardrails.check(action, base_state)
        assert not result.allowed
        assert "not in approved list" in result.reason

    def test_blocks_concurrent_limit(self, guardrails, base_state):
        base_state.active_sessions = [
            SessionSummary(session_id="s1", objective="q", domain="b", mode="h", phase="analysis"),
            SessionSummary(session_id="s2", objective="q", domain="b", mode="h", phase="analysis"),
        ]
        action = BrainAction(
            type=ActionType.START_RESEARCH,
            objective="test",
            domain="bioinformatics",
        )
        result = guardrails.check(action, base_state)
        assert not result.allowed
        assert "Concurrent session limit" in result.reason

    def test_blocks_budget_exhausted(self, guardrails, storage, base_state):
        storage.record_spend(100.0, "session")
        action = BrainAction(
            type=ActionType.START_RESEARCH,
            objective="test",
            domain="bioinformatics",
        )
        result = guardrails.check(action, base_state)
        assert not result.allowed
        assert "budget exhausted" in result.reason.lower()

    def test_emergency_stop(self, storage):
        config = GuardrailsConfig(emergency_stop=True)
        guardrails = ResearchGuardrails(config, storage)
        action = BrainAction(type=ActionType.START_RESEARCH, objective="test")
        state = RunnerState(tick_number=1, uptime_seconds=0, domain="bio")
        result = guardrails.check(action, state)
        assert not result.allowed
        assert "Emergency stop" in result.reason


# ---------------------------------------------------------------------------
# Watchdog tests
# ---------------------------------------------------------------------------


class TestWatchdog:
    def test_starts_closed(self):
        w = Watchdog(WatchdogConfig())
        assert w.state == WatchdogState.CLOSED
        assert w.should_attempt()

    def test_opens_after_threshold(self):
        w = Watchdog(WatchdogConfig(failure_threshold=3, cooldown_seconds=9999))
        w.record_failure()
        w.record_failure()
        assert w.state == WatchdogState.CLOSED
        w.record_failure()
        assert w.state == WatchdogState.OPEN
        assert not w.should_attempt()

    def test_half_open_after_cooldown(self):
        w = Watchdog(WatchdogConfig(failure_threshold=1, cooldown_seconds=0))
        w.record_failure()
        assert w.state == WatchdogState.OPEN
        # Cooldown is 0, so should transition to half-open
        assert w.should_attempt()
        assert w.state == WatchdogState.HALF_OPEN

    def test_half_open_success_closes(self):
        w = Watchdog(WatchdogConfig(failure_threshold=1, cooldown_seconds=0))
        w.record_failure()
        w.should_attempt()  # transitions to half-open
        w.record_success()
        assert w.state == WatchdogState.CLOSED

    def test_half_open_failure_reopens(self):
        w = Watchdog(WatchdogConfig(failure_threshold=1, cooldown_seconds=0))
        w.record_failure()
        w.should_attempt()  # transitions to half-open
        w.record_failure()
        assert w.state == WatchdogState.OPEN


# ---------------------------------------------------------------------------
# Brain decision parsing tests
# ---------------------------------------------------------------------------


class TestBrainDecision:
    def test_decision_model(self):
        d = BrainDecision(
            actions=[BrainAction(type=ActionType.START_RESEARCH, objective="test")],
            reasoning="testing",
            next_check_in=120,
            memory={"key": "value"},
        )
        assert len(d.actions) == 1
        assert d.actions[0].type == ActionType.START_RESEARCH
        assert d.next_check_in == 120

    def test_empty_decision(self):
        d = BrainDecision(reasoning="waiting")
        assert len(d.actions) == 0
        assert d.next_check_in == 300  # default

    def test_runner_state_model(self):
        s = RunnerState(
            tick_number=5,
            uptime_seconds=1500.0,
            domain="physics",
            total_papers=3,
            total_cost_usd=25.0,
        )
        assert s.tick_number == 5
        assert s.domain == "physics"


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestConfig:
    def test_defaults(self):
        c = RuntimeConfig()
        assert c.domain == "bioinformatics"
        assert c.default_interval == 300
        assert c.guardrails.daily_compute_budget_usd == 100.0
        assert c.watchdog.failure_threshold == 3
        assert not c.dry_run

    def test_custom_config(self):
        c = RuntimeConfig(
            domain="physics",
            default_interval=600,
            guardrails=GuardrailsConfig(daily_compute_budget_usd=50.0),
            user_instructions="Focus on dark matter",
            dry_run=True,
        )
        assert c.domain == "physics"
        assert c.guardrails.daily_compute_budget_usd == 50.0
        assert c.dry_run

    def test_health_port_default(self):
        c = RuntimeConfig()
        assert c.health_port == 8080

    def test_health_port_custom(self):
        c = RuntimeConfig(health_port=9090)
        assert c.health_port == 9090


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_empty_metrics(self, storage):
        m = compute_metrics(storage)
        assert m.total_sessions == 0
        assert m.reputation_score == 0.0

    def test_metrics_with_completed_sessions(self, storage):
        for i in range(3):
            storage.register_session(
                SessionSummary(
                    session_id=f"s-{i}",
                    objective=f"question {i}",
                    domain="bio",
                    mode="hypothesis",
                    phase="planning",
                    started_at=f"2026-03-0{i + 1}T12:00:00",
                )
            )
            storage.update_session(
                f"s-{i}",
                phase="complete",
                cost_usd=10.0,
                completed_at=f"2026-03-0{i + 1}T14:00:00",
                translation_score=8.0,
            )
        m = compute_metrics(storage)
        assert m.completed_sessions == 3
        assert m.failed_sessions == 0
        assert m.completion_rate == 1.0
        assert m.avg_translation_score == 8.0
        assert m.high_quality_papers == 3
        assert m.total_cost_usd == 30.0
        assert m.avg_cost_per_paper == 10.0
        assert m.reputation_score > 0

    def test_metrics_with_failures(self, storage):
        storage.register_session(
            SessionSummary(
                session_id="s-ok",
                objective="q",
                domain="bio",
                mode="h",
                phase="planning",
                started_at="2026-03-01T12:00:00",
            )
        )
        storage.update_session(
            "s-ok", phase="complete", cost_usd=5.0, completed_at="2026-03-01T14:00:00"
        )
        storage.register_session(
            SessionSummary(
                session_id="s-fail",
                objective="q",
                domain="bio",
                mode="h",
                phase="planning",
                started_at="2026-03-01T12:00:00",
            )
        )
        storage.update_session("s-fail", phase="failed")

        m = compute_metrics(storage)
        assert m.completed_sessions == 1
        assert m.failed_sessions == 1
        assert m.completion_rate == 0.5


# ---------------------------------------------------------------------------
# Health server tests
# ---------------------------------------------------------------------------


class TestHealthServer:
    def test_initial_state(self):
        h = HealthServer(port=8080)
        assert not h.running
        assert h.tick_count == 0
        assert h.watchdog_state == "closed"

    def test_update(self):
        h = HealthServer(port=8080)
        h.update(
            tick_count=5,
            last_tick_time="2026-03-06T12:00:00",
            watchdog_state="closed",
            active_sessions=2,
            total_papers=10,
            daily_cost=25.50,
        )
        assert h.tick_count == 5
        assert h.active_sessions == 2
        assert h.total_papers == 10
        assert h.daily_cost == 25.50

    def test_update_ignores_unknown_keys(self):
        h = HealthServer(port=8080)
        h.update(nonexistent_field="should_not_crash")
        assert not hasattr(h, "nonexistent_field")

    @pytest.mark.asyncio
    async def test_health_endpoint_healthy(self):
        from aiohttp.test_utils import TestClient, TestServer

        h = HealthServer(port=0)
        h.running = True
        h.start_time = 1000.0
        h.watchdog_state = "closed"
        h.domain = "bioinformatics"

        async with TestClient(TestServer(h._app)) as client:
            resp = await client.get("/health")
            assert resp.status == 200
            data = await resp.json()
            assert data["status"] == "healthy"
            assert data["running"] is True
            assert data["domain"] == "bioinformatics"

    @pytest.mark.asyncio
    async def test_health_endpoint_degraded(self):
        from aiohttp.test_utils import TestClient, TestServer

        h = HealthServer(port=0)
        h.running = True
        h.start_time = 1000.0
        h.watchdog_state = "open"  # circuit breaker open = degraded

        async with TestClient(TestServer(h._app)) as client:
            resp = await client.get("/health")
            assert resp.status == 503
            data = await resp.json()
            assert data["status"] == "degraded"


# ---------------------------------------------------------------------------
# Brain retry / fallback tests
# ---------------------------------------------------------------------------


class TestBrainRetry:
    @pytest.mark.asyncio
    async def test_failing_llm_returns_safe_fallback(self, storage):
        """When the LLM fails on every attempt, reason() returns a safe fallback."""
        from unittest.mock import AsyncMock, patch

        mock_llm = AsyncMock()
        mock_llm.complete_json = AsyncMock(side_effect=RuntimeError("connection refused"))

        config = RuntimeConfig()
        brain = ResearchBrain(llm=mock_llm, storage=storage, config=config)
        state = RunnerState(tick_number=1, uptime_seconds=60.0, domain="bioinformatics")

        with patch("apollobot.runtime.brain.asyncio.sleep", new_callable=AsyncMock):
            decision = await brain.reason(state)

        assert decision.actions == []
        assert decision.reasoning == "LLM unavailable \u2014 backing off"
        assert decision.next_check_in == 600
        # Should have been called LLM_CALL_MAX_ATTEMPTS (3) times
        assert mock_llm.complete_json.call_count == 3

    @pytest.mark.asyncio
    async def test_non_retryable_error_fails_immediately(self, storage):
        """Auth/budget errors are not retried."""
        from unittest.mock import AsyncMock, patch

        mock_llm = AsyncMock()
        mock_llm.complete_json = AsyncMock(
            side_effect=RuntimeError("authentication failed: invalid API key")
        )

        config = RuntimeConfig()
        brain = ResearchBrain(llm=mock_llm, storage=storage, config=config)
        state = RunnerState(tick_number=1, uptime_seconds=60.0, domain="bioinformatics")

        with patch("apollobot.runtime.brain.asyncio.sleep", new_callable=AsyncMock):
            decision = await brain.reason(state)

        assert decision.actions == []
        assert "Reasoning failed" in decision.reasoning
        assert decision.next_check_in == config.error_interval
        # Should NOT have retried
        assert mock_llm.complete_json.call_count == 1

    @pytest.mark.asyncio
    async def test_succeeds_on_second_attempt(self, storage):
        """If the LLM fails once then succeeds, the valid decision is returned."""
        from unittest.mock import AsyncMock, patch

        mock_llm = AsyncMock()
        mock_llm.complete_json = AsyncMock(
            side_effect=[
                RuntimeError("server error"),
                {"actions": [], "reasoning": "all clear", "nextCheckIn": 300},
            ]
        )

        config = RuntimeConfig()
        brain = ResearchBrain(llm=mock_llm, storage=storage, config=config)
        state = RunnerState(tick_number=1, uptime_seconds=60.0, domain="bioinformatics")

        with patch("apollobot.runtime.brain.asyncio.sleep", new_callable=AsyncMock):
            decision = await brain.reason(state)

        assert decision.reasoning == "all clear"
        assert decision.next_check_in == 300
        assert mock_llm.complete_json.call_count == 2


# ---------------------------------------------------------------------------
# PidFile tests
# ---------------------------------------------------------------------------


class TestPidFile:
    def test_acquire_release_cycle(self, tmp_path):
        pid_path = str(tmp_path / "test.pid")
        pf = PidFile(pid_path)

        assert pf.acquire() is True
        assert Path(pid_path).exists()

        # PID file should contain our PID
        import os

        assert Path(pid_path).read_text().strip() == str(os.getpid())

        pf.release()
        assert not Path(pid_path).exists()

    def test_stale_pid_overwritten(self, tmp_path):
        """A PID file with a dead process should be treated as stale."""
        pid_path = str(tmp_path / "test.pid")

        # Write a fake PID that almost certainly doesn't exist
        fake_pid = 4_000_000
        Path(pid_path).write_text(str(fake_pid))

        pf = PidFile(pid_path)
        # Should detect the stale PID and succeed
        assert pf.acquire() is True

        import os

        assert Path(pid_path).read_text().strip() == str(os.getpid())

        pf.release()

    def test_double_acquire_fails_when_running(self, tmp_path):
        """Acquiring twice with our own (running) PID should fail."""
        pid_path = str(tmp_path / "test.pid")

        pf1 = PidFile(pid_path)
        assert pf1.acquire() is True

        pf2 = PidFile(pid_path)
        assert pf2.acquire() is False

        pf1.release()

    def test_is_running_no_file(self, tmp_path):
        pid_path = str(tmp_path / "nonexistent.pid")
        pf = PidFile(pid_path)
        running, pid = pf.is_running()
        assert running is False
        assert pid is None

    def test_is_running_with_active_process(self, tmp_path):
        """is_running returns True for a live PID."""
        import os

        pid_path = str(tmp_path / "test.pid")
        Path(pid_path).write_text(str(os.getpid()))

        pf = PidFile(pid_path)
        running, pid = pf.is_running()
        assert running is True
        assert pid == os.getpid()

    def test_is_running_with_dead_process(self, tmp_path):
        """is_running returns False for a dead PID."""
        pid_path = str(tmp_path / "test.pid")
        Path(pid_path).write_text("4000000")

        pf = PidFile(pid_path)
        running, pid = pf.is_running()
        assert running is False
        assert pid == 4_000_000

    def test_release_idempotent(self, tmp_path):
        """Calling release multiple times should not error."""
        pid_path = str(tmp_path / "test.pid")
        pf = PidFile(pid_path)
        pf.acquire()
        pf.release()
        pf.release()  # should not raise


# ---------------------------------------------------------------------------
# Event emitter tests
# ---------------------------------------------------------------------------


class TestRunnerEventEmitter:
    @pytest.mark.asyncio
    async def test_subscribe_and_emit(self):
        emitter = RunnerEventEmitter()
        received: list[RunnerEvent] = []

        emitter.subscribe(RunnerEventType.TICK_START, received.append)

        event = RunnerEvent(RunnerEventType.TICK_START, tick=1)
        await emitter.emit(event)

        assert len(received) == 1
        assert received[0].event_type == RunnerEventType.TICK_START
        assert received[0].tick == 1

    @pytest.mark.asyncio
    async def test_wildcard_subscriber_receives_all(self):
        emitter = RunnerEventEmitter()
        received: list[RunnerEvent] = []

        emitter.subscribe(None, received.append)

        await emitter.emit(RunnerEvent(RunnerEventType.TICK_START, tick=1))
        await emitter.emit(RunnerEvent(RunnerEventType.TICK_COMPLETE, tick=1))
        await emitter.emit(RunnerEvent(RunnerEventType.RUNTIME_STOPPED, tick=2))

        assert len(received) == 3
        assert {e.event_type for e in received} == {
            RunnerEventType.TICK_START,
            RunnerEventType.TICK_COMPLETE,
            RunnerEventType.RUNTIME_STOPPED,
        }

    @pytest.mark.asyncio
    async def test_filtered_subscriber_ignores_other_types(self):
        emitter = RunnerEventEmitter()
        received: list[RunnerEvent] = []

        emitter.subscribe(RunnerEventType.TICK_FAILED, received.append)

        await emitter.emit(RunnerEvent(RunnerEventType.TICK_START, tick=1))
        await emitter.emit(RunnerEvent(RunnerEventType.TICK_COMPLETE, tick=1))
        await emitter.emit(RunnerEvent(RunnerEventType.TICK_FAILED, tick=2, data={"error": "boom"}))

        assert len(received) == 1
        assert received[0].event_type == RunnerEventType.TICK_FAILED
        assert received[0].data["error"] == "boom"

    @pytest.mark.asyncio
    async def test_bad_callback_does_not_crash_emitter(self):
        emitter = RunnerEventEmitter()
        received: list[RunnerEvent] = []

        def exploding_callback(event: RunnerEvent) -> None:
            raise ValueError("subscriber on fire")

        emitter.subscribe(RunnerEventType.TICK_START, exploding_callback)
        emitter.subscribe(RunnerEventType.TICK_START, received.append)

        # Should not raise despite the exploding callback
        await emitter.emit(RunnerEvent(RunnerEventType.TICK_START, tick=1))

        # The well-behaved subscriber still got the event
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_async_callback(self):
        emitter = RunnerEventEmitter()
        received: list[RunnerEvent] = []

        async def async_cb(event: RunnerEvent) -> None:
            received.append(event)

        emitter.subscribe(RunnerEventType.SESSION_STARTED, async_cb)

        await emitter.emit(
            RunnerEvent(RunnerEventType.SESSION_STARTED, tick=3, data={"session_id": "s-1"})
        )

        assert len(received) == 1
        assert received[0].data["session_id"] == "s-1"

    @pytest.mark.asyncio
    async def test_unsubscribe(self):
        emitter = RunnerEventEmitter()
        received: list[RunnerEvent] = []

        emitter.subscribe(RunnerEventType.TICK_START, received.append)
        await emitter.emit(RunnerEvent(RunnerEventType.TICK_START, tick=1))
        assert len(received) == 1

        emitter.unsubscribe(received.append)
        await emitter.emit(RunnerEvent(RunnerEventType.TICK_START, tick=2))
        # Still 1 — the second event was not delivered
        assert len(received) == 1


# ---------------------------------------------------------------------------
# Trajectory analysis tests
# ---------------------------------------------------------------------------


class TestTrajectory:
    def test_empty_trajectory(self, storage):
        from apollobot.runtime.trajectory import ResearchTrajectory

        trajectory = ResearchTrajectory(storage)
        analysis = trajectory.analyze()
        assert analysis.total_papers == 0
        assert len(analysis.recommendations) > 0

    def test_trajectory_with_data(self, storage):
        from apollobot.runtime.trajectory import ResearchTrajectory

        # Add sessions across two domains
        for i in range(4):
            storage.register_session(
                SessionSummary(
                    session_id=f"s-bio-{i}",
                    objective=f"bio question {i}",
                    domain="bioinformatics",
                    mode="hypothesis",
                    phase="planning",
                    started_at=f"2026-03-0{i + 1}T12:00:00",
                )
            )
            storage.update_session(
                f"s-bio-{i}",
                phase="complete",
                cost_usd=8.0,
                completed_at=f"2026-03-0{i + 1}T14:00:00",
                translation_score=8.5,
            )

        storage.register_session(
            SessionSummary(
                session_id="s-phys-0",
                objective="physics question",
                domain="physics",
                mode="hypothesis",
                phase="planning",
                started_at="2026-03-01T12:00:00",
            )
        )
        storage.update_session(
            "s-phys-0",
            phase="complete",
            cost_usd=15.0,
            completed_at="2026-03-01T14:00:00",
            translation_score=6.0,
        )

        trajectory = ResearchTrajectory(storage)
        analysis = trajectory.analyze()

        assert analysis.total_papers == 5
        assert analysis.total_cost == 47.0
        assert len(analysis.domain_insights) == 2

        # Format for brain should produce text
        brain_text = trajectory.format_for_brain(analysis)
        assert "bioinformatics" in brain_text
        assert "physics" in brain_text

    def test_trajectory_underexplored(self, storage):
        from apollobot.runtime.trajectory import ResearchTrajectory

        storage.register_session(
            SessionSummary(
                session_id="s-1",
                objective="q",
                domain="bioinformatics",
                mode="h",
                phase="planning",
                started_at="2026-03-01T12:00:00",
            )
        )
        storage.update_session(
            "s-1", phase="complete", cost_usd=5.0, completed_at="2026-03-01T14:00:00"
        )

        trajectory = ResearchTrajectory(storage)
        analysis = trajectory.analyze(["bioinformatics", "physics", "astronomy"])

        # physics and astronomy haven't been explored
        assert "physics" in analysis.underexplored_domains
        assert "astronomy" in analysis.underexplored_domains


# ---------------------------------------------------------------------------
# Runtime provenance tests
# ---------------------------------------------------------------------------


class TestRuntimeProvenance:
    def test_log_decision(self, tmp_path):
        from apollobot.runtime.provenance import RuntimeProvenanceLogger

        prov = RuntimeProvenanceLogger(str(tmp_path / "prov"))

        prov.log_decision(
            tick=1, reasoning="exploring", actions=["start_research"], next_check_in=300
        )

        entries = prov.get_recent_entries("decisions")
        assert len(entries) == 1
        assert entries[0]["tick"] == 1
        assert entries[0]["reasoning"] == "exploring"

    def test_log_enforcement(self, tmp_path):
        from apollobot.runtime.provenance import RuntimeProvenanceLogger

        prov = RuntimeProvenanceLogger(str(tmp_path / "prov"))

        prov.log_enforcement(
            tick=2, action_type="start_research", allowed=False, reason="budget exhausted"
        )

        entries = prov.get_recent_entries("enforcements")
        assert len(entries) == 1
        assert entries[0]["allowed"] is False

    def test_log_lifecycle(self, tmp_path):
        from apollobot.runtime.provenance import RuntimeProvenanceLogger

        prov = RuntimeProvenanceLogger(str(tmp_path / "prov"))

        prov.log_lifecycle("runtime_started", {"domain": "bio"})
        prov.log_lifecycle("runtime_stopped", {"reason": "shutdown"})

        entries = prov.get_recent_entries("lifecycle")
        assert len(entries) == 2
        assert entries[0]["event"] == "runtime_started"
        assert entries[1]["event"] == "runtime_stopped"

    def test_empty_log(self, tmp_path):
        from apollobot.runtime.provenance import RuntimeProvenanceLogger

        prov = RuntimeProvenanceLogger(str(tmp_path / "prov"))
        assert prov.get_recent_entries("decisions") == []


# ---------------------------------------------------------------------------
# Guardrails endpoint tests
# ---------------------------------------------------------------------------


class TestGuardrailsEndpoint:
    @pytest.mark.asyncio
    async def test_guardrails_update_via_http(self):
        from aiohttp.test_utils import TestClient, TestServer

        h = HealthServer(port=0)
        h.on_guardrails_update = lambda updates: updates

        async with TestClient(TestServer(h._app)) as client:
            resp = await client.post("/guardrails", json={"daily_compute_budget_usd": 200.0})
            assert resp.status == 200
            data = await resp.json()
            assert data["updated"]["daily_compute_budget_usd"] == 200.0

    @pytest.mark.asyncio
    async def test_guardrails_update_no_handler(self):
        from aiohttp.test_utils import TestClient, TestServer

        h = HealthServer(port=0)
        # No on_guardrails_update set

        async with TestClient(TestServer(h._app)) as client:
            resp = await client.post("/guardrails", json={"daily_compute_budget_usd": 200.0})
            assert resp.status == 501
