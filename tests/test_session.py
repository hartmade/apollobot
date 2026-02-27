"""Tests for apollobot.core.session module."""

import pytest
import json
from pathlib import Path

from apollobot.core.mission import Mission, ResearchMode
from apollobot.core.session import Session, Phase, PhaseResult, CostTracker


class TestPhase:
    """Tests for Phase enum."""

    def test_all_phases_exist(self):
        """Verify all research phases are defined."""
        phases = [
            Phase.PLANNING,
            Phase.LITERATURE_REVIEW,
            Phase.DATA_ACQUISITION,
            Phase.ANALYSIS,
            Phase.STATISTICAL_TESTING,
            Phase.MANUSCRIPT_DRAFTING,
            Phase.SELF_REVIEW,
            Phase.COMPLETE,
            Phase.FAILED,
        ]
        assert len(phases) == 9

    def test_phase_values(self):
        """Verify phase string values."""
        assert Phase.PLANNING.value == "planning"
        assert Phase.COMPLETE.value == "complete"
        assert Phase.FAILED.value == "failed"


class TestCostTracker:
    """Tests for CostTracker model."""

    def test_initial_cost_is_zero(self):
        """Verify cost tracker starts at zero."""
        tracker = CostTracker()
        assert tracker.llm_calls == 0
        assert tracker.total_cost == 0.0

    def test_record_llm_call(self):
        """Test recording an LLM call."""
        tracker = CostTracker()
        tracker.record_llm_call(
            input_tokens=1000,
            output_tokens=500,
            cost=0.05,
        )
        assert tracker.llm_calls == 1
        assert tracker.llm_input_tokens == 1000
        assert tracker.llm_output_tokens == 500
        assert tracker.estimated_cost_usd == 0.05

    def test_multiple_llm_calls_accumulate(self):
        """Test that multiple calls accumulate."""
        tracker = CostTracker()
        tracker.record_llm_call(1000, 500, 0.05)
        tracker.record_llm_call(2000, 1000, 0.10)
        assert tracker.llm_calls == 2
        assert tracker.llm_input_tokens == 3000
        assert tracker.llm_output_tokens == 1500
        assert tracker.estimated_cost_usd == pytest.approx(0.15)

    def test_total_cost_includes_compute(self):
        """Test total cost includes compute costs."""
        tracker = CostTracker()
        tracker.record_llm_call(1000, 500, 0.05)
        tracker.compute_cost_usd = 1.00
        assert tracker.total_cost == 1.05


class TestSession:
    """Tests for Session model."""

    def test_create_session(self):
        """Test creating a session from a mission."""
        mission = Mission.from_objective("Test objective")
        session = Session(mission=mission)

        assert session.mission == mission
        assert session.current_phase == Phase.PLANNING
        assert session.cost.total_cost == 0.0

    def test_session_directory(self, temp_dir):
        """Test session directory path."""
        mission = Mission(objective="Test", metadata={"output_dir": str(temp_dir)})
        session = Session(mission=mission)

        expected = temp_dir / mission.id
        assert session.session_dir == expected

    def test_init_directories(self, temp_dir):
        """Test session directory creation."""
        mission = Mission(objective="Test", metadata={"output_dir": str(temp_dir)})
        session = Session(mission=mission)
        session.init_directories()

        assert session.session_dir.exists()
        assert (session.session_dir / "figures").exists()
        assert (session.session_dir / "data" / "raw").exists()
        assert (session.session_dir / "data" / "processed").exists()
        assert (session.session_dir / "analysis" / "scripts").exists()
        assert (session.session_dir / "provenance").exists()
        assert (session.session_dir / "mission.yaml").exists()

    def test_phase_transitions(self):
        """Test beginning and completing phases."""
        mission = Mission.from_objective("Test")
        session = Session(mission=mission)

        session.begin_phase(Phase.LITERATURE_REVIEW)
        assert session.current_phase == Phase.LITERATURE_REVIEW
        assert Phase.LITERATURE_REVIEW.value in session.phase_results

        session.complete_phase(Phase.LITERATURE_REVIEW, summary="Found 50 papers")
        result = session.phase_results[Phase.LITERATURE_REVIEW.value]
        assert result.summary == "Found 50 papers"
        assert result.completed_at != ""

    def test_fail_phase(self):
        """Test failing a phase."""
        mission = Mission.from_objective("Test")
        session = Session(mission=mission)

        session.begin_phase(Phase.DATA_ACQUISITION)
        session.fail_phase(Phase.DATA_ACQUISITION, "API timeout")

        assert session.current_phase == Phase.FAILED
        result = session.phase_results[Phase.DATA_ACQUISITION.value]
        assert "API timeout" in result.errors

    def test_budget_check(self):
        """Test budget checking."""
        mission = Mission(objective="Test")
        mission.constraints.compute_budget = 10.0
        session = Session(mission=mission)

        assert session.check_budget() is True

        session.cost.estimated_cost_usd = 15.0
        assert session.check_budget() is False

    def test_provenance_logging(self):
        """Test event logging for provenance."""
        mission = Mission.from_objective("Test")
        session = Session(mission=mission)

        session.log_event("test_event", {"key": "value"})

        assert len(session.provenance_log) == 1
        assert session.provenance_log[0]["event"] == "test_event"
        assert session.provenance_log[0]["key"] == "value"

    def test_warnings_default_empty(self):
        """Test that warnings list starts empty."""
        mission = Mission.from_objective("Test")
        session = Session(mission=mission)
        assert session.warnings == []

    def test_warnings_append(self):
        """Test appending warnings."""
        mission = Mission.from_objective("Test")
        session = Session(mission=mission)
        session.warnings.append("No papers found")
        session.warnings.append("No datasets acquired")
        assert len(session.warnings) == 2
        assert "No papers found" in session.warnings

    def test_warnings_roundtrip(self, temp_dir):
        """Test warnings survive save/load cycle."""
        mission = Mission(objective="Test", metadata={"output_dir": str(temp_dir)})
        session = Session(mission=mission)
        session.init_directories()
        session.warnings.append("Test warning")
        session.save_state()

        loaded = Session.load_state(session.session_dir)
        assert loaded.warnings == ["Test warning"]

    def test_session_state_roundtrip(self, temp_dir):
        """Test saving and loading session state."""
        mission = Mission(objective="Test", metadata={"output_dir": str(temp_dir)})
        session = Session(mission=mission)
        session.init_directories()

        session.begin_phase(Phase.LITERATURE_REVIEW)
        session.cost.record_llm_call(1000, 500, 0.05)
        session.key_findings.append("Important finding")
        session.save_state()

        loaded = Session.load_state(session.session_dir)
        assert loaded.mission.objective == "Test"
        assert loaded.current_phase == Phase.LITERATURE_REVIEW
        assert loaded.cost.llm_calls == 1
        assert "Important finding" in loaded.key_findings
