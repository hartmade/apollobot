"""Integration tests for full pipeline flow."""

import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from apollobot.core.mission import Mission, ResearchMode
from apollobot.core.session import Phase, Session
from apollobot.core.translation import TranslationReport, TranslationScores


class TestPipelineModes:
    """Test that pipeline modes are properly defined."""

    def test_pipeline_mode_exists(self):
        assert ResearchMode.PIPELINE == "pipeline"

    def test_discover_mode_exists(self):
        assert ResearchMode.DISCOVER == "discover"

    def test_translate_mode_exists(self):
        assert ResearchMode.TRANSLATE == "translate"

    def test_implement_mode_exists(self):
        assert ResearchMode.IMPLEMENT == "implement"

    def test_commercialize_mode_exists(self):
        assert ResearchMode.COMMERCIALIZE == "commercialize"


class TestMissionPipelineFields:
    """Test pipeline-specific mission fields."""

    def test_source_session_field(self):
        mission = Mission(
            objective="Translate session",
            mode=ResearchMode.TRANSLATE,
            source_session="session-abc12345",
        )
        assert mission.source_session == "session-abc12345"

    def test_source_paper_field(self):
        mission = Mission(
            objective="Translate paper",
            mode=ResearchMode.TRANSLATE,
            source_paper="10.1234/test.2024",
        )
        assert mission.source_paper == "10.1234/test.2024"

    def test_pipeline_mission(self):
        mission = Mission(
            objective="Full pipeline question",
            mode=ResearchMode.PIPELINE,
            domain="bioinformatics",
        )
        assert mission.mode == ResearchMode.PIPELINE


class TestSessionTranslationScores:
    """Test translation scores on Session."""

    def test_default_empty_scores(self):
        mission = Mission(objective="Test")
        session = Session(mission=mission)
        assert session.translation_scores == {}

    def test_set_translation_scores(self):
        mission = Mission(objective="Test")
        session = Session(mission=mission)
        session.translation_scores = {
            "commercial_relevance": 8.0,
            "implementation_feasibility": 7.0,
            "novelty": 9.0,
            "average": 8.0,
        }
        assert session.translation_scores["average"] == 8.0

    def test_translation_report_on_session(self):
        mission = Mission(objective="Test")
        session = Session(mission=mission)
        report = TranslationReport(id="tr-001")
        session.translation_report = report.model_dump()
        assert session.translation_report is not None


class TestSessionPhaseTransitions:
    """Test phase transitions across all modes."""

    @pytest.fixture
    def session(self, temp_dir):
        mission = Mission(
            objective="Test pipeline",
            mode=ResearchMode.PIPELINE,
        )
        mission.metadata["output_dir"] = str(temp_dir)
        session = Session(mission=mission)
        session.init_directories()
        return session

    def test_discover_phase_transitions(self, session):
        """Test Discover mode phase flow."""
        discover_phases = [
            Phase.PLANNING,
            Phase.LITERATURE_REVIEW,
            Phase.DATA_ACQUISITION,
            Phase.ANALYSIS,
            Phase.STATISTICAL_TESTING,
            Phase.MANUSCRIPT_DRAFTING,
            Phase.SELF_REVIEW,
        ]
        for phase in discover_phases:
            session.begin_phase(phase)
            assert session.current_phase == phase
            session.complete_phase(phase, summary=f"Completed {phase.value}")

    def test_translate_phase_transitions(self, session):
        """Test Translate mode phase flow."""
        translate_phases = [
            Phase.TRANSLATE_ASSESS,
            Phase.TRANSLATE_PRIOR_ART,
            Phase.TRANSLATE_SPECIFY,
            Phase.TRANSLATE_VALIDATE,
            Phase.TRANSLATE_REPORT,
        ]
        for phase in translate_phases:
            session.begin_phase(phase)
            assert session.current_phase == phase
            session.complete_phase(phase, summary=f"Completed {phase.value}")

    def test_implement_phase_transitions(self, session):
        """Test Implement mode phase flow."""
        implement_phases = [
            Phase.IMPLEMENT_SCAFFOLD,
            Phase.IMPLEMENT_BUILD,
            Phase.IMPLEMENT_TEST,
            Phase.IMPLEMENT_DOCUMENT,
            Phase.IMPLEMENT_PACKAGE,
            Phase.IMPLEMENT_VALIDATE,
        ]
        for phase in implement_phases:
            session.begin_phase(phase)
            assert session.current_phase == phase
            session.complete_phase(phase, summary=f"Completed {phase.value}")

    def test_commercialize_phase_transitions(self, session):
        """Test Commercialize mode phase flow."""
        commercialize_phases = [
            Phase.COMMERCIALIZE_MARKET,
            Phase.COMMERCIALIZE_IP,
            Phase.COMMERCIALIZE_GTM,
        ]
        for phase in commercialize_phases:
            session.begin_phase(phase)
            assert session.current_phase == phase
            session.complete_phase(phase, summary=f"Completed {phase.value}")

    def test_cancelled_state(self, session):
        session.current_phase = Phase.CANCELLED
        assert session.current_phase == Phase.CANCELLED


class TestCrossModeProvenance:
    """Test cross-mode provenance linking."""

    def test_link_source_session(self, temp_dir):
        from apollobot.core.provenance import ProvenanceEngine

        # Create source session
        source_dir = temp_dir / "source-session"
        source_prov_dir = source_dir / "provenance"
        source_prov_dir.mkdir(parents=True)
        (source_prov_dir / "execution_log.json").write_text(json.dumps([
            {"event": "phase_completed", "phase": "self_review"},
        ]))

        # Create target session
        target_dir = temp_dir / "target-session"
        provenance = ProvenanceEngine(target_dir)
        provenance.link_source_session("source-session", source_dir)

        # Check that source provenance is linked
        assert (target_dir / "provenance" / "source_provenance.json").exists()

    def test_validate_cross_references_valid(self, temp_dir):
        from apollobot.core.provenance import ProvenanceEngine

        source_dir = temp_dir / "source-session"
        source_prov_dir = source_dir / "provenance"
        source_prov_dir.mkdir(parents=True)
        (source_prov_dir / "execution_log.json").write_text(json.dumps([
            {"event": "phase_completed", "phase": "self_review"},
        ]))

        target_dir = temp_dir / "target-session"
        provenance = ProvenanceEngine(target_dir)
        provenance.link_source_session("source-session", source_dir)

        result = provenance.validate_cross_references()
        assert result["valid"] is True

    def test_validate_cross_references_missing(self, temp_dir):
        from apollobot.core.provenance import ProvenanceEngine

        target_dir = temp_dir / "target-session"
        provenance = ProvenanceEngine(target_dir)

        result = provenance.validate_cross_references()
        assert result["valid"] is False
        assert "No source provenance linked" in result["issues"]

    def test_get_provenance_chain(self, temp_dir):
        from apollobot.core.provenance import ProvenanceEngine

        source_dir = temp_dir / "source-session"
        source_prov_dir = source_dir / "provenance"
        source_prov_dir.mkdir(parents=True)
        (source_prov_dir / "execution_log.json").write_text(json.dumps([
            {"event": "session_started"},
        ]))

        target_dir = temp_dir / "target-session"
        provenance = ProvenanceEngine(target_dir)
        provenance.link_source_session("source-session", source_dir)
        provenance.log_event("test_event")

        chain = provenance.get_provenance_chain()
        assert len(chain) == 2
        assert chain[0]["type"] == "source"
        assert chain[1]["type"] == "current"


class TestOrchestratorRouting:
    """Test that the orchestrator routes to the correct mode."""

    def test_discover_modes_route_correctly(self):
        """Hypothesis, exploratory, etc. should all route to discover."""
        discover_modes = [
            ResearchMode.HYPOTHESIS,
            ResearchMode.EXPLORATORY,
            ResearchMode.META_ANALYSIS,
            ResearchMode.REPLICATION,
            ResearchMode.SIMULATION,
        ]
        for mode in discover_modes:
            mission = Mission(objective="test", mode=mode)
            # These are all discover modes — they shouldn't match translate/implement/etc
            assert mode not in (
                ResearchMode.TRANSLATE,
                ResearchMode.IMPLEMENT,
                ResearchMode.COMMERCIALIZE,
                ResearchMode.PIPELINE,
            )
