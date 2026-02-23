"""Unit tests for Translate mode."""

import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from apollobot.core.mission import Mission, ResearchMode
from apollobot.core.session import Phase, Session
from apollobot.core.translation import (
    FeasibilityAssessment,
    FeasibilityRating,
    IPLandscape,
    ImplementationSpec,
    MarketAnalysis,
    TranslationReport,
    TranslationScores,
    TranslationStatus,
)


# ---------------------------------------------------------------------------
# Translation model tests
# ---------------------------------------------------------------------------


class TestTranslationScores:
    def test_average_calculation(self):
        scores = TranslationScores(
            commercial_relevance=8.0,
            implementation_feasibility=7.0,
            novelty=9.0,
        )
        assert scores.average == 8.0

    def test_translation_candidate_above_threshold(self):
        scores = TranslationScores(
            commercial_relevance=8.0,
            implementation_feasibility=7.0,
            novelty=9.0,
        )
        assert scores.is_translation_candidate is True

    def test_translation_candidate_below_threshold(self):
        scores = TranslationScores(
            commercial_relevance=5.0,
            implementation_feasibility=4.0,
            novelty=6.0,
        )
        assert scores.is_translation_candidate is False

    def test_translation_candidate_at_threshold(self):
        scores = TranslationScores(
            commercial_relevance=7.0,
            implementation_feasibility=7.0,
            novelty=7.0,
        )
        assert scores.is_translation_candidate is True

    def test_default_scores(self):
        scores = TranslationScores()
        assert scores.commercial_relevance == 0.0
        assert scores.implementation_feasibility == 0.0
        assert scores.novelty == 0.0
        assert scores.average == 0.0


class TestTranslationReport:
    def test_default_report(self):
        report = TranslationReport()
        assert report.status == TranslationStatus.PENDING
        assert report.source_session_id == ""
        assert report.source_paper_doi == ""

    def test_report_with_source(self):
        report = TranslationReport(
            id="tr-001",
            source_session_id="session-abc12345",
        )
        assert report.id == "tr-001"
        assert report.source_session_id == "session-abc12345"

    def test_report_serialization(self):
        report = TranslationReport(
            id="tr-001",
            translation_scores=TranslationScores(
                commercial_relevance=8.0,
                implementation_feasibility=7.0,
                novelty=9.0,
            ),
        )
        data = report.model_dump()
        assert data["translation_scores"]["commercial_relevance"] == 8.0

    def test_report_json_roundtrip(self):
        report = TranslationReport(
            id="tr-002",
            status=TranslationStatus.COMPLETED,
            ip_landscape=IPLandscape(freedom_to_operate="clear"),
        )
        json_str = report.model_dump_json()
        restored = TranslationReport.model_validate_json(json_str)
        assert restored.id == "tr-002"
        assert restored.ip_landscape.freedom_to_operate == "clear"


class TestIPLandscape:
    def test_default_landscape(self):
        ip = IPLandscape()
        assert ip.freedom_to_operate == ""
        assert ip.existing_patents == []

    def test_landscape_with_data(self):
        ip = IPLandscape(
            freedom_to_operate="restricted",
            patentability_assessment="Novel combination approach",
            key_claims_at_risk=["claim1", "claim2"],
        )
        assert ip.freedom_to_operate == "restricted"
        assert len(ip.key_claims_at_risk) == 2


class TestImplementationSpec:
    def test_default_spec(self):
        spec = ImplementationSpec()
        assert spec.title == ""
        assert spec.components == []

    def test_spec_with_data(self):
        spec = ImplementationSpec(
            title="Biomarker Detection API",
            target_platform="Python library",
            estimated_cost=50000.0,
        )
        assert spec.title == "Biomarker Detection API"
        assert spec.estimated_cost == 50000.0


class TestFeasibilityAssessment:
    def test_default_assessment(self):
        fa = FeasibilityAssessment()
        assert fa.overall_rating == FeasibilityRating.UNKNOWN

    def test_assessment_with_data(self):
        fa = FeasibilityAssessment(
            overall_rating=FeasibilityRating.HIGH,
            technical_feasibility=8.5,
            key_risks=["Data availability", "Regulatory approval"],
        )
        assert fa.overall_rating == FeasibilityRating.HIGH
        assert fa.technical_feasibility == 8.5
        assert len(fa.key_risks) == 2


# ---------------------------------------------------------------------------
# Translator agent tests
# ---------------------------------------------------------------------------


class TestResearchTranslator:
    @pytest.fixture
    def mock_llm(self):
        llm = MagicMock()
        response = MagicMock()
        response.text = json.dumps({
            "commercial_relevance": 8,
            "implementation_feasibility": 7,
            "novelty": 9,
            "summary": "High translation potential",
        })
        response.provider = "anthropic"
        response.model = "claude-sonnet"
        response.input_tokens = 100
        response.output_tokens = 50
        response.cost_usd = 0.01
        llm.complete = AsyncMock(return_value=response)
        return llm

    @pytest.fixture
    def mock_mcp(self):
        return MagicMock()

    @pytest.fixture
    def translator(self, mock_llm, mock_mcp, temp_dir):
        from apollobot.core.provenance import ProvenanceEngine
        from apollobot.agents.translator import ResearchTranslator

        provenance = ProvenanceEngine(temp_dir)
        return ResearchTranslator(
            llm=mock_llm,
            mcp=mock_mcp,
            provenance=provenance,
        )

    @pytest.fixture
    def session(self, temp_dir):
        mission = Mission(
            objective="Test translation",
            mode=ResearchMode.TRANSLATE,
            source_session="session-abc",
        )
        mission.metadata["output_dir"] = str(temp_dir)
        session = Session(mission=mission)
        session.init_directories()
        session.key_findings = ["Finding 1: significant result"]

        # Create manuscript for source material
        (session.session_dir / "manuscript.md").write_text("# Test Paper\n\nResults here.")

        # Initialize translation report
        report = TranslationReport(id="tr-test", source_session_id="session-abc")
        session.translation_report = report.model_dump()

        return session

    @pytest.mark.asyncio
    async def test_assess_phase(self, translator, session, mock_llm):
        """Test that the assess phase produces translation scores."""
        report = TranslationReport.model_validate(session.translation_report)
        summary, findings = await translator._assess(session, report)

        assert "Assessment complete" in summary
        assert report.translation_scores.commercial_relevance == 8.0

    @pytest.mark.asyncio
    async def test_prior_art_phase(self, translator, session, mock_llm):
        """Test that prior art phase analyzes IP landscape."""
        mock_llm.complete.return_value.text = json.dumps({
            "freedom_to_operate": "clear",
            "patentability_assessment": "Patentable",
            "prior_art_summary": "No blocking prior art",
        })
        report = TranslationReport.model_validate(session.translation_report)
        summary, findings = await translator._prior_art(session, report)

        assert "IP landscape" in summary
        assert report.ip_landscape.freedom_to_operate == "clear"

    @pytest.mark.asyncio
    async def test_specify_phase(self, translator, session, mock_llm):
        """Test that specify phase creates implementation spec."""
        mock_llm.complete.return_value.text = json.dumps({
            "title": "Test Implementation",
            "target_platform": "Python library",
            "architecture_overview": "Microservices",
        })
        report = TranslationReport.model_validate(session.translation_report)
        summary, findings = await translator._specify(session, report)

        assert report.implementation_spec.title == "Test Implementation"

    @pytest.mark.asyncio
    async def test_validate_phase(self, translator, session, mock_llm):
        """Test that validate phase assesses feasibility."""
        mock_llm.complete.return_value.text = json.dumps({
            "overall_rating": "high",
            "technical_feasibility": 8,
            "key_risks": ["Data availability"],
        })
        report = TranslationReport.model_validate(session.translation_report)
        summary, findings = await translator._validate(session, report)

        assert report.feasibility.overall_rating == FeasibilityRating.HIGH
