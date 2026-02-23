"""Unit tests for Commercialize mode."""

import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from apollobot.core.mission import Mission, ResearchMode
from apollobot.core.session import Phase, Session
from apollobot.core.translation import (
    IPLandscape,
    ImplementationSpec,
    MarketAnalysis,
    TranslationReport,
)


class TestCommercializer:
    @pytest.fixture
    def mock_llm(self):
        llm = MagicMock()
        response = MagicMock()
        response.text = json.dumps({
            "total_addressable_market": "$5B",
            "serviceable_market": "$500M",
            "segments": [
                {
                    "name": "Pharma R&D",
                    "size_estimate": "$2B",
                    "growth_rate": "12%",
                    "key_players": ["Pfizer", "Roche"],
                    "entry_barriers": ["Regulatory"],
                },
            ],
            "competitive_landscape": "Fragmented market",
            "differentiation": ["AI-powered", "Provenance-based"],
            "pricing_strategy": "SaaS subscription",
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
    def commercializer(self, mock_llm, mock_mcp, temp_dir):
        from apollobot.core.provenance import ProvenanceEngine
        from apollobot.agents.commercializer import Commercializer

        provenance = ProvenanceEngine(temp_dir)
        return Commercializer(
            llm=mock_llm,
            mcp=mock_mcp,
            provenance=provenance,
        )

    @pytest.fixture
    def session(self, temp_dir):
        mission = Mission(
            objective="Test commercialization",
            mode=ResearchMode.COMMERCIALIZE,
            source_session="session-impl-001",
        )
        mission.metadata["output_dir"] = str(temp_dir)
        session = Session(mission=mission)
        session.init_directories()
        session.key_findings = ["Finding 1"]

        report = TranslationReport(
            id="tr-test",
            implementation_spec=ImplementationSpec(
                title="Biomarker Detection API",
                description="AI-powered biomarker detection",
                target_platform="SaaS API",
            ),
            ip_landscape=IPLandscape(
                freedom_to_operate="clear",
                prior_art_summary="No blocking patents",
            ),
        )
        session.translation_report = report.model_dump()

        return session

    @pytest.mark.asyncio
    async def test_market_analysis_phase(self, commercializer, session, mock_llm):
        """Test that market analysis produces segments."""
        from apollobot.agents.commercializer import CommercializationReport

        report = CommercializationReport()
        session.translation_report = TranslationReport.model_validate(session.translation_report)

        summary, findings = await commercializer._market_analysis(session, report)

        assert "Market analysis" in summary
        assert report.market_analysis.total_addressable_market == "$5B"
        assert len(report.market_analysis.segments) == 1
        assert report.market_analysis.segments[0].name == "Pharma R&D"

    @pytest.mark.asyncio
    async def test_ip_strategy_phase(self, commercializer, session, mock_llm):
        """Test that IP strategy produces recommendations."""
        from apollobot.agents.commercializer import CommercializationReport

        mock_llm.complete.return_value.text = "File provisional patents first."

        report = CommercializationReport()
        session.translation_report = TranslationReport.model_validate(session.translation_report)

        summary, findings = await commercializer._ip_strategy(session, report)

        assert "IP strategy" in summary
        assert report.ip_strategy == "File provisional patents first."

    @pytest.mark.asyncio
    async def test_gtm_phase(self, commercializer, session, mock_llm):
        """Test that GTM phase creates launch plan."""
        from apollobot.agents.commercializer import CommercializationReport

        mock_llm.complete.return_value.text = json.dumps({
            "launch_timeline": "Q1 2025 beta, Q3 2025 GA",
            "revenue_projections": {"year_1": "$100K", "year_2": "$500K"},
            "channels": ["Direct sales", "Partner channel"],
            "partnerships": ["Academic institutions"],
            "regulatory": ["FDA clearance needed"],
            "milestones": ["Beta launch", "First enterprise customer"],
        })

        report = CommercializationReport()
        report.market_analysis = MarketAnalysis(
            total_addressable_market="$5B",
            pricing_strategy="SaaS",
            segments=[],
        )
        session.translation_report = TranslationReport.model_validate(session.translation_report)

        summary, findings = await commercializer._go_to_market(session, report)

        assert "Go-to-market" in summary
        assert len(report.partnerships) == 1

    @pytest.mark.asyncio
    async def test_save_report(self, commercializer, session):
        """Test that report is saved to disk."""
        from apollobot.agents.commercializer import CommercializationReport

        report = CommercializationReport()
        report.market_analysis = MarketAnalysis(total_addressable_market="$5B")
        report.ip_strategy = "File patents"
        report.go_to_market = "Launch plan"

        commercializer._save_report(session, report)

        comm_dir = session.session_dir / "commercialization"
        assert (comm_dir / "market_analysis.json").exists()
        assert (comm_dir / "ip_strategy.md").exists()
        assert (comm_dir / "go_to_market.md").exists()


class TestCommercializePhases:
    """Test phase enum values for commercialize mode."""

    def test_commercialize_phases_exist(self):
        assert Phase.COMMERCIALIZE_MARKET == "commercialize_market"
        assert Phase.COMMERCIALIZE_IP == "commercialize_ip"
        assert Phase.COMMERCIALIZE_GTM == "commercialize_gtm"
