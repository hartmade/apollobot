"""
Commercializer — automated market analysis and go-to-market planning.

The Commercialize mode takes a completed Implementation and produces:
1. Market Analysis — TAM/SAM, competitive landscape, segments
2. IP Strategy — Patent filing recommendations, licensing approach
3. Go-to-Market — Pricing, channels, launch strategy

Uses web data, patent databases, and market data MCP servers.
"""

from __future__ import annotations

import json
from typing import Any

from apollobot.agents import LLMProvider
from apollobot.agents.executor import CheckpointHandler
from apollobot.core.provenance import ProvenanceEngine
from apollobot.core.session import Phase, Session
from apollobot.core.translation import (
    MarketAnalysis,
    MarketSegment,
)
from apollobot.mcp import MCPClient


class CommercializationReport:
    """Wraps the commercialization outputs."""

    def __init__(self) -> None:
        self.market_analysis: MarketAnalysis = MarketAnalysis()
        self.ip_strategy: str = ""
        self.go_to_market: str = ""
        self.revenue_projections: dict[str, Any] = {}
        self.launch_timeline: str = ""
        self.partnerships: list[str] = []
        self.regulatory_considerations: list[str] = []


class Commercializer:
    """
    Produces commercialization analysis from implementation specs.

    Three automated phases:
    1. Market Analysis — TAM/SAM, segments, competition
    2. IP Strategy — Patent, licensing, trade secret recommendations
    3. Go-to-Market — Pricing, channels, launch plan
    """

    def __init__(
        self,
        llm: LLMProvider,
        mcp: MCPClient,
        provenance: ProvenanceEngine,
        checkpoint_handler: CheckpointHandler | None = None,
    ) -> None:
        self.llm = llm
        self.mcp = mcp
        self.provenance = provenance
        self.checkpoint = checkpoint_handler or CheckpointHandler()

    async def commercialize(self, session: Session) -> Session:
        """Execute the full commercialization pipeline."""
        comm_report = CommercializationReport()

        phases = [
            (Phase.COMMERCIALIZE_MARKET, self._market_analysis),
            (Phase.COMMERCIALIZE_IP, self._ip_strategy),
            (Phase.COMMERCIALIZE_GTM, self._go_to_market),
        ]

        for phase, handler in phases:
            if not session.check_budget():
                session.fail_phase(phase, "Budget exceeded")
                break

            await self.checkpoint.notify(phase.value, f"Starting {phase.value}")
            session.begin_phase(phase)

            try:
                summary, findings = await handler(session, comm_report)
                session.complete_phase(phase, summary=summary, findings=findings)
            except Exception as e:
                session.fail_phase(phase, str(e))
                self.provenance.log_event("commercialize_phase_error", {
                    "phase": phase.value, "error": str(e),
                })
                continue

            session.save_state()
            self.provenance.save()

        if session.current_phase != Phase.FAILED:
            session.current_phase = Phase.COMPLETE

        # Save report
        self._save_report(session, comm_report)
        session.save_state()
        self.provenance.save()

        return session

    # ------------------------------------------------------------------
    # Phase 1: Market Analysis
    # ------------------------------------------------------------------

    async def _market_analysis(
        self, session: Session, report: CommercializationReport
    ) -> tuple[str, list[dict[str, Any]]]:
        self.provenance.log_event("commercialize_market_started")

        tr = session.translation_report
        spec_title = tr.implementation_spec.title if tr else session.mission.objective
        spec_desc = tr.implementation_spec.description if tr else ""
        assessment = tr.assessment_summary if tr else ""

        resp = await self.llm.complete(
            messages=[{"role": "user", "content": (
                f"Conduct a market analysis for:\n\n"
                f"Product: {spec_title}\n"
                f"Description: {spec_desc}\n"
                f"Assessment: {assessment}\n\n"
                "Analyze:\n"
                "1. Total addressable market (TAM) and serviceable market (SAM)\n"
                "2. Market segments with size estimates and growth rates\n"
                "3. Key players and competitive landscape\n"
                "4. Entry barriers\n"
                "5. Differentiation opportunities\n"
                "6. Pricing strategy recommendations\n\n"
                "Respond in JSON: {total_addressable_market, serviceable_market, "
                "segments: [{name, size_estimate, growth_rate, key_players, entry_barriers}], "
                "competitive_landscape, differentiation (list), pricing_strategy}"
            )}],
            system=(
                "You are a market analyst specializing in technology products "
                "derived from scientific research. Provide evidence-based market sizing."
            ),
        )

        session.cost.record_llm_call(
            resp.input_tokens, resp.output_tokens, resp.cost_usd
        )

        try:
            data = json.loads(self._extract_json(resp.text))
        except (json.JSONDecodeError, ValueError):
            data = {"total_addressable_market": "Unknown", "competitive_landscape": resp.text[:500]}

        segments = [
            MarketSegment(
                name=s.get("name", ""),
                size_estimate=s.get("size_estimate", ""),
                growth_rate=s.get("growth_rate", ""),
                key_players=s.get("key_players", []),
                entry_barriers=s.get("entry_barriers", []),
            )
            for s in data.get("segments", [])
        ]

        report.market_analysis = MarketAnalysis(
            total_addressable_market=data.get("total_addressable_market", ""),
            serviceable_market=data.get("serviceable_market", ""),
            segments=segments,
            competitive_landscape=data.get("competitive_landscape", ""),
            differentiation=data.get("differentiation", []),
            pricing_strategy=data.get("pricing_strategy", ""),
        )

        return (
            f"Market analysis: TAM = {report.market_analysis.total_addressable_market}",
            [{"type": "market_analysis", "tam": report.market_analysis.total_addressable_market}],
        )

    # ------------------------------------------------------------------
    # Phase 2: IP Strategy
    # ------------------------------------------------------------------

    async def _ip_strategy(
        self, session: Session, report: CommercializationReport
    ) -> tuple[str, list[dict[str, Any]]]:
        self.provenance.log_event("commercialize_ip_started")

        tr = session.translation_report
        ip_landscape = tr.ip_landscape if tr else None

        resp = await self.llm.complete(
            messages=[{"role": "user", "content": (
                f"Develop an IP strategy for:\n\n"
                f"Product: {tr.implementation_spec.title if tr else session.mission.objective}\n"
                f"FTO: {ip_landscape.freedom_to_operate if ip_landscape else 'unknown'}\n"
                f"Prior art: {ip_landscape.prior_art_summary if ip_landscape else 'not analyzed'}\n"
                f"Patentability: {ip_landscape.patentability_assessment if ip_landscape else 'unknown'}\n\n"
                "Recommend:\n"
                "1. Patent filing strategy (what to patent, when, where)\n"
                "2. Trade secret vs. patent decision framework\n"
                "3. Licensing approach (exclusive, non-exclusive, FRAND)\n"
                "4. Defensive publications if needed\n"
                "5. Freedom to operate risk mitigation\n"
                "6. Estimated IP costs and timeline"
            )}],
            system=(
                "You are an IP strategy consultant for technology companies. "
                "Provide actionable IP recommendations."
            ),
        )

        session.cost.record_llm_call(
            resp.input_tokens, resp.output_tokens, resp.cost_usd
        )

        report.ip_strategy = resp.text

        return (
            "IP strategy developed",
            [{"type": "ip_strategy", "summary": resp.text[:200]}],
        )

    # ------------------------------------------------------------------
    # Phase 3: Go-to-Market
    # ------------------------------------------------------------------

    async def _go_to_market(
        self, session: Session, report: CommercializationReport
    ) -> tuple[str, list[dict[str, Any]]]:
        self.provenance.log_event("commercialize_gtm_started")

        tr = session.translation_report
        market = report.market_analysis

        resp = await self.llm.complete(
            messages=[{"role": "user", "content": (
                f"Create a go-to-market plan for:\n\n"
                f"Product: {tr.implementation_spec.title if tr else session.mission.objective}\n"
                f"TAM: {market.total_addressable_market}\n"
                f"Pricing: {market.pricing_strategy}\n"
                f"Segments: {', '.join(s.name for s in market.segments[:5])}\n"
                f"Differentiation: {', '.join(market.differentiation[:5])}\n\n"
                "Plan should include:\n"
                "1. Launch timeline (phases)\n"
                "2. Revenue projections (Year 1-3)\n"
                "3. Sales channels\n"
                "4. Marketing strategy\n"
                "5. Partnership opportunities\n"
                "6. Regulatory considerations\n"
                "7. Key metrics and milestones\n\n"
                "Respond in JSON: {launch_timeline, revenue_projections: "
                "{year_1, year_2, year_3}, channels (list), "
                "partnerships (list), regulatory (list), milestones (list)}"
            )}],
            system=(
                "You are a GTM strategist for deep-tech products. "
                "Create realistic, phased go-to-market plans."
            ),
        )

        session.cost.record_llm_call(
            resp.input_tokens, resp.output_tokens, resp.cost_usd
        )

        try:
            data = json.loads(self._extract_json(resp.text))
        except (json.JSONDecodeError, ValueError):
            data = {"launch_timeline": resp.text[:500]}

        report.go_to_market = resp.text
        report.launch_timeline = data.get("launch_timeline", "")
        report.revenue_projections = data.get("revenue_projections", {})
        report.partnerships = data.get("partnerships", [])
        report.regulatory_considerations = data.get("regulatory", [])

        return (
            "Go-to-market plan created",
            [{"type": "gtm", "partnerships": len(report.partnerships)}],
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _save_report(self, session: Session, report: CommercializationReport) -> None:
        """Save the full commercialization report."""
        report_dir = session.session_dir / "commercialization"
        report_dir.mkdir(parents=True, exist_ok=True)

        (report_dir / "market_analysis.json").write_text(
            report.market_analysis.model_dump_json(indent=2)
        )
        (report_dir / "ip_strategy.md").write_text(
            f"# IP Strategy\n\n{report.ip_strategy}"
        )
        (report_dir / "go_to_market.md").write_text(
            f"# Go-to-Market Plan\n\n{report.go_to_market}"
        )
        if report.revenue_projections:
            (report_dir / "revenue_projections.json").write_text(
                json.dumps(report.revenue_projections, indent=2)
            )

    @staticmethod
    def _extract_json(text: str) -> str:
        if "```json" in text:
            return text.split("```json")[1].split("```")[0].strip()
        if "```" in text:
            return text.split("```")[1].split("```")[0].strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return text[start:end]
        return text
