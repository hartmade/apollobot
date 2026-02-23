"""
Research Translator — converts research findings into implementation specs.

The Translate mode takes a completed Discover session (or external paper)
and produces:
1. Assessment of translation potential
2. Prior art / IP landscape analysis
3. Implementation specification
4. Feasibility validation
5. Translation report

Each phase uses LLM + domain-specific MCP servers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from apollobot.agents import LLMProvider
from apollobot.agents.executor import CheckpointHandler
from apollobot.core.mission import Mission
from apollobot.core.provenance import ProvenanceEngine
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
from apollobot.mcp import MCPClient


class ResearchTranslator:
    """
    Translates research findings into actionable implementation specifications.

    Five phases:
    1. Assess — Evaluate translation potential
    2. Prior Art — IP landscape analysis
    3. Specify — Create implementation specification
    4. Validate — Feasibility assessment
    5. Report — Compile translation report
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

    async def translate(self, session: Session) -> Session:
        """
        Execute the full translation pipeline.

        The session should already have a translation report stub
        with source_session_id or source_paper_doi set.
        """
        report = session.translation_report or TranslationReport()
        report.status = TranslationStatus.IN_PROGRESS

        phases = [
            (Phase.TRANSLATE_ASSESS, self._assess),
            (Phase.TRANSLATE_PRIOR_ART, self._prior_art),
            (Phase.TRANSLATE_SPECIFY, self._specify),
            (Phase.TRANSLATE_VALIDATE, self._validate),
            (Phase.TRANSLATE_REPORT, self._compile_report),
        ]

        for phase, handler in phases:
            if not session.check_budget():
                session.fail_phase(phase, "Budget exceeded")
                report.status = TranslationStatus.FAILED
                break

            await self.checkpoint.notify(phase.value, f"Starting {phase.value}")
            session.begin_phase(phase)

            try:
                summary, findings = await handler(session, report)
                session.complete_phase(phase, summary=summary, findings=findings)
            except Exception as e:
                session.fail_phase(phase, str(e))
                report.status = TranslationStatus.FAILED
                self.provenance.log_event("translate_phase_error", {
                    "phase": phase.value,
                    "error": str(e),
                })
                break

            session.save_state()
            self.provenance.save()

        if report.status != TranslationStatus.FAILED:
            report.status = TranslationStatus.COMPLETED
            session.current_phase = Phase.COMPLETE

        session.translation_report = report
        session.save_state()
        self.provenance.save()

        return session

    # ------------------------------------------------------------------
    # Phase 1: Assess translation potential
    # ------------------------------------------------------------------

    async def _assess(
        self, session: Session, report: TranslationReport
    ) -> tuple[str, list[dict[str, Any]]]:
        self.provenance.log_event("translate_assess_started")

        # Gather source material
        source_text = self._gather_source_material(session)

        resp = await self.llm.complete(
            messages=[{"role": "user", "content": (
                f"Assess the translation potential of these research findings:\n\n"
                f"{source_text[:6000]}\n\n"
                "Evaluate on three dimensions (each 0-10):\n"
                "1. Commercial relevance — Is there market demand? Who would pay?\n"
                "2. Implementation feasibility — Can this be built with current technology?\n"
                "3. Novelty — How differentiated is this from existing solutions?\n\n"
                "Respond in JSON format:\n"
                '{"commercial_relevance": <0-10>, "implementation_feasibility": <0-10>, '
                '"novelty": <0-10>, "summary": "<assessment summary>", '
                '"key_applications": ["<app1>", "<app2>"], '
                '"target_markets": ["<market1>", "<market2>"]}'
            )}],
            system=(
                "You are a technology transfer expert evaluating research for "
                "commercial translation potential. Be realistic and evidence-based."
            ),
        )

        session.cost.record_llm_call(
            resp.input_tokens, resp.output_tokens, resp.cost_usd
        )

        try:
            assessment = json.loads(self._extract_json(resp.text))
        except (json.JSONDecodeError, ValueError):
            assessment = {
                "commercial_relevance": 5,
                "implementation_feasibility": 5,
                "novelty": 5,
                "summary": resp.text[:500],
            }

        report.translation_scores = TranslationScores(
            commercial_relevance=float(assessment.get("commercial_relevance", 5)),
            implementation_feasibility=float(assessment.get("implementation_feasibility", 5)),
            novelty=float(assessment.get("novelty", 5)),
        )
        report.assessment_summary = assessment.get("summary", "")

        self.provenance.log_llm_call(
            provider=resp.provider, model=resp.model,
            purpose="translation_assessment",
            input_tokens=resp.input_tokens, output_tokens=resp.output_tokens,
            cost_usd=resp.cost_usd,
            response_summary=report.assessment_summary[:200],
        )

        return (
            f"Assessment complete: avg score {report.translation_scores.average:.1f}/10",
            [{"type": "assessment", "scores": report.translation_scores.model_dump()}],
        )

    # ------------------------------------------------------------------
    # Phase 2: Prior art / IP landscape
    # ------------------------------------------------------------------

    async def _prior_art(
        self, session: Session, report: TranslationReport
    ) -> tuple[str, list[dict[str, Any]]]:
        self.provenance.log_event("translate_prior_art_started")

        source_text = self._gather_source_material(session)

        resp = await self.llm.complete(
            messages=[{"role": "user", "content": (
                f"Analyze the intellectual property landscape for:\n\n"
                f"{source_text[:4000]}\n\n"
                "Provide:\n"
                "1. Key existing patents in this space\n"
                "2. Freedom to operate assessment (clear/restricted/blocked)\n"
                "3. Patentability of the novel findings\n"
                "4. Recommended IP strategy\n"
                "5. Prior art summary\n\n"
                "Respond in JSON format with fields: freedom_to_operate, "
                "patentability_assessment, recommended_ip_strategy, prior_art_summary, "
                "key_claims_at_risk (list)"
            )}],
            system=(
                "You are an IP analyst specializing in technology transfer. "
                "Assess patent landscapes and freedom to operate."
            ),
        )

        session.cost.record_llm_call(
            resp.input_tokens, resp.output_tokens, resp.cost_usd
        )

        try:
            ip_data = json.loads(self._extract_json(resp.text))
        except (json.JSONDecodeError, ValueError):
            ip_data = {"freedom_to_operate": "unknown", "prior_art_summary": resp.text[:500]}

        report.ip_landscape = IPLandscape(
            freedom_to_operate=ip_data.get("freedom_to_operate", "unknown"),
            patentability_assessment=ip_data.get("patentability_assessment", ""),
            recommended_ip_strategy=ip_data.get("recommended_ip_strategy", ""),
            prior_art_summary=ip_data.get("prior_art_summary", ""),
            key_claims_at_risk=ip_data.get("key_claims_at_risk", []),
        )

        return (
            f"IP landscape analyzed: FTO = {report.ip_landscape.freedom_to_operate}",
            [{"type": "ip_landscape", "fto": report.ip_landscape.freedom_to_operate}],
        )

    # ------------------------------------------------------------------
    # Phase 3: Implementation specification
    # ------------------------------------------------------------------

    async def _specify(
        self, session: Session, report: TranslationReport
    ) -> tuple[str, list[dict[str, Any]]]:
        self.provenance.log_event("translate_specify_started")

        source_text = self._gather_source_material(session)
        assessment = report.assessment_summary

        resp = await self.llm.complete(
            messages=[{"role": "user", "content": (
                f"Create a detailed implementation specification for:\n\n"
                f"Research findings:\n{source_text[:4000]}\n\n"
                f"Translation assessment:\n{assessment}\n\n"
                "Produce:\n"
                "1. Title and description of the implementation\n"
                "2. Target platform (Python library, web API, clinical tool, etc.)\n"
                "3. Architecture overview\n"
                "4. Key components with descriptions\n"
                "5. Data requirements\n"
                "6. Testing strategy\n"
                "7. Deployment strategy\n"
                "8. Estimated cost and timeline\n\n"
                "Respond in JSON format with fields: title, description, "
                "target_platform, architecture_overview, components (list of "
                "{name, description, priority}), data_requirements (list), "
                "testing_strategy, deployment_strategy, estimated_cost (USD number), "
                "estimated_timeline"
            )}],
            system=(
                "You are a technical architect creating implementation "
                "specifications from research findings. Be specific and actionable."
            ),
        )

        session.cost.record_llm_call(
            resp.input_tokens, resp.output_tokens, resp.cost_usd
        )

        try:
            spec_data = json.loads(self._extract_json(resp.text))
        except (json.JSONDecodeError, ValueError):
            spec_data = {"title": "Implementation", "description": resp.text[:500]}

        report.implementation_spec = ImplementationSpec(
            title=spec_data.get("title", ""),
            description=spec_data.get("description", ""),
            target_platform=spec_data.get("target_platform", ""),
            architecture_overview=spec_data.get("architecture_overview", ""),
            components=spec_data.get("components", []),
            data_requirements=spec_data.get("data_requirements", []),
            testing_strategy=spec_data.get("testing_strategy", ""),
            deployment_strategy=spec_data.get("deployment_strategy", ""),
            estimated_cost=float(spec_data.get("estimated_cost", 0)),
            estimated_timeline=spec_data.get("estimated_timeline", ""),
        )

        return (
            f"Spec created: {report.implementation_spec.title}",
            [{"type": "implementation_spec", "title": report.implementation_spec.title}],
        )

    # ------------------------------------------------------------------
    # Phase 4: Feasibility validation
    # ------------------------------------------------------------------

    async def _validate(
        self, session: Session, report: TranslationReport
    ) -> tuple[str, list[dict[str, Any]]]:
        self.provenance.log_event("translate_validate_started")

        spec = report.implementation_spec

        resp = await self.llm.complete(
            messages=[{"role": "user", "content": (
                f"Validate the feasibility of this implementation:\n\n"
                f"Title: {spec.title}\n"
                f"Platform: {spec.target_platform}\n"
                f"Architecture: {spec.architecture_overview}\n"
                f"Components: {json.dumps(spec.components, default=str)[:2000]}\n"
                f"IP Status: FTO = {report.ip_landscape.freedom_to_operate}\n\n"
                "Assess:\n"
                "1. Overall feasibility rating (high/medium/low)\n"
                "2. Technical feasibility score (0-10)\n"
                "3. Resource requirements\n"
                "4. Timeline estimate\n"
                "5. Key risks\n"
                "6. Mitigation strategies\n"
                "7. Infrastructure needs\n\n"
                "Respond in JSON format with fields: overall_rating, "
                "technical_feasibility, resource_requirements, timeline_estimate, "
                "key_risks (list), mitigation_strategies (list), "
                "infrastructure_needs (list)"
            )}],
            system=(
                "You are a technical feasibility analyst. "
                "Be realistic about challenges and risks."
            ),
        )

        session.cost.record_llm_call(
            resp.input_tokens, resp.output_tokens, resp.cost_usd
        )

        try:
            feas_data = json.loads(self._extract_json(resp.text))
        except (json.JSONDecodeError, ValueError):
            feas_data = {"overall_rating": "medium", "technical_feasibility": 5}

        rating_map = {"high": FeasibilityRating.HIGH, "medium": FeasibilityRating.MEDIUM, "low": FeasibilityRating.LOW}
        report.feasibility = FeasibilityAssessment(
            overall_rating=rating_map.get(
                feas_data.get("overall_rating", "medium"), FeasibilityRating.MEDIUM
            ),
            technical_feasibility=float(feas_data.get("technical_feasibility", 5)),
            resource_requirements=feas_data.get("resource_requirements", ""),
            timeline_estimate=feas_data.get("timeline_estimate", ""),
            key_risks=feas_data.get("key_risks", []),
            mitigation_strategies=feas_data.get("mitigation_strategies", []),
            infrastructure_needs=feas_data.get("infrastructure_needs", []),
        )

        return (
            f"Feasibility: {report.feasibility.overall_rating.value}",
            [{"type": "feasibility", "rating": report.feasibility.overall_rating.value}],
        )

    # ------------------------------------------------------------------
    # Phase 5: Compile translation report
    # ------------------------------------------------------------------

    async def _compile_report(
        self, session: Session, report: TranslationReport
    ) -> tuple[str, list[dict[str, Any]]]:
        self.provenance.log_event("translate_report_started")

        # Save report to session directory
        report_path = session.session_dir / "translation_report.json"
        report_path.write_text(report.model_dump_json(indent=2))

        # Generate human-readable summary
        resp = await self.llm.complete(
            messages=[{"role": "user", "content": (
                f"Write an executive summary for this translation report:\n\n"
                f"Scores: commercial={report.translation_scores.commercial_relevance}, "
                f"feasibility={report.translation_scores.implementation_feasibility}, "
                f"novelty={report.translation_scores.novelty}\n"
                f"Assessment: {report.assessment_summary}\n"
                f"IP: FTO={report.ip_landscape.freedom_to_operate}\n"
                f"Spec: {report.implementation_spec.title}\n"
                f"Feasibility: {report.feasibility.overall_rating.value}\n"
                f"Risks: {', '.join(report.feasibility.key_risks[:5])}\n\n"
                "Write a concise 2-3 paragraph executive summary."
            )}],
            system="You are writing an executive summary for a technology transfer report.",
        )

        session.cost.record_llm_call(
            resp.input_tokens, resp.output_tokens, resp.cost_usd
        )

        # Save summary
        summary_path = session.session_dir / "translation_summary.md"
        summary_path.write_text(
            f"# Translation Report: {report.implementation_spec.title}\n\n"
            f"**Source:** {report.source_session_id or report.source_paper_doi}\n"
            f"**Scores:** Commercial {report.translation_scores.commercial_relevance}/10 | "
            f"Feasibility {report.translation_scores.implementation_feasibility}/10 | "
            f"Novelty {report.translation_scores.novelty}/10\n"
            f"**Average:** {report.translation_scores.average:.1f}/10\n"
            f"**FTO:** {report.ip_landscape.freedom_to_operate}\n"
            f"**Feasibility:** {report.feasibility.overall_rating.value}\n\n"
            f"## Executive Summary\n\n{resp.text}\n"
        )

        return (
            "Translation report compiled",
            [{"type": "translation_report", "id": report.id}],
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _gather_source_material(self, session: Session) -> str:
        """Collect the source material for translation."""
        parts = []

        # From manuscript
        manuscript = session.session_dir / "manuscript.md"
        if manuscript.exists():
            parts.append(manuscript.read_text()[:4000])

        # From key findings
        if session.key_findings:
            parts.append("Key findings:\n" + "\n".join(f"- {f}" for f in session.key_findings))

        # From self-review
        review = session.session_dir / "review" / "self_review.md"
        if review.exists():
            parts.append("Self-review:\n" + review.read_text()[:2000])

        if not parts:
            parts.append(f"Research objective: {session.mission.objective}")

        return "\n\n---\n\n".join(parts)

    @staticmethod
    def _extract_json(text: str) -> str:
        """Extract JSON from LLM response."""
        if "```json" in text:
            return text.split("```json")[1].split("```")[0].strip()
        if "```" in text:
            return text.split("```")[1].split("```")[0].strip()
        # Try to find JSON object directly
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return text[start:end]
        return text
