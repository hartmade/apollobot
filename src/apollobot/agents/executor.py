"""
Research Executor — runs the research plan phase by phase.

This is the main agent loop.  It takes a ResearchPlan and a Session,
then executes each phase: literature review, data acquisition,
analysis, statistical testing, and manuscript drafting.

The executor is the "autonomous researcher" — it makes decisions,
handles errors, and adapts when things don't go as planned.
"""

from __future__ import annotations

import asyncio
import json
import logging
import traceback
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

from apollobot.agents import LLMProvider, LLMResponse
from apollobot.agents.planner import AnalysisStep, ResearchPlan
from apollobot.core.mission import CheckpointAction, Mission
from apollobot.core.provenance import ProvenanceEngine
from apollobot.core.session import Phase, Session
from apollobot.mcp import MCPClient


class CheckpointHandler:
    """Handles checkpoint approvals — can be overridden for different UIs."""

    async def request_approval(self, phase: str, summary: str) -> bool:
        """Default: auto-approve everything (override for interactive mode)."""
        return True

    async def notify(self, phase: str, summary: str) -> None:
        """Default: no-op (override to send notifications)."""
        pass


class ResearchExecutor:
    """
    Executes a research plan autonomously.

    The executor is the core agent loop.  It:
    1. Reviews literature
    2. Acquires data via MCP servers
    3. Runs analyses
    4. Performs statistical testing
    5. Drafts a manuscript
    6. Self-reviews the output

    At each phase, it can:
    - Adapt the plan based on what it finds
    - Request human approval at checkpoints
    - Handle errors and retry or adjust
    - Track costs and abort if over budget
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

    # ------------------------------------------------------------------
    # Main execution loop
    # ------------------------------------------------------------------

    async def execute(self, session: Session, plan: ResearchPlan) -> Session:
        """
        Execute the full research plan.
        Returns the updated session with all results.
        """
        session.init_directories()

        phases = [
            (Phase.LITERATURE_REVIEW, self._literature_review),
            (Phase.DATA_ACQUISITION, self._acquire_data),
            (Phase.ANALYSIS, self._run_analyses),
            (Phase.STATISTICAL_TESTING, self._statistical_testing),
            (Phase.MANUSCRIPT_DRAFTING, self._draft_manuscript),
            (Phase.SELF_REVIEW, self._self_review),
            (Phase.MANUSCRIPT_REVISION, self._revise_manuscript),
        ]

        for phase, handler in phases:
            # Budget check
            if not session.check_budget():
                session.fail_phase(phase, "Budget exceeded")
                self.provenance.log_event("budget_exceeded", {
                    "spent": session.cost.total_cost,
                    "limit": session.mission.constraints.compute_budget,
                })
                await self.checkpoint.notify(
                    phase.value,
                    f"Budget exceeded: ${session.cost.total_cost:.2f} / "
                    f"${session.mission.constraints.compute_budget:.2f}",
                )
                break

            # Checkpoint handling
            await self._handle_checkpoint(session.mission, phase.value)

            # Notify phase start
            await self.checkpoint.notify(phase.value, f"Starting {phase.value}")

            # Execute phase
            session.begin_phase(phase)
            try:
                summary, findings = await handler(session, plan)
                session.complete_phase(phase, summary=summary, findings=findings)
                # Notify phase completion with summary
                await self.checkpoint.notify(phase.value, summary)
            except Exception as e:
                tb = traceback.format_exc()
                logger.error("Phase %s failed:\n%s", phase.value, tb)
                session.fail_phase(phase, str(e))
                self.provenance.log_event("phase_error", {
                    "phase": phase.value,
                    "error": str(e),
                    "traceback": tb,
                })
                await self.checkpoint.notify(
                    phase.value, f"Phase failed: {e}"
                )
                # Decide whether to continue or abort
                if phase in (Phase.LITERATURE_REVIEW, Phase.DATA_ACQUISITION):
                    # These are critical — can't continue without them
                    break
                # For later phases, try to continue with partial results
                continue

            # Save state after each phase
            session.save_state()
            self.provenance.save()

        if session.current_phase != Phase.FAILED:
            session.current_phase = Phase.COMPLETE
            session.save_state()

        # Generate replication kit
        self.provenance.generate_replication_kit(session.session_dir)
        self.provenance.save()

        return session

    # ------------------------------------------------------------------
    # Phase implementations
    # ------------------------------------------------------------------

    async def _literature_review(
        self, session: Session, plan: ResearchPlan
    ) -> tuple[str, list[dict[str, Any]]]:
        """
        Phase 1: Review existing literature.

        Searches across literature MCP servers, synthesizes findings,
        identifies gaps, and refines the research approach.
        """
        self.provenance.log_event("literature_review_started")

        all_papers: list[dict[str, Any]] = []

        for query in plan.literature_queries:
            # Search across all available literature servers
            for server in self.mcp.get_servers():
                if server.domain == "shared" or server.domain == session.mission.domain:
                    try:
                        results = await self.mcp.query(
                            server.name,
                            "search",
                            {"query": query, "limit": 20},
                        )
                        papers = results.get("papers", results.get("results", []))
                        all_papers.extend(papers)
                        self.provenance.log_event("literature_search", {
                            "server": server.name,
                            "query": query,
                            "results_count": len(papers),
                        })
                    except Exception as e:
                        logger.warning(
                            "Literature search failed: server=%s query=%r error=%s",
                            server.name, query, e,
                        )
                        self.provenance.log_event("literature_search_error", {
                            "server": server.name,
                            "query": query,
                            "error": str(e),
                        })

        # Deduplicate by title/DOI (filter out None/non-dict entries)
        seen = set()
        unique_papers = []
        for p in all_papers:
            if not isinstance(p, dict):
                continue
            key = p.get("doi") or p.get("title", "")
            if key and key not in seen:
                seen.add(key)
                unique_papers.append(p)

        # Warn if no papers found
        if not unique_papers:
            msg = (
                "No papers retrieved from external databases. "
                "Literature synthesis will rely solely on model knowledge."
            )
            logger.warning(msg)
            session.warnings.append(msg)
            self.provenance.log_event("empty_literature_results", {
                "queries": plan.literature_queries,
                "warning": msg,
            })

        # Use LLM to synthesize literature
        synthesis_preamble = ""
        if not unique_papers:
            synthesis_preamble = (
                "WARNING: No papers were retrieved from external databases. "
                "The synthesis below is based solely on model knowledge.\n\n"
            )

        synthesis_resp = await self.llm.complete(
            messages=[{"role": "user", "content": (
                synthesis_preamble
                + f"Research objective: {session.mission.objective}\n\n"
                f"I found {len(unique_papers)} relevant papers. Here are the key ones:\n\n"
                + "\n".join(
                    f"- {p.get('title') or 'Untitled'} ({p.get('year') or 'n.d.'}): "
                    f"{(p.get('abstract') or 'No abstract')[:300]}"
                    for p in unique_papers[:30]
                )
                + "\n\nSynthesize these findings into:\n"
                "1. Current state of knowledge\n"
                "2. Key gaps and contradictions\n"
                "3. How this informs our research approach\n"
                "4. Any adjustments to our hypotheses"
            )}],
            system="You are conducting a literature review. Be thorough and critical.",
        )

        session.literature_corpus = unique_papers[:100]  # Keep top 100

        self.provenance.log_llm_call(
            provider=synthesis_resp.provider,
            model=synthesis_resp.model,
            purpose="literature_synthesis",
            input_tokens=synthesis_resp.input_tokens,
            output_tokens=synthesis_resp.output_tokens,
            cost_usd=synthesis_resp.cost_usd,
            response_summary=synthesis_resp.text[:200],
        )

        session.cost.record_llm_call(
            synthesis_resp.input_tokens,
            synthesis_resp.output_tokens,
            synthesis_resp.cost_usd,
        )

        return (
            f"Reviewed {len(unique_papers)} papers across {len(plan.literature_queries)} queries",
            [{"type": "literature_synthesis", "text": synthesis_resp.text}],
        )

    async def _acquire_data(
        self, session: Session, plan: ResearchPlan
    ) -> tuple[str, list[dict[str, Any]]]:
        """
        Phase 2: Acquire datasets via MCP servers.
        """
        self.provenance.log_event("data_acquisition_started", {
            "requirements": [
                {"description": r.description, "source_type": r.source_type,
                 "server_name": r.server_name, "query_params": r.query_params}
                for r in plan.data_requirements
            ],
        })

        # Build set of registered server names for matching
        registered_servers = {s.name for s in self.mcp.get_servers()}

        acquired = []
        for req in plan.data_requirements:
            # Accept any requirement that references a registered MCP server,
            # regardless of source_type.  The LLM often generates source_type
            # values like "download", "api_call", or "database" instead of the
            # canonical "mcp_server".
            has_server = bool(req.server_name and req.server_name in registered_servers)
            if has_server or (req.source_type == "mcp_server" and req.server_name):
                try:
                    result = await self.mcp.query(
                        req.server_name,
                        "download" if "download" in str(req.query_params) else "query",
                        req.query_params,
                    )
                    dataset_info = {
                        "source": req.server_name,
                        "description": req.description,
                        "query": req.query_params,
                        "status": "acquired",
                        "result_summary": str(result)[:500],
                    }
                    acquired.append(dataset_info)
                    session.datasets.append(dataset_info)

                    # Save raw data
                    data_path = session.session_dir / "data" / "raw" / f"{req.server_name}_{len(acquired)}.json"
                    data_path.write_text(json.dumps(result, indent=2, default=str))

                    self.provenance.log_data_transform(
                        source=req.server_name,
                        operation="acquire",
                        description=req.description,
                        output_data=json.dumps(result, default=str),
                    )

                except Exception as e:
                    self.provenance.log_event("data_acquisition_error", {
                        "source": req.server_name,
                        "error": str(e),
                    })
                    if req.priority == "required" and req.fallback:
                        # Try fallback
                        self.provenance.log_event("trying_fallback", {"fallback": req.fallback})

        if not acquired:
            msg = (
                f"No datasets acquired from {len(plan.data_requirements)} requirements. "
                "Analysis will proceed with limited or no data."
            )
            logger.warning(msg)
            session.warnings.append(msg)
            self.provenance.log_event("empty_data_acquisition", {
                "requirements_count": len(plan.data_requirements),
                "warning": msg,
            })

        return (
            f"Acquired {len(acquired)} datasets from {len(plan.data_requirements)} requirements",
            acquired,
        )

    async def _run_analyses(
        self, session: Session, plan: ResearchPlan
    ) -> tuple[str, list[dict[str, Any]]]:
        """
        Phase 3: Run analysis pipeline.

        For each analysis step in the plan, the agent:
        1. Generates the appropriate code/script
        2. Executes it
        3. Captures results and artifacts
        """
        self.provenance.log_event("analysis_started")

        # Build file content previews for the LLM
        file_previews = []
        raw_dir = session.session_dir / "data" / "raw"
        for f in sorted(raw_dir.glob("*.json"))[:5]:
            try:
                content = f.read_text()[:1500]
                file_previews.append(f"=== {f.name} ===\n{content}")
            except Exception:
                pass
        file_context = "\n\n".join(file_previews) if file_previews else "No data files found."

        results = []
        for step in plan.analysis_steps:
            # Ask LLM to generate analysis code
            code_resp = await self.llm.complete(
                messages=[{"role": "user", "content": (
                    f"Generate Python code for this analysis step:\n\n"
                    f"Name: {step.name}\n"
                    f"Description: {step.description}\n"
                    f"Method: {step.method}\n"
                    f"Parameters: {json.dumps(step.parameters)}\n"
                    f"Expected output: {step.expected_output}\n\n"
                    f"Available data files in {session.session_dir / 'data' / 'raw'}:\n"
                    f"{[str(f.name) for f in (session.session_dir / 'data' / 'raw').glob('*')]}\n\n"
                    f"Data file contents (use these exact structures in your code):\n"
                    f"{file_context}\n\n"
                    "Write clean, documented Python code using standard scientific Python "
                    "(numpy, pandas, scipy, scikit-learn, statsmodels). "
                    "Save results to the session's data/processed directory. "
                    "Save any figures to the session's figures directory. "
                    "Print a JSON summary of results to stdout."
                )}],
                system=(
                    "You are a computational scientist writing analysis code. "
                    "Write clean, correct, documented code. Use appropriate "
                    "statistical methods. Handle edge cases."
                ),
            )

            session.cost.record_llm_call(
                code_resp.input_tokens, code_resp.output_tokens, code_resp.cost_usd
            )

            # Extract code from response
            code = self._extract_code(code_resp.text)

            # Save script — sanitize name (LLM may generate names with / or
            # other path-unsafe characters like "QM/MM Transition State.py")
            safe_name = step.name.replace("/", "_").replace("\\", "_").replace("..", "_")
            script_path = session.session_dir / "analysis" / "scripts" / f"{safe_name}.py"
            script_path.parent.mkdir(parents=True, exist_ok=True)
            script_path.write_text(code)

            self.provenance.log_data_transform(
                source="llm_generated",
                operation=step.method,
                description=step.description,
                script_ref=str(script_path),
                parameters=step.parameters,
            )

            # Execute (in v1, we use subprocess; in production, sandboxed execution)
            try:
                import subprocess
                result = subprocess.run(
                    ["python", str(script_path)],
                    capture_output=True,
                    text=True,
                    timeout=600,  # 10 minute timeout per step
                    cwd=str(session.session_dir),
                )

                step_result = {
                    "step": step.name,
                    "status": "completed" if result.returncode == 0 else "failed",
                    "stdout": result.stdout[:2000],
                    "stderr": result.stderr[:1000] if result.returncode != 0 else "",
                }
                results.append(step_result)

                self.provenance.log_event("analysis_step_completed", {
                    "step": step.name,
                    "returncode": result.returncode,
                })

            except subprocess.TimeoutExpired:
                results.append({"step": step.name, "status": "timeout"})
                self.provenance.log_event("analysis_step_timeout", {"step": step.name})

            except Exception as e:
                results.append({"step": step.name, "status": "error", "error": str(e)})

        completed = sum(1 for r in results if r["status"] == "completed")
        return (
            f"Completed {completed}/{len(plan.analysis_steps)} analysis steps",
            results,
        )

    async def _statistical_testing(
        self, session: Session, plan: ResearchPlan
    ) -> tuple[str, list[dict[str, Any]]]:
        """
        Phase 4: Formal statistical testing.

        Runs the statistical tests specified in the plan,
        with proper correction for multiple comparisons.
        """
        self.provenance.log_event("statistical_testing_started")

        # Gather all analysis results
        analysis_results = session.phase_results.get("analysis", None)
        if not analysis_results:
            return ("No analysis results to test", [])

        # Ask LLM to design and run statistical tests
        stats_resp = await self.llm.complete(
            messages=[{"role": "user", "content": (
                f"Research objective: {session.mission.objective}\n\n"
                f"Hypotheses:\n" + "\n".join(f"- {h}" for h in session.mission.hypotheses) + "\n\n"
                f"Statistical framework: {plan.statistical_framework}\n\n"
                f"Analysis results summary:\n{json.dumps(analysis_results.findings[:5], indent=2, default=str)}\n\n"
                "Generate Python code that:\n"
                "1. Loads the processed data\n"
                "2. Runs appropriate statistical tests for each hypothesis\n"
                "3. Applies multiple comparison correction\n"
                "4. Reports effect sizes with confidence intervals\n"
                "5. Outputs a structured JSON summary of results\n"
                "6. Clearly states whether each hypothesis is supported, rejected, or inconclusive"
            )}],
            system=(
                "You are a biostatistician. Use appropriate tests. "
                "Always report effect sizes. Apply multiple comparison correction. "
                "Be conservative — when in doubt, report inconclusive."
            ),
        )

        session.cost.record_llm_call(
            stats_resp.input_tokens, stats_resp.output_tokens, stats_resp.cost_usd
        )

        code = self._extract_code(stats_resp.text)
        script_path = session.session_dir / "analysis" / "scripts" / "statistical_tests.py"
        script_path.write_text(code)

        # Execute
        import subprocess
        result = subprocess.run(
            ["python", str(script_path)],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(session.session_dir),
        )

        findings = []
        if result.returncode == 0:
            try:
                findings = json.loads(result.stdout)
                if isinstance(findings, dict):
                    findings = [findings]
            except json.JSONDecodeError:
                findings = [{"raw_output": result.stdout[:2000]}]

        return (
            f"Statistical testing {'completed' if result.returncode == 0 else 'failed'}",
            findings,
        )

    async def _draft_manuscript(
        self, session: Session, plan: ResearchPlan
    ) -> tuple[str, list[dict[str, Any]]]:
        """
        Phase 5: Draft the manuscript.

        Uses data inventory as ground truth to produce a complete paper draft.
        Includes anti-hallucination guardrails and evidence-level classification.
        """
        self.provenance.log_event("manuscript_drafting_started")

        # Build data inventory — the ground truth for all claims
        data_inventory = self._build_data_inventory(session)

        # Classify evidence level based on analysis success rate
        analysis_result = session.phase_results.get("analysis")
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

        hedging_map = {
            "moderate": "results suggest / evidence indicates",
            "weak": "preliminary findings hint / limited evidence suggests",
            "theoretical": "we hypothesize / computational analysis proposes",
        }

        system_prompt = (
            "You are writing a scientific paper for peer review.\n\n"
            "STRICT RULES — VIOLATION MEANS REJECTION:\n"
            "1. NEVER invent numbers, statistics, p-values, or measurements not in the data inventory\n"
            "2. If the data inventory shows no results for a claim, write "
            '"No empirical data was obtained for this analysis"\n'
            "3. Tables must ONLY contain values from the data inventory — no fabricated rows\n"
            f"4. Use hedging language proportional to evidence: {evidence_level} evidence → "
            f"{hedging_map[evidence_level]}\n"
            "5. Distinguish computational predictions from experimental results\n"
            "6. All claims must cite specific data from the inventory by filename\n\n"
            f"Evidence level for this manuscript: {evidence_level}"
        )

        # Generate each section
        sections = ["abstract", "introduction", "methods", "results", "discussion", "conclusion"]
        manuscript_parts = {}

        for section in sections:
            resp = await self.llm.complete(
                messages=[{"role": "user", "content": (
                    f"Write the {section.upper()} section of a scientific paper.\n\n"
                    f"Research objective: {session.mission.objective}\n"
                    f"Domain: {session.mission.domain}\n"
                    f"Approach: {plan.approach}\n\n"
                    f"Literature context: {len(session.literature_corpus)} papers reviewed\n"
                    f"Datasets used: {len(session.datasets)}\n\n"
                    f"DATA INVENTORY (ground truth — only report what appears here):\n"
                    f"{data_inventory}\n\n"
                    "Write in clear, precise scientific prose. "
                    "Be specific about methods and results. "
                    "Acknowledge limitations in the discussion. "
                    "If no data supports a claim, state that explicitly rather than inventing data."
                )}],
                system=system_prompt,
            )

            manuscript_parts[section] = resp.text
            session.cost.record_llm_call(
                resp.input_tokens, resp.output_tokens, resp.cost_usd
            )

        # Assemble manuscript
        manuscript = self._assemble_latex(session, plan, manuscript_parts)
        manuscript_path = session.session_dir / "manuscript.tex"
        manuscript_path.write_text(manuscript)

        # Also save as markdown for easy reading
        md_path = session.session_dir / "manuscript.md"
        md_parts = [f"# {session.mission.title}\n"]
        for section, text in manuscript_parts.items():
            md_parts.append(f"## {section.title()}\n\n{text}\n")
        md_path.write_text("\n".join(md_parts))

        return (
            "Manuscript draft completed",
            [{"type": "manuscript", "sections": list(manuscript_parts.keys()),
              "evidence_level": evidence_level}],
        )

    async def _self_review(
        self, session: Session, plan: ResearchPlan
    ) -> tuple[str, list[dict[str, Any]]]:
        """
        Phase 6: Self-review.

        The agent critically reviews its own work, checking for:
        - Statistical validity
        - Methodological soundness
        - Logical consistency
        - Overstatement of findings
        - Missing limitations
        """
        self.provenance.log_event("self_review_started")

        manuscript_path = session.session_dir / "manuscript.md"
        manuscript_text = manuscript_path.read_text() if manuscript_path.exists() else ""

        review_resp = await self.llm.complete(
            messages=[{"role": "user", "content": (
                f"Critically review this research manuscript:\n\n"
                f"{manuscript_text[:8000]}\n\n"
                "Evaluate:\n"
                "1. Statistical validity — are tests appropriate? Are results correctly interpreted?\n"
                "2. Methodological soundness — are methods appropriate for the research question?\n"
                "3. Data quality — are there concerns about the data sources used?\n"
                "4. Logical consistency — do conclusions follow from results?\n"
                "5. Overstatement — are claims proportional to evidence?\n"
                "6. Missing limitations — what should be acknowledged?\n"
                "7. Reproducibility — could someone replicate this?\n"
                "8. Novelty — does this add meaningful knowledge?\n\n"
                "Be ruthlessly honest. This is an internal quality check."
            )}],
            system=(
                "You are a harsh but fair peer reviewer. "
                "Your job is to find problems before external reviewers do. "
                "Be specific about issues and suggest concrete fixes."
            ),
        )

        session.cost.record_llm_call(
            review_resp.input_tokens, review_resp.output_tokens, review_resp.cost_usd
        )

        # Save review
        review_path = session.session_dir / "review" / "self_review.md"
        review_path.write_text(review_resp.text)

        # Statistical audit
        stats_audit = await self._run_statistical_audit(session)
        audit_path = session.session_dir / "review" / "statistical_audit.json"
        audit_path.write_text(json.dumps(stats_audit, indent=2))

        # Translation potential scoring
        translation_scores = await self._assess_translation_potential(session)
        session.translation_scores = translation_scores

        findings = [
            {"type": "review", "text": review_resp.text[:500]},
            {"type": "stats_audit", "data": stats_audit},
            {"type": "translation_scores", "data": translation_scores},
        ]

        if translation_scores.get("average", 0) >= 7.0:
            findings.append({"type": "translation_candidate", "flagged": True})

        return ("Self-review completed", findings)

    async def _revise_manuscript(
        self, session: Session, plan: ResearchPlan
    ) -> tuple[str, list[dict[str, Any]]]:
        """
        Phase 6b: Revise manuscript based on self-review.

        Reads self_review.md, identifies actionable criticisms,
        rewrites the manuscript addressing them without fabricating data.
        """
        self.provenance.log_event("manuscript_revision_started")

        review_path = session.session_dir / "review" / "self_review.md"
        review_text = review_path.read_text() if review_path.exists() else ""

        manuscript_path = session.session_dir / "manuscript.md"
        manuscript_text = manuscript_path.read_text() if manuscript_path.exists() else ""

        data_inventory = self._build_data_inventory(session)

        resp = await self.llm.complete(
            messages=[{"role": "user", "content": (
                "Revise this manuscript to address every criticism from the self-review.\n\n"
                f"SELF-REVIEW:\n{review_text[:6000]}\n\n"
                f"CURRENT MANUSCRIPT:\n{manuscript_text[:12000]}\n\n"
                f"DATA INVENTORY (ground truth):\n{data_inventory[:6000]}\n\n"
                "For each criticism:\n"
                "- If data exists to address it, add the data\n"
                "- If no data exists, remove the unsupported claim and note the limitation\n"
                "- Fix all statistical errors identified\n"
                "- Restate overstatements with appropriate hedging\n"
                "- Add missing limitations\n\n"
                "Output the complete revised manuscript in markdown."
            )}],
            system=(
                "You are revising a scientific manuscript based on peer review feedback. "
                "NEVER fabricate data to address a criticism. If the reviewer says data is missing, "
                "acknowledge the limitation — do not invent data to fill the gap. "
                "Remove claims that cannot be supported by the data inventory."
            ),
        )

        session.cost.record_llm_call(
            resp.input_tokens, resp.output_tokens, resp.cost_usd
        )

        # Overwrite manuscript markdown
        manuscript_path.write_text(resp.text)

        # Regenerate LaTeX from revised markdown sections
        sections = ["abstract", "introduction", "methods", "results", "discussion", "conclusion"]
        manuscript_parts = {}
        current_section = None
        current_lines: list[str] = []

        for line in resp.text.split("\n"):
            lower = line.strip().lower()
            matched_section = None
            for s in sections:
                if lower.startswith(f"## {s}") or lower.startswith(f"# {s}"):
                    matched_section = s
                    break
            if matched_section:
                if current_section:
                    manuscript_parts[current_section] = "\n".join(current_lines)
                current_section = matched_section
                current_lines = []
            elif current_section:
                current_lines.append(line)

        if current_section:
            manuscript_parts[current_section] = "\n".join(current_lines)

        if manuscript_parts:
            latex = self._assemble_latex(session, plan, manuscript_parts)
            tex_path = session.session_dir / "manuscript.tex"
            tex_path.write_text(latex)

        self.provenance.log_event("manuscript_revision_completed")

        return (
            "Manuscript revised based on self-review",
            [{"type": "revision", "changes": "Addressed self-review criticisms"}],
        )

    # ------------------------------------------------------------------
    # Data inventory
    # ------------------------------------------------------------------

    def _build_data_inventory(self, session: Session) -> str:
        """Build a structured inventory of all acquired and processed data."""
        inventory_parts = []

        # Raw data files
        raw_dir = session.session_dir / "data" / "raw"
        for f in sorted(raw_dir.glob("*.json")):
            try:
                data = json.loads(f.read_text())
                if isinstance(data, dict):
                    record_counts = {k: len(v) if isinstance(v, list) else 1 for k, v in data.items()}
                    preview = json.dumps(data, indent=2, default=str)[:2000]
                elif isinstance(data, list):
                    record_counts = {"records": len(data)}
                    preview = json.dumps(data[:3], indent=2, default=str)[:2000]
                else:
                    record_counts = {"value": 1}
                    preview = str(data)[:500]
                inventory_parts.append(
                    f"### {f.name}\nRecord counts: {record_counts}\nPreview:\n{preview}\n"
                )
            except Exception:
                inventory_parts.append(f"### {f.name}\n[Could not read]\n")

        # Processed data files
        proc_dir = session.session_dir / "data" / "processed"
        for f in sorted(proc_dir.glob("*")):
            try:
                content = f.read_text()[:3000]
                inventory_parts.append(f"### {f.name} (processed)\n{content}\n")
            except Exception:
                inventory_parts.append(f"### {f.name} (processed)\n[Could not read]\n")

        # Analysis stdout results
        analysis_result = session.phase_results.get("analysis")
        if analysis_result:
            for finding in analysis_result.findings:
                if finding.get("status") == "completed" and finding.get("stdout"):
                    inventory_parts.append(
                        f"### Analysis: {finding['step']}\n{finding['stdout']}\n"
                    )

        # Statistical testing results
        stats_result = session.phase_results.get("statistical_testing")
        if stats_result and stats_result.findings:
            inventory_parts.append(
                f"### Statistical Tests\n{json.dumps(stats_result.findings, indent=2, default=str)[:3000]}\n"
            )

        if not inventory_parts:
            return "NO DATA AVAILABLE. All sections must state that no empirical data was collected."

        return "\n".join(inventory_parts)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _handle_checkpoint(self, mission: Mission, phase_name: str) -> None:
        """Check if this phase requires approval."""
        for cp in mission.checkpoints:
            if cp.after == phase_name:
                if cp.action == CheckpointAction.REQUIRE_APPROVAL:
                    approved = await self.checkpoint.request_approval(
                        phase_name, f"About to begin {phase_name}"
                    )
                    if not approved:
                        raise RuntimeError(f"Checkpoint rejected at {phase_name}")
                elif cp.action == CheckpointAction.NOTIFY:
                    await self.checkpoint.notify(phase_name, f"Completed {phase_name}")

    def _extract_code(self, text: str) -> str:
        """Extract Python code from LLM response."""
        import re
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        if "```python" in text:
            return text.split("```python")[1].split("```")[0].strip()
        if "```" in text:
            return text.split("```")[1].split("```")[0].strip()
        return text

    def _assemble_latex(
        self,
        session: Session,
        plan: ResearchPlan,
        parts: dict[str, str],
    ) -> str:
        """Assemble a LaTeX document from manuscript parts."""
        # Simple LaTeX template — in production, use Jinja2 templates
        return (
            "\\documentclass[12pt]{article}\n"
            "\\usepackage{amsmath,graphicx,hyperref,natbib}\n"
            "\\usepackage[margin=1in]{geometry}\n\n"
            f"\\title{{{session.mission.title}}}\n"
            f"\\author{{{session.mission.metadata.get('author', 'OCR Agent')}}}\n"
            "\\date{\\today}\n\n"
            "\\begin{document}\n"
            "\\maketitle\n\n"
            "\\begin{abstract}\n"
            f"{parts.get('abstract', '')}\n"
            "\\end{abstract}\n\n"
            "\\section{Introduction}\n"
            f"{parts.get('introduction', '')}\n\n"
            "\\section{Methods}\n"
            f"{parts.get('methods', '')}\n\n"
            "\\section{Results}\n"
            f"{parts.get('results', '')}\n\n"
            "\\section{Discussion}\n"
            f"{parts.get('discussion', '')}\n\n"
            "\\section{Conclusion}\n"
            f"{parts.get('conclusion', '')}\n\n"
            "\\end{document}\n"
        )

    async def _assess_translation_potential(
        self, session: Session
    ) -> dict[str, float]:
        """
        Assess translation potential of research findings.

        Scores commercial relevance, implementation feasibility, and
        novelty on a 0-10 scale. Findings scoring >= 7 average are
        flagged as translation candidates.
        """
        manuscript_path = session.session_dir / "manuscript.md"
        manuscript_text = manuscript_path.read_text()[:4000] if manuscript_path.exists() else ""

        resp = await self.llm.complete(
            messages=[{"role": "user", "content": (
                f"Assess the translation potential of these research findings:\n\n"
                f"Objective: {session.mission.objective}\n"
                f"Key findings: {', '.join(session.key_findings[:5])}\n\n"
                f"Manuscript excerpt:\n{manuscript_text[:3000]}\n\n"
                "Score each dimension 0-10:\n"
                "1. Commercial relevance — market demand, willingness to pay\n"
                "2. Implementation feasibility — can this be built today?\n"
                "3. Novelty — differentiation from existing solutions\n\n"
                'Respond ONLY with JSON: {"commercial_relevance": <0-10>, '
                '"implementation_feasibility": <0-10>, "novelty": <0-10>}'
            )}],
            system=(
                "You are a technology transfer specialist assessing research "
                "findings for commercial translation. Be realistic."
            ),
        )

        session.cost.record_llm_call(
            resp.input_tokens, resp.output_tokens, resp.cost_usd
        )

        try:
            scores = json.loads(self._extract_json(resp.text))
        except (json.JSONDecodeError, ValueError):
            scores = {"commercial_relevance": 5.0, "implementation_feasibility": 5.0, "novelty": 5.0}

        cr = float(scores.get("commercial_relevance", 5.0))
        if_ = float(scores.get("implementation_feasibility", 5.0))
        nv = float(scores.get("novelty", 5.0))

        return {
            "commercial_relevance": cr,
            "implementation_feasibility": if_,
            "novelty": nv,
            "average": (cr + if_ + nv) / 3,
        }

    @staticmethod
    def _extract_json(text: str) -> str:
        """Extract JSON from LLM response."""
        import re
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        if "```json" in text:
            return text.split("```json")[1].split("```")[0].strip()
        if "```" in text:
            return text.split("```")[1].split("```")[0].strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return text[start:end]
        return text

    async def _run_statistical_audit(self, session: Session) -> dict[str, Any]:
        """LLM-based statistical audit of manuscript claims vs actual data."""
        manuscript_path = session.session_dir / "manuscript.md"
        manuscript_text = manuscript_path.read_text()[:6000] if manuscript_path.exists() else ""
        data_inventory = self._build_data_inventory(session)

        resp = await self.llm.complete(
            messages=[{"role": "user", "content": (
                "Audit the statistical claims in this manuscript against the actual data.\n\n"
                f"MANUSCRIPT:\n{manuscript_text}\n\n"
                f"DATA INVENTORY:\n{data_inventory[:4000]}\n\n"
                "Check each of these and respond ONLY with JSON:\n"
                "1. multiple_comparisons — was correction applied when needed?\n"
                "2. effect_sizes — are they reported with CIs?\n"
                "3. sample_sizes — adequate for claims made?\n"
                "4. assumptions — are statistical test assumptions met?\n"
                "5. fabrication_check — do all numbers in manuscript match data inventory?\n\n"
                '{"checks": [{"name": "...", "status": "pass|fail|not_applicable", "note": "..."}], '
                '"overall": "pass|pass_with_notes|fail", "fabrication_detected": true/false}'
            )}],
            system="You are a statistical auditor. Be strict. Flag any number in the manuscript not traceable to the data inventory.",
        )

        session.cost.record_llm_call(
            resp.input_tokens, resp.output_tokens, resp.cost_usd
        )

        try:
            audit = json.loads(self._extract_json(resp.text))
        except (json.JSONDecodeError, ValueError):
            audit = {
                "checks": [{"name": "audit_failed", "status": "fail", "note": "Could not parse audit"}],
                "overall": "fail",
            }

        audit["audit_version"] = "0.2.0"
        return audit
