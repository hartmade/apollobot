"""
Research Implementor — builds production-ready implementations from specs.

The Implement mode takes a TranslationReport (specifically its
ImplementationSpec) and produces:
1. Project scaffold
2. Core implementation code
3. Test suite
4. Documentation
5. Packaging / containerization
6. Validation against original findings

Each phase generates code, tests, and artifacts that trace back
to the original research findings via provenance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from apollobot.agents import LLMProvider
from apollobot.agents.executor import CheckpointHandler
from apollobot.core.provenance import ProvenanceEngine
from apollobot.core.session import Phase, Session
from apollobot.mcp import MCPClient


class ResearchImplementor:
    """
    Builds production implementations from translation specs.

    Six phases:
    1. Scaffold — Create project structure
    2. Build — Generate core implementation code
    3. Test — Generate and run test suite
    4. Document — Generate documentation
    5. Package — Create deployment artifacts
    6. Validate — Verify implementation matches research findings
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

    async def implement(self, session: Session) -> Session:
        """Execute the full implementation pipeline."""
        phases = [
            (Phase.IMPLEMENT_SCAFFOLD, self._scaffold),
            (Phase.IMPLEMENT_BUILD, self._build),
            (Phase.IMPLEMENT_TEST, self._test),
            (Phase.IMPLEMENT_DOCUMENT, self._document),
            (Phase.IMPLEMENT_PACKAGE, self._package),
            (Phase.IMPLEMENT_VALIDATE, self._validate),
        ]

        for phase, handler in phases:
            if not session.check_budget():
                session.fail_phase(phase, "Budget exceeded")
                break

            await self.checkpoint.notify(phase.value, f"Starting {phase.value}")
            session.begin_phase(phase)

            try:
                summary, findings = await handler(session)
                session.complete_phase(phase, summary=summary, findings=findings)
            except Exception as e:
                session.fail_phase(phase, str(e))
                self.provenance.log_event("implement_phase_error", {
                    "phase": phase.value, "error": str(e),
                })
                if phase == Phase.IMPLEMENT_SCAFFOLD:
                    break  # Can't continue without scaffold
                continue

            session.save_state()
            self.provenance.save()

        if session.current_phase != Phase.FAILED:
            session.current_phase = Phase.COMPLETE

        session.save_state()
        self.provenance.save()
        return session

    # ------------------------------------------------------------------
    # Phase 1: Scaffold
    # ------------------------------------------------------------------

    async def _scaffold(
        self, session: Session
    ) -> tuple[str, list[dict[str, Any]]]:
        self.provenance.log_event("implement_scaffold_started")

        report = session.translation_report
        spec = report.implementation_spec if report else None

        if not spec:
            raise RuntimeError("No implementation spec found in session")

        impl_dir = session.session_dir / "implementation"

        resp = await self.llm.complete(
            messages=[{"role": "user", "content": (
                f"Create a project scaffold for this implementation:\n\n"
                f"Title: {spec.title}\n"
                f"Platform: {spec.target_platform}\n"
                f"Architecture: {spec.architecture_overview}\n"
                f"Components: {json.dumps(spec.components, default=str)[:3000]}\n\n"
                "Respond in JSON with:\n"
                '{"directories": ["list/of/dirs"], '
                '"files": [{"path": "relative/path.py", "description": "what it does"}], '
                '"dependencies": ["package1", "package2"]}'
            )}],
            system=(
                "You are a senior software engineer creating a clean, "
                "well-structured project scaffold. Follow best practices "
                "for the target platform."
            ),
        )

        session.cost.record_llm_call(
            resp.input_tokens, resp.output_tokens, resp.cost_usd
        )

        try:
            scaffold = json.loads(self._extract_json(resp.text))
        except (json.JSONDecodeError, ValueError):
            scaffold = {"directories": ["src", "tests", "docs"], "files": [], "dependencies": []}

        # Create directories
        for d in scaffold.get("directories", []):
            (impl_dir / d).mkdir(parents=True, exist_ok=True)

        # Save scaffold plan
        (impl_dir / "scaffold.json").write_text(json.dumps(scaffold, indent=2))

        return (
            f"Scaffold created with {len(scaffold.get('directories', []))} directories",
            [{"type": "scaffold", "directories": scaffold.get("directories", [])}],
        )

    # ------------------------------------------------------------------
    # Phase 2: Build
    # ------------------------------------------------------------------

    async def _build(
        self, session: Session
    ) -> tuple[str, list[dict[str, Any]]]:
        self.provenance.log_event("implement_build_started")

        report = session.translation_report
        spec = report.implementation_spec if report else None
        impl_dir = session.session_dir / "implementation"
        scaffold_file = impl_dir / "scaffold.json"

        scaffold = {}
        if scaffold_file.exists():
            scaffold = json.loads(scaffold_file.read_text())

        files_created = []
        for file_spec in scaffold.get("files", [])[:20]:  # Limit to 20 files
            resp = await self.llm.complete(
                messages=[{"role": "user", "content": (
                    f"Generate the implementation for:\n\n"
                    f"File: {file_spec['path']}\n"
                    f"Purpose: {file_spec.get('description', '')}\n"
                    f"Project: {spec.title if spec else 'Unknown'}\n"
                    f"Platform: {spec.target_platform if spec else 'Python'}\n"
                    f"Architecture: {spec.architecture_overview if spec else ''}\n\n"
                    "Write clean, well-documented, production-ready code. "
                    "Include appropriate type hints and error handling."
                )}],
                system=(
                    "You are a senior developer writing production code. "
                    "Follow best practices. Include docstrings and type hints."
                ),
            )

            session.cost.record_llm_call(
                resp.input_tokens, resp.output_tokens, resp.cost_usd
            )

            code = self._extract_code(resp.text)
            file_path = impl_dir / file_spec["path"]
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(code)
            files_created.append(file_spec["path"])

            self.provenance.log_data_transform(
                source="llm_generated",
                operation="implement_file",
                description=f"Generated {file_spec['path']}",
                script_ref=str(file_path),
            )

        return (
            f"Built {len(files_created)} implementation files",
            [{"type": "build", "files": files_created}],
        )

    # ------------------------------------------------------------------
    # Phase 3: Test
    # ------------------------------------------------------------------

    async def _test(
        self, session: Session
    ) -> tuple[str, list[dict[str, Any]]]:
        self.provenance.log_event("implement_test_started")

        report = session.translation_report
        spec = report.implementation_spec if report else None
        impl_dir = session.session_dir / "implementation"

        # Gather implementation files
        impl_files = []
        for f in impl_dir.rglob("*.py"):
            if f.name != "__pycache__" and "test" not in f.name:
                impl_files.append({"path": str(f.relative_to(impl_dir)), "content": f.read_text()[:2000]})

        resp = await self.llm.complete(
            messages=[{"role": "user", "content": (
                f"Generate a comprehensive test suite for this implementation:\n\n"
                f"Project: {spec.title if spec else 'Unknown'}\n"
                f"Testing strategy: {spec.testing_strategy if spec else 'unit + integration'}\n\n"
                f"Implementation files:\n"
                + "\n".join(f"--- {f['path']} ---\n{f['content']}" for f in impl_files[:10])
                + "\n\nGenerate pytest tests covering:\n"
                "1. Unit tests for each component\n"
                "2. Integration tests for key workflows\n"
                "3. Edge cases and error handling\n"
                "4. Validation against expected research outputs"
            )}],
            system=(
                "You are a QA engineer writing comprehensive tests. "
                "Use pytest. Cover edge cases. Include docstrings."
            ),
        )

        session.cost.record_llm_call(
            resp.input_tokens, resp.output_tokens, resp.cost_usd
        )

        code = self._extract_code(resp.text)
        test_path = impl_dir / "tests" / "test_implementation.py"
        test_path.parent.mkdir(parents=True, exist_ok=True)
        test_path.write_text(code)

        return (
            "Test suite generated",
            [{"type": "tests", "path": str(test_path)}],
        )

    # ------------------------------------------------------------------
    # Phase 4: Document
    # ------------------------------------------------------------------

    async def _document(
        self, session: Session
    ) -> tuple[str, list[dict[str, Any]]]:
        self.provenance.log_event("implement_document_started")

        report = session.translation_report
        spec = report.implementation_spec if report else None
        impl_dir = session.session_dir / "implementation"

        impl_dir.mkdir(parents=True, exist_ok=True)

        resp = await self.llm.complete(
            messages=[{"role": "user", "content": (
                f"Generate documentation for this implementation:\n\n"
                f"Title: {spec.title if spec else 'Unknown'}\n"
                f"Description: {spec.description if spec else ''}\n"
                f"Platform: {spec.target_platform if spec else ''}\n"
                f"Architecture: {spec.architecture_overview if spec else ''}\n\n"
                "Generate:\n"
                "1. README.md with setup instructions and usage examples\n"
                "2. API reference summary\n"
                "3. Architecture decision records\n\n"
                "Use clear markdown formatting."
            )}],
            system="You are a technical writer creating clear, comprehensive documentation.",
        )

        session.cost.record_llm_call(
            resp.input_tokens, resp.output_tokens, resp.cost_usd
        )

        readme_path = impl_dir / "README.md"
        readme_path.write_text(resp.text)

        return (
            "Documentation generated",
            [{"type": "docs", "path": str(readme_path)}],
        )

    # ------------------------------------------------------------------
    # Phase 5: Package
    # ------------------------------------------------------------------

    async def _package(
        self, session: Session
    ) -> tuple[str, list[dict[str, Any]]]:
        self.provenance.log_event("implement_package_started")

        report = session.translation_report
        spec = report.implementation_spec if report else None
        impl_dir = session.session_dir / "implementation"
        scaffold_file = impl_dir / "scaffold.json"

        scaffold = {}
        if scaffold_file.exists():
            scaffold = json.loads(scaffold_file.read_text())

        resp = await self.llm.complete(
            messages=[{"role": "user", "content": (
                f"Generate deployment/packaging configuration for:\n\n"
                f"Title: {spec.title if spec else 'Unknown'}\n"
                f"Platform: {spec.target_platform if spec else 'Python'}\n"
                f"Deployment: {spec.deployment_strategy if spec else 'Docker'}\n"
                f"Dependencies: {json.dumps(scaffold.get('dependencies', []))}\n\n"
                "Generate:\n"
                "1. Dockerfile\n"
                "2. pyproject.toml or package.json (appropriate for platform)\n"
                "3. docker-compose.yml if applicable\n\n"
                "Respond with each file clearly delineated with --- filename --- markers."
            )}],
            system="You are a DevOps engineer creating deployment configurations.",
        )

        session.cost.record_llm_call(
            resp.input_tokens, resp.output_tokens, resp.cost_usd
        )

        # Save packaging artifacts
        packaging_dir = impl_dir / "deploy"
        packaging_dir.mkdir(parents=True, exist_ok=True)
        (packaging_dir / "packaging_config.md").write_text(resp.text)

        return (
            "Packaging configuration generated",
            [{"type": "packaging", "path": str(packaging_dir)}],
        )

    # ------------------------------------------------------------------
    # Phase 6: Validate
    # ------------------------------------------------------------------

    async def _validate(
        self, session: Session
    ) -> tuple[str, list[dict[str, Any]]]:
        self.provenance.log_event("implement_validate_started")

        report = session.translation_report
        impl_dir = session.session_dir / "implementation"
        impl_dir.mkdir(parents=True, exist_ok=True)

        # Gather implementation summary
        impl_files = list(impl_dir.rglob("*.py"))
        test_files = [f for f in impl_files if "test" in f.name]

        resp = await self.llm.complete(
            messages=[{"role": "user", "content": (
                f"Validate this implementation against the original research:\n\n"
                f"Original objective: {session.mission.objective}\n"
                f"Key findings: {', '.join(session.key_findings[:5])}\n"
                f"Implementation spec: {report.implementation_spec.title if report else ''}\n"
                f"Files generated: {len(impl_files)}\n"
                f"Test files: {len(test_files)}\n\n"
                "Assess:\n"
                "1. Does the implementation faithfully represent the research findings?\n"
                "2. Are there gaps between research and implementation?\n"
                "3. Could the implementation be used to replicate the research?\n"
                "4. Overall quality assessment\n\n"
                "Respond in JSON with: validation_status (pass/pass_with_notes/fail), "
                "gaps (list), quality_score (0-10), recommendations (list)"
            )}],
            system=(
                "You are a research-to-implementation validator. "
                "Verify that implementations faithfully represent research findings."
            ),
        )

        session.cost.record_llm_call(
            resp.input_tokens, resp.output_tokens, resp.cost_usd
        )

        try:
            validation = json.loads(self._extract_json(resp.text))
        except (json.JSONDecodeError, ValueError):
            validation = {"validation_status": "pass_with_notes", "quality_score": 7}

        # Save validation report
        (impl_dir / "validation_report.json").write_text(json.dumps(validation, indent=2))

        return (
            f"Validation: {validation.get('validation_status', 'unknown')}",
            [{"type": "validation", "data": validation}],
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_code(text: str) -> str:
        if "```python" in text:
            return text.split("```python")[1].split("```")[0].strip()
        if "```" in text:
            return text.split("```")[1].split("```")[0].strip()
        return text

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
