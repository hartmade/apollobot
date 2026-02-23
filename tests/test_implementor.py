"""Unit tests for Implement mode."""

import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from apollobot.core.mission import Mission, ResearchMode
from apollobot.core.session import Phase, Session
from apollobot.core.translation import (
    ImplementationSpec,
    TranslationReport,
)


class TestResearchImplementor:
    @pytest.fixture
    def mock_llm(self):
        llm = MagicMock()
        response = MagicMock()
        response.text = json.dumps({
            "directories": ["src", "tests", "docs"],
            "files": [
                {"path": "src/main.py", "description": "Main entry point"},
                {"path": "src/utils.py", "description": "Utility functions"},
            ],
            "dependencies": ["numpy", "pandas"],
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
    def implementor(self, mock_llm, mock_mcp, temp_dir):
        from apollobot.core.provenance import ProvenanceEngine
        from apollobot.agents.implementor import ResearchImplementor

        provenance = ProvenanceEngine(temp_dir)
        return ResearchImplementor(
            llm=mock_llm,
            mcp=mock_mcp,
            provenance=provenance,
        )

    @pytest.fixture
    def session(self, temp_dir):
        mission = Mission(
            objective="Test implementation",
            mode=ResearchMode.IMPLEMENT,
            source_session="session-translate-001",
        )
        mission.metadata["output_dir"] = str(temp_dir)
        session = Session(mission=mission)
        session.init_directories()
        session.key_findings = ["Finding 1"]

        # Set up translation report with spec
        report = TranslationReport(
            id="tr-test",
            implementation_spec=ImplementationSpec(
                title="Test API",
                target_platform="Python library",
                architecture_overview="Simple module",
            ),
        )
        session.translation_report = report.model_dump()

        return session

    @pytest.mark.asyncio
    async def test_scaffold_phase(self, implementor, session):
        """Test that scaffold creates project structure."""
        # Reconstruct report for the implementor
        from apollobot.core.translation import TranslationReport
        session.translation_report = TranslationReport.model_validate(session.translation_report)

        summary, findings = await implementor._scaffold(session)

        assert "Scaffold created" in summary
        impl_dir = session.session_dir / "implementation"
        assert impl_dir.exists()
        assert (impl_dir / "scaffold.json").exists()

    @pytest.mark.asyncio
    async def test_scaffold_creates_directories(self, implementor, session):
        """Test that scaffold creates the specified directories."""
        from apollobot.core.translation import TranslationReport
        session.translation_report = TranslationReport.model_validate(session.translation_report)

        await implementor._scaffold(session)

        impl_dir = session.session_dir / "implementation"
        scaffold = json.loads((impl_dir / "scaffold.json").read_text())
        for d in scaffold.get("directories", []):
            assert (impl_dir / d).exists()

    @pytest.mark.asyncio
    async def test_build_phase(self, implementor, session, mock_llm):
        """Test that build generates implementation files."""
        from apollobot.core.translation import TranslationReport
        session.translation_report = TranslationReport.model_validate(session.translation_report)

        # First scaffold
        await implementor._scaffold(session)

        # Mock code generation response
        mock_llm.complete.return_value.text = "```python\ndef main():\n    pass\n```"

        summary, findings = await implementor._build(session)
        assert "Built" in summary

    @pytest.mark.asyncio
    async def test_test_phase(self, implementor, session, mock_llm):
        """Test that test phase generates test suite."""
        from apollobot.core.translation import TranslationReport
        session.translation_report = TranslationReport.model_validate(session.translation_report)

        mock_llm.complete.return_value.text = "```python\nimport pytest\n\ndef test_example():\n    assert True\n```"

        summary, findings = await implementor._test(session)
        assert "Test suite" in summary

    @pytest.mark.asyncio
    async def test_document_phase(self, implementor, session, mock_llm):
        """Test that document phase generates README."""
        from apollobot.core.translation import TranslationReport
        session.translation_report = TranslationReport.model_validate(session.translation_report)

        mock_llm.complete.return_value.text = "# Test API\n\nA test implementation."

        summary, findings = await implementor._document(session)
        assert "Documentation" in summary
        assert (session.session_dir / "implementation" / "README.md").exists()

    @pytest.mark.asyncio
    async def test_package_phase(self, implementor, session, mock_llm):
        """Test that package phase generates deployment config."""
        from apollobot.core.translation import TranslationReport
        session.translation_report = TranslationReport.model_validate(session.translation_report)

        # Create scaffold first
        await implementor._scaffold(session)

        mock_llm.complete.return_value.text = "--- Dockerfile ---\nFROM python:3.11\n"

        summary, findings = await implementor._package(session)
        assert "Packaging" in summary

    @pytest.mark.asyncio
    async def test_validate_phase(self, implementor, session, mock_llm):
        """Test that validate phase checks implementation."""
        from apollobot.core.translation import TranslationReport
        session.translation_report = TranslationReport.model_validate(session.translation_report)

        mock_llm.complete.return_value.text = json.dumps({
            "validation_status": "pass",
            "quality_score": 8,
            "gaps": [],
            "recommendations": ["Add more tests"],
        })

        summary, findings = await implementor._validate(session)
        assert "Validation" in summary

    @pytest.mark.asyncio
    async def test_scaffold_without_spec_raises(self, implementor, session):
        """Test that scaffold fails without implementation spec."""
        session.translation_report = None

        with pytest.raises(RuntimeError, match="No implementation spec"):
            await implementor._scaffold(session)


class TestImplementPhases:
    """Test phase enum values for implement mode."""

    def test_implement_phases_exist(self):
        assert Phase.IMPLEMENT_SCAFFOLD == "implement_scaffold"
        assert Phase.IMPLEMENT_BUILD == "implement_build"
        assert Phase.IMPLEMENT_TEST == "implement_test"
        assert Phase.IMPLEMENT_DOCUMENT == "implement_document"
        assert Phase.IMPLEMENT_PACKAGE == "implement_package"
        assert Phase.IMPLEMENT_VALIDATE == "implement_validate"
