"""Tests for apollobot.core.mission module."""

import pytest
from pathlib import Path
import tempfile

from apollobot.core.mission import (
    Mission,
    ResearchMode,
    Constraints,
    Checkpoint,
    CheckpointAction,
    OutputSpec,
)


class TestResearchMode:
    """Tests for ResearchMode enum."""

    def test_all_modes_exist(self):
        """Verify all 5 research modes are defined."""
        assert ResearchMode.HYPOTHESIS.value == "hypothesis"
        assert ResearchMode.EXPLORATORY.value == "exploratory"
        assert ResearchMode.META_ANALYSIS.value == "meta-analysis"
        assert ResearchMode.REPLICATION.value == "replication"
        assert ResearchMode.SIMULATION.value == "simulation"

    def test_mode_from_string(self):
        """Verify modes can be created from strings."""
        assert ResearchMode("hypothesis") == ResearchMode.HYPOTHESIS
        assert ResearchMode("exploratory") == ResearchMode.EXPLORATORY


class TestConstraints:
    """Tests for Constraints model."""

    def test_default_constraints(self):
        """Verify default constraint values."""
        c = Constraints()
        assert c.compute_budget == 50.0
        assert c.time_limit == "48h"
        assert c.data_sources == "public_only"
        assert c.ethics == "observational_only"
        assert c.max_llm_calls == 5000
        assert c.max_datasets == 20

    def test_custom_constraints(self):
        """Verify custom constraints can be set."""
        c = Constraints(compute_budget=100.0, time_limit="24h")
        assert c.compute_budget == 100.0
        assert c.time_limit == "24h"


class TestMission:
    """Tests for Mission model."""

    def test_create_from_objective(self):
        """Test creating a mission from a simple objective string."""
        mission = Mission.from_objective(
            "Does exercise improve cognitive function?",
            mode="hypothesis",
            domain="bioinformatics",
        )
        assert mission.objective == "Does exercise improve cognitive function?"
        assert mission.mode == ResearchMode.HYPOTHESIS
        assert mission.domain == "bioinformatics"
        assert mission.id.startswith("session-")

    def test_mission_auto_title(self):
        """Test that title is auto-generated from objective."""
        mission = Mission(objective="A very long research question that exceeds eighty characters in length")
        assert len(mission.title) <= 80
        assert mission.title == mission.objective[:80].strip()

    def test_mission_default_resource_pack(self):
        """Test that resource_pack defaults to domain."""
        mission = Mission(objective="Test", domain="physics")
        assert mission.resource_pack == "physics"

    def test_time_limit_parsing(self):
        """Test parsing of time limit strings."""
        mission = Mission(objective="Test")

        mission.constraints.time_limit = "48h"
        assert mission.time_limit_seconds() == 48 * 3600

        mission.constraints.time_limit = "30m"
        assert mission.time_limit_seconds() == 30 * 60

        mission.constraints.time_limit = "2d"
        assert mission.time_limit_seconds() == 2 * 86400

    def test_mission_yaml_roundtrip(self, temp_dir):
        """Test saving and loading mission from YAML."""
        import yaml as pyyaml

        mission = Mission.from_objective(
            "Test objective",
            mode="exploratory",
            domain="cs_ml",
        )

        yaml_path = temp_dir / "mission.yaml"
        # Use model_dump with mode='json' to get serializable output
        data = mission.model_dump(mode='json')
        yaml_path.write_text(pyyaml.dump(data, default_flow_style=False))

        loaded = Mission.from_yaml(yaml_path)
        assert loaded.objective == mission.objective
        assert loaded.mode == mission.mode
        assert loaded.domain == mission.domain

    def test_mission_with_hypotheses(self):
        """Test mission with explicit hypotheses."""
        mission = Mission(
            objective="Test causal relationship",
            hypotheses=["H1: A causes B", "H2: B mediates C"],
            mode=ResearchMode.HYPOTHESIS,
        )
        assert len(mission.hypotheses) == 2
        assert "H1: A causes B" in mission.hypotheses

    def test_mission_replication_mode(self):
        """Test mission in replication mode with paper ID."""
        mission = Mission.from_objective(
            "Replicate findings",
            mode="replication",
            paper_id="arxiv:2401.12345",
        )
        assert mission.mode == ResearchMode.REPLICATION
        assert mission.paper_id == "arxiv:2401.12345"
