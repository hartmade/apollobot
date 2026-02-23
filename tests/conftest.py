"""Pytest configuration and fixtures."""

import pytest
from pathlib import Path
import tempfile
import shutil


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test artifacts."""
    dirpath = tempfile.mkdtemp()
    yield Path(dirpath)
    shutil.rmtree(dirpath, ignore_errors=True)


@pytest.fixture
def sample_objective():
    """Sample research objective for testing."""
    return "Does gut microbiome diversity correlate with epigenetic age acceleration?"


@pytest.fixture
def sample_mission_data():
    """Sample mission data for testing."""
    return {
        "objective": "Test research objective",
        "mode": "hypothesis",
        "domain": "bioinformatics",
        "hypotheses": ["H1: Test hypothesis"],
    }
