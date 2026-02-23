"""Tests for apollobot.core configuration module."""

import pytest
from pathlib import Path
import tempfile
import yaml

from apollobot.core import (
    APOLLO_HOME,
    APOLLO_SESSIONS_DIR,
    APOLLO_CONFIG_FILE,
    APOLLO_SERVERS_FILE,
    UserIdentity,
    APIConfig,
    ComputeConfig,
    ApolloConfig,
    load_config,
    save_config,
    load_custom_servers,
)


class TestPathConstants:
    """Tests for path constants."""

    def test_apollo_home_is_path(self):
        """Verify APOLLO_HOME is a Path object."""
        assert isinstance(APOLLO_HOME, Path)
        assert APOLLO_HOME.name == ".apollobot"

    def test_apollo_sessions_dir_is_path(self):
        """Verify APOLLO_SESSIONS_DIR is a Path object."""
        assert isinstance(APOLLO_SESSIONS_DIR, Path)
        assert APOLLO_SESSIONS_DIR.name == "apollobot-research"

    def test_config_file_in_home(self):
        """Verify config file is under APOLLO_HOME."""
        assert APOLLO_CONFIG_FILE.parent == APOLLO_HOME
        assert APOLLO_CONFIG_FILE.name == "config.yaml"

    def test_servers_file_in_home(self):
        """Verify servers file is under APOLLO_HOME."""
        assert APOLLO_SERVERS_FILE.parent == APOLLO_HOME
        assert APOLLO_SERVERS_FILE.name == "servers.yaml"


class TestUserIdentity:
    """Tests for UserIdentity model."""

    def test_default_identity(self):
        """Verify default identity is empty."""
        identity = UserIdentity()
        assert identity.name == ""
        assert identity.affiliation == ""
        assert identity.email == ""
        assert identity.orcid == ""

    def test_custom_identity(self):
        """Test creating identity with values."""
        identity = UserIdentity(
            name="Alice Researcher",
            affiliation="Example University",
            email="alice@example.edu",
            orcid="0000-0001-2345-6789",
        )
        assert identity.name == "Alice Researcher"
        assert identity.orcid == "0000-0001-2345-6789"


class TestAPIConfig:
    """Tests for APIConfig model."""

    def test_default_provider(self):
        """Verify default provider is anthropic."""
        config = APIConfig()
        assert config.default_provider == "anthropic"

    def test_get_key_anthropic(self):
        """Test getting Anthropic API key."""
        config = APIConfig(
            default_provider="anthropic",
            anthropic_api_key="sk-ant-test123",
        )
        assert config.get_key() == "sk-ant-test123"

    def test_get_key_openai(self):
        """Test getting OpenAI API key."""
        config = APIConfig(
            default_provider="openai",
            openai_api_key="sk-openai-test456",
        )
        assert config.get_key() == "sk-openai-test456"

    def test_get_key_empty_when_not_set(self):
        """Test get_key returns empty string when no key set."""
        config = APIConfig()
        assert config.get_key() == ""


class TestComputeConfig:
    """Tests for ComputeConfig model."""

    def test_default_compute_config(self):
        """Verify default compute configuration."""
        config = ComputeConfig()
        assert config.mode == "local"
        assert config.max_budget_usd == 50.0

    def test_custom_compute_config(self):
        """Test custom compute configuration."""
        config = ComputeConfig(mode="cloud", max_budget_usd=100.0)
        assert config.mode == "cloud"
        assert config.max_budget_usd == 100.0


class TestApolloConfig:
    """Tests for ApolloConfig model."""

    def test_default_config(self):
        """Verify default configuration values."""
        config = ApolloConfig()
        assert config.default_domain == "bioinformatics"
        assert config.default_mode == "hypothesis"
        assert isinstance(config.identity, UserIdentity)
        assert isinstance(config.api, APIConfig)
        assert isinstance(config.compute, ComputeConfig)

    def test_config_with_custom_servers(self):
        """Test config with custom MCP servers."""
        config = ApolloConfig(
            custom_servers=[
                {"name": "my-server", "url": "https://example.com/mcp"}
            ]
        )
        assert len(config.custom_servers) == 1
        assert config.custom_servers[0]["name"] == "my-server"


class TestConfigIO:
    """Tests for config save/load functions."""

    def test_load_config_returns_defaults_when_no_file(self, monkeypatch, temp_dir):
        """Test load_config returns defaults when file doesn't exist."""
        fake_config = temp_dir / "nonexistent" / "config.yaml"
        monkeypatch.setattr("apollobot.core.APOLLO_CONFIG_FILE", fake_config)

        config = load_config()
        assert isinstance(config, ApolloConfig)
        assert config.default_domain == "bioinformatics"

    def test_save_and_load_config(self, monkeypatch, temp_dir):
        """Test saving and loading configuration."""
        fake_home = temp_dir / ".apollobot"
        fake_config = fake_home / "config.yaml"
        monkeypatch.setattr("apollobot.core.APOLLO_HOME", fake_home)
        monkeypatch.setattr("apollobot.core.APOLLO_CONFIG_FILE", fake_config)

        config = ApolloConfig(
            identity=UserIdentity(name="Test User"),
            default_domain="physics",
        )
        save_config(config)

        assert fake_config.exists()

        loaded = load_config()
        assert loaded.identity.name == "Test User"
        assert loaded.default_domain == "physics"

    def test_load_custom_servers_empty_when_no_file(self, monkeypatch, temp_dir):
        """Test load_custom_servers returns empty list when no file."""
        fake_servers = temp_dir / "servers.yaml"
        monkeypatch.setattr("apollobot.core.APOLLO_SERVERS_FILE", fake_servers)

        servers = load_custom_servers()
        assert servers == []

    def test_load_custom_servers_from_file(self, monkeypatch, temp_dir):
        """Test loading custom servers from YAML file."""
        fake_servers = temp_dir / "servers.yaml"
        monkeypatch.setattr("apollobot.core.APOLLO_SERVERS_FILE", fake_servers)

        servers_data = {
            "custom_servers": [
                {"name": "test-server", "url": "https://test.example.com"}
            ]
        }
        fake_servers.write_text(yaml.dump(servers_data))

        servers = load_custom_servers()
        assert len(servers) == 1
        assert servers[0]["name"] == "test-server"
