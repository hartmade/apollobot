"""
Core configuration and utilities for ApolloBot.

Provides:
- Path constants (APOLLO_HOME, APOLLO_SESSIONS_DIR, etc.)
- Configuration models (ApolloConfig, UserIdentity, APIConfig, ComputeConfig)
- Config loading/saving functions
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from apollobot.notifications.config import NotificationsConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

APOLLO_HOME: Path = Path.home() / ".apollobot"
APOLLO_SESSIONS_DIR: Path = Path.home() / "apollobot-research"
APOLLO_CONFIG_FILE: Path = APOLLO_HOME / "config.yaml"
APOLLO_SERVERS_FILE: Path = APOLLO_HOME / "servers.yaml"


# ---------------------------------------------------------------------------
# Configuration models
# ---------------------------------------------------------------------------


class UserIdentity(BaseModel):
    """User identity for research attribution."""

    name: str = ""
    affiliation: str = ""
    email: str = ""
    orcid: str = ""


class APIConfig(BaseModel):
    """API configuration for LLM providers."""

    default_provider: str = "anthropic"
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    minimax_api_key: str = ""

    def get_key(self) -> str:
        """Get the API key for the default provider."""
        if self.default_provider == "anthropic":
            return self.anthropic_api_key
        elif self.default_provider == "openai":
            return self.openai_api_key
        elif self.default_provider == "minimax":
            return self.minimax_api_key
        elif (
            self.default_provider == "acceptance"
            and os.getenv("APOLLOBOT_ENV", "development").lower() == "acceptance"
        ):
            return "acceptance-fixture"
        return ""


class ComputeConfig(BaseModel):
    """Compute resource configuration."""

    mode: str = "local"  # local, cloud, hybrid
    max_budget_usd: float = 50.0


class JournalConfig(BaseModel):
    """Legacy journal configuration retained for profile compatibility."""

    enabled: bool = False
    base_url: str = ""
    hmac_secret: str = ""
    timeout: float = 30.0


class ApolloConfig(BaseModel):
    """Main configuration for ApolloBot."""

    identity: UserIdentity = Field(default_factory=UserIdentity)
    api: APIConfig = Field(default_factory=APIConfig)
    compute: ComputeConfig = Field(default_factory=ComputeConfig)
    default_domain: str = "bioinformatics"
    default_mode: str = "hypothesis"
    output_dir: str = str(APOLLO_SESSIONS_DIR)
    custom_servers: list[dict[str, Any]] = Field(default_factory=list)
    notifications: NotificationsConfig = Field(default_factory=NotificationsConfig)
    journal: JournalConfig = Field(default_factory=JournalConfig)


# ---------------------------------------------------------------------------
# Config loading/saving
# ---------------------------------------------------------------------------


def load_config() -> ApolloConfig:
    """Load YAML configuration and overlay deploy-time environment secrets."""
    config = ApolloConfig()
    if APOLLO_CONFIG_FILE.exists():
        try:
            data = yaml.safe_load(APOLLO_CONFIG_FILE.read_text())
            config = ApolloConfig(**data)
        except Exception as error:
            logger.warning("Could not load ApolloBot config: %s", error)

    api_updates = {
        "default_provider": os.getenv("APOLLOBOT_MODEL_PROVIDER"),
        "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "minimax_api_key": os.getenv("MINIMAX_API_KEY"),
    }
    configured_api = config.api.model_dump()
    configured_api.update({key: value for key, value in api_updates.items() if value})

    config_updates: dict[str, Any] = {
        "api": APIConfig(**configured_api),
    }
    environment_fields = {
        "default_domain": os.getenv("APOLLOBOT_DEFAULT_DOMAIN"),
        "default_mode": os.getenv("APOLLOBOT_DEFAULT_MODE"),
        "output_dir": os.getenv("APOLLOBOT_OUTPUT_DIR"),
    }
    config_updates.update({key: value for key, value in environment_fields.items() if value})
    return config.model_copy(update=config_updates)


def save_config(config: ApolloConfig) -> None:
    """Save configuration to YAML file."""
    APOLLO_HOME.mkdir(parents=True, exist_ok=True)
    APOLLO_CONFIG_FILE.write_text(yaml.dump(config.model_dump(), default_flow_style=False))


def load_custom_servers() -> list[dict[str, Any]]:
    """Load custom MCP servers from servers.yaml."""
    if APOLLO_SERVERS_FILE.exists():
        try:
            data = yaml.safe_load(APOLLO_SERVERS_FILE.read_text())
            return data.get("custom_servers", [])
        except Exception as error:
            logger.warning("Could not load ApolloBot server config: %s", error)
    return []


__all__ = [
    "APOLLO_HOME",
    "APOLLO_SESSIONS_DIR",
    "APOLLO_CONFIG_FILE",
    "APOLLO_SERVERS_FILE",
    "UserIdentity",
    "APIConfig",
    "ComputeConfig",
    "ApolloConfig",
    "load_config",
    "save_config",
    "load_custom_servers",
    "JournalConfig",
    "NotificationsConfig",
]
