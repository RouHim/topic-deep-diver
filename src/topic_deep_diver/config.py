"""
Configuration management for Topic Deep Diver MCP server.
"""

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


class ServerConfig(BaseModel):
    """Server configuration settings."""

    name: str = "Topic Deep Diver"
    version: str = "0.1.0"
    host: str = "localhost"
    port: int = 8000
    log_level: str = "INFO"


class MCPConfig(BaseModel):
    """MCP Protocol configuration."""

    protocol_version: str = "2025-06-18"
    transport: Literal["stdio", "sse", "streamable-http"] = "stdio"


class ResearchConfig(BaseModel):
    """Research configuration."""

    default_scope: str = "comprehensive"
    max_sources: int = 50
    timeout_minutes: int = 10
    cache_enabled: bool = True


class SearchEngineConfig(BaseModel):
    """Search engine configuration."""

    base_url: str = ""  # Must be configured in config.yaml or env
    enabled: bool = True
    timeout: int = 30
    max_results: int = 20


class SearchEnginesConfig(BaseModel):
    """All search engines configuration."""

    searxng: SearchEngineConfig = Field(default_factory=SearchEngineConfig)


class Configuration(BaseModel):
    """Main application configuration."""

    server: ServerConfig = Field(default_factory=ServerConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    research: ResearchConfig = Field(default_factory=ResearchConfig)
    search_engines: SearchEnginesConfig = Field(default_factory=SearchEnginesConfig)


class ConfigManager:
    """Manages application configuration from files and environment variables."""

    def __init__(self, config_path: Path | None = None):
        self.config_path = config_path or Path("config/config.yaml")
        self._config: Configuration | None = None
        self._load_environment()

    def _load_environment(self) -> None:
        """Load environment variables from .env file."""
        env_file = Path(".env")
        if env_file.exists():
            load_dotenv(env_file)

    def _load_yaml_config(self) -> dict[str, Any]:
        """Load configuration from YAML file."""
        if self.config_path.exists():
            with open(self.config_path) as f:
                return yaml.safe_load(f) or {}
        return {}

    def _override_with_env(self, config_dict: dict[str, Any]) -> dict[str, Any]:
        """Override configuration with environment variables."""
        # Server config
        if os.getenv("SERVER_HOST"):
            config_dict.setdefault("server", {})["host"] = os.getenv("SERVER_HOST")

        server_port = os.getenv("SERVER_PORT")
        if server_port:
            config_dict.setdefault("server", {})["port"] = int(server_port)

        if os.getenv("LOG_LEVEL"):
            config_dict.setdefault("server", {})["log_level"] = os.getenv("LOG_LEVEL")

        # Research config
        cache_enabled = os.getenv("CACHE_ENABLED")
        if cache_enabled:
            config_dict.setdefault("research", {})["cache_enabled"] = (
                cache_enabled.lower() == "true"
            )

        return config_dict

    def load_config(self) -> Configuration:
        """Load and validate configuration."""
        if self._config is None:
            # Load from YAML file
            config_dict = self._load_yaml_config()

            # Override with environment variables
            config_dict = self._override_with_env(config_dict)

            # Create and validate configuration
            self._config = Configuration(**config_dict)

        return self._config

    def get_config(self) -> Configuration:
        """Get current configuration."""
        return self.load_config()

    def reload_config(self) -> Configuration:
        """Reload configuration from files."""
        self._config = None
        return self.load_config()


# Global configuration manager instance
config_manager = ConfigManager()


def get_config() -> Configuration:
    """Get the current application configuration."""
    return config_manager.get_config()
