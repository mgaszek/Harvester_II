"""
Configuration management for Harvester II trading system.
Handles loading of config.json and environment variables.
"""

import json
import logging
import os
from pathlib import Path
import re
from typing import Any

from dotenv import load_dotenv


class Config:
    """Configuration manager for Harvester II trading system."""

    def __init__(self, config_path: str = "config.json", env_path: str = ".env"):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to config.json file
            env_path: Path to .env file
        """
        self.config_path = Path(config_path)
        # If config.json not found in current directory, try parent directory
        if not self.config_path.exists():
            parent_config = Path(__file__).parent.parent / config_path
            if parent_config.exists():
                self.config_path = parent_config

        self.env_path = Path(env_path)
        # If .env not found in current directory, try parent directory
        if not self.env_path.exists():
            parent_env = Path(__file__).parent.parent / env_path
            if parent_env.exists():
                self.env_path = parent_env

        self._config_data: dict[str, Any] = {}

        self._load_config()
        self._load_env()

    def _load_config(self) -> None:
        """Load configuration from JSON file."""
        try:
            if self.config_path.exists():
                with open(self.config_path) as f:
                    self._config_data = json.load(f)
            else:
                raise FileNotFoundError(f"Config file not found: {self.config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file: {e}")
        except OSError as e:
            raise RuntimeError(f"Failed to read config file: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error loading config: {e}")

    def _load_env(self) -> None:
        """Load environment variables from .env file."""
        try:
            if self.env_path.exists():
                load_dotenv(self.env_path)
                # Environment variables are now accessed directly via os.getenv()
                # No longer storing them in a dictionary to avoid potential security issues
        except OSError as e:
            print(f"Warning: Failed to read .env file: {e}")
        except Exception as e:
            print(f"Warning: Unexpected error loading .env file: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key: Configuration key (e.g., 'risk_management.equity')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self._config_data

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def get_env(self, key: str, default: str = "") -> str:
        """
        Get environment variable value.

        Args:
            key: Environment variable name
            default: Default value if not found

        Returns:
            Environment variable value
        """
        return os.getenv(key, default)

    @property
    def system(self) -> dict[str, Any]:
        """Get system configuration."""
        return self.get("system", {})

    @property
    def universe(self) -> dict[str, Any]:
        """Get universe configuration."""
        return self.get("universe", {})

    @property
    def signals(self) -> dict[str, Any]:
        """Get signals configuration."""
        return self.get("signals", {})

    @property
    def risk_management(self) -> dict[str, Any]:
        """Get risk management configuration."""
        return self.get("risk_management", {})

    @property
    def macro_risk(self) -> dict[str, Any]:
        """Get macro risk configuration."""
        return self.get("macro_risk", {})

    @property
    def data_sources(self) -> dict[str, Any]:
        """Get data sources configuration."""
        return self.get("data_sources", {})

    @property
    def trading(self) -> dict[str, Any]:
        """Get trading configuration."""
        return self.get("trading", {})

    @property
    def logging(self) -> dict[str, Any]:
        """Get logging configuration."""
        return self.get("logging", {})

    @property
    def notifications(self) -> dict[str, Any]:
        """Get notifications configuration."""
        return self.get("notifications", {})

    @property
    def backtesting(self) -> dict[str, Any]:
        """Get backtesting configuration."""
        return self.get("backtesting", {})

    def validate(self) -> bool:
        """
        Validate configuration completeness and data integrity.

        Returns:
            True if configuration is valid
        """
        required_sections = [
            "system",
            "universe",
            "signals",
            "risk_management",
            "macro_risk",
            "data_sources",
            "trading",
        ]

        for section in required_sections:
            if not self.get(section):
                print(f"Warning: Missing required configuration section: {section}")
                return False

        # Validate critical parameters
        if not self.get("universe.assets"):
            print("Error: No assets defined in universe")
            return False

        if not self.get("risk_management.equity"):
            print("Error: No equity amount defined")
            return False

        # Validate data types and ranges
        try:
            # Validate equity is positive number
            equity = self.get("risk_management.equity")
            if not isinstance(equity, (int, float)) or equity <= 0:
                print(f"Error: Equity must be a positive number, got {equity}")
                return False

            # Validate position sizing parameters
            max_position_pct = self.get("risk_management.max_position_percent", 0.1)
            if (
                not isinstance(max_position_pct, (int, float))
                or not 0 < max_position_pct <= 1
            ):
                print(
                    f"Error: max_position_percent must be between 0 and 1, got {max_position_pct}"
                )
                return False

            # Validate symbols are uppercase strings
            assets = self.get("universe.assets", [])
            if not isinstance(assets, list):
                print("Error: universe.assets must be a list")
                return False

            for asset in assets:
                if not isinstance(asset, str):
                    print(f"Error: Asset {asset} must be a string")
                    return False
                if asset != asset.upper():
                    print(f"Error: Asset {asset} must be uppercase")
                    return False

            # Validate signal thresholds
            cri_threshold = self.get("signals.cri_threshold", 0.5)
            if (
                not isinstance(cri_threshold, (int, float))
                or not 0 <= cri_threshold <= 1
            ):
                print(
                    f"Error: cri_threshold must be between 0 and 1, got {cri_threshold}"
                )
                return False

            panic_threshold = self.get("signals.panic_threshold", 2.0)
            if not isinstance(panic_threshold, (int, float)) or panic_threshold <= 0:
                print(f"Error: panic_threshold must be positive, got {panic_threshold}")
                return False

            # Validate macro risk parameters
            g_score_threshold = self.get("macro_risk.g_score_threshold", 2.0)
            if (
                not isinstance(g_score_threshold, (int, float))
                or g_score_threshold <= 0
            ):
                print(
                    f"Error: g_score_threshold must be positive, got {g_score_threshold}"
                )
                return False

            # Validate trading schedule
            run_time = self.get("trading.schedule.run_time", "16:00")
            if (
                not isinstance(run_time, str)
                or len(run_time) != 5
                or run_time[2] != ":"
            ):
                print(f"Error: run_time must be in HH:MM format, got {run_time}")
                return False

            try:
                hour, minute = map(int, run_time.split(":"))
                if not (0 <= hour <= 23 and 0 <= minute <= 59):
                    print(f"Error: Invalid time format for run_time: {run_time}")
                    return False
            except ValueError:
                print(f"Error: run_time must be valid time format, got {run_time}")
                return False

        except Exception as e:
            print(f"Error during configuration validation: {e}")
            return False

        return True

    def __str__(self) -> str:
        """String representation of configuration."""
        return f"Config(loaded from {self.config_path})"


# Global configuration instance
config = Config()


def get_config() -> Config:
    """Get the global configuration instance."""
    return config


def reload_config() -> Config:
    """Reload configuration from files."""
    global config
    config = Config()
    return config


class SensitiveDataFilter(logging.Filter):
    """Logging filter to redact sensitive data like API keys from log messages."""

    def __init__(self):
        super().__init__()
        # Patterns for sensitive data that should be redacted
        self.sensitive_patterns = [
            # API keys (various formats)
            r'([Aa][Pp][Ii]_?[Kk][Ee][Yy]|[Kk][Ee][Yy])\s*[:=]\s*["\']?([A-Za-z0-9_\-]{20,})["\']?',
            # Generic key patterns
            r'([Ss][Ee][Cc][Rr][Ee][Tt]|[Tt][Oo][Kk][Ee][Nn]|[Pp][Aa][Ss][Ss][Ww][Oo][Rr][Dd])\s*[:=]\s*["\']?([^"\s]{8,})["\']?',
            # Email patterns
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            # URLs with potential tokens
            r"https?://[^\s]*?(?:key|token|secret|password)[^\s]*",
            # Slack webhook URLs
            r"https://hooks\.slack\.com/services/[A-Z0-9/]+",
        ]

    def filter(self, record):
        """Filter log record to redact sensitive data."""
        if hasattr(record, "getMessage"):
            message = record.getMessage()
        else:
            message = str(record.msg)

        # Redact sensitive patterns
        for pattern in self.sensitive_patterns:
            message = re.sub(pattern, r"\1: [REDACTED]", message, flags=re.IGNORECASE)

        # Update the record
        record.msg = message
        record.message = message

        return True
