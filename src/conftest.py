"""
Pytest configuration and shared fixtures for Harvester II tests.
"""

from pathlib import Path
import sys
from unittest.mock import Mock

import pandas as pd
import pytest

# Add src directory to path for imports
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import modules inside fixtures to avoid top-level import issues


@pytest.fixture
def sample_config():
    """Create a sample configuration for testing."""
    from config import Config

    # Create a mock config that doesn't try to load files
    config = Mock(spec=Config)
    config.get.return_value = {}
    config.get_env.return_value = ""

    # Configure specific return values
    def mock_get(key, default=None):
        config_dict = {
            "system.lookback_window": 90,
            "universe.cri_threshold": 0.4,
            "signals.panic_threshold": 3.0,
            "macro_risk.g_score_threshold": 2,
            "signals.indicators.atr_period": 14,
            "signals.indicators.volume_period": 14,
            "signals.indicators.trends_period": 14,
            "risk_management.equity": 100000,
            "risk_management.base_position_fraction": 0.005,
            "risk_management.max_open_positions": 4,
            "risk_management.daily_drawdown_limit": 0.02,
            "risk_management.position_sizing.min_position_size": 100,
            "risk_management.position_sizing.max_position_size": 5000,
            "risk_management.position_sizing.risk_per_trade": 0.005,
            "universe.assets": ["SPY", "QQQ"],
            "trading.schedule.run_time": "16:00",
            "database.encrypted": False,
            "logging": {"file_path": "logs/test.log", "level": "INFO"},
        }
        return config_dict.get(key, default)

    config.get.side_effect = mock_get
    config.logging = {"file_path": "logs/test.log", "level": "INFO"}

    return config


@pytest.fixture
def mock_data_manager(sample_config):
    """Create a mock data manager for testing."""
    from data_manager import DataManager

    data_manager = Mock(spec=DataManager)
    data_manager.config = sample_config

    # Mock price data
    price_data = pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0],
            "High": [105.0, 106.0, 107.0],
            "Low": [95.0, 96.0, 97.0],
            "Close": [103.0, 104.0, 105.0],
            "Volume": [1000000, 1100000, 1200000],
        },
        index=pd.date_range("2023-01-01", periods=3),
    )

    data_manager.get_price_data.return_value = price_data
    data_manager.calculate_technical_indicators.return_value = price_data.assign(
        ATR=[1.5, 1.6, 1.7],
        Volume_MA=[1050000.0, 1100000.0, 1150000.0],
        Returns=[0.0, 0.009709, 0.009615],
        Returns_5d=[0.0, 0.0, 0.0],
        Volatility=[0.01, 0.015, 0.012],
    )

    # Mock trends data
    trends_data = pd.DataFrame(
        {"value": [75, 78, 80]}, index=pd.date_range("2023-01-01", periods=3)
    )
    data_manager.get_google_trends.return_value = trends_data

    # Mock get_universe_data to return a dict with symbol data
    def mock_get_universe_data(symbols, **kwargs):
        result = {}
        for symbol in symbols:
            result[symbol] = price_data.copy()
        return result

    data_manager.get_universe_data.side_effect = mock_get_universe_data

    return data_manager


@pytest.fixture
def sample_signal_calculator(sample_config, mock_data_manager):
    """Create a signal calculator with mocked dependencies."""
    from signals import SignalCalculator

    return SignalCalculator(sample_config, mock_data_manager)


@pytest.fixture
def sample_risk_manager(sample_config):
    """Create a risk manager for testing."""
    from risk_manager import RiskManager

    return RiskManager(sample_config)


@pytest.fixture
def mock_portfolio_manager(
    sample_config, sample_risk_manager, mock_data_manager, sample_signal_calculator
):
    """Create a mock portfolio manager for testing."""
    from portfolio import PortfolioManager

    portfolio_manager = Mock(spec=PortfolioManager)
    portfolio_manager.config = sample_config
    portfolio_manager.risk_manager = sample_risk_manager
    portfolio_manager.data_manager = mock_data_manager
    portfolio_manager.signal_calculator = sample_signal_calculator
    portfolio_manager.positions = {}
    portfolio_manager.orders = []

    return portfolio_manager


@pytest.fixture
def sample_components(
    sample_config,
    mock_data_manager,
    sample_signal_calculator,
    sample_risk_manager,
    mock_portfolio_manager,
):
    """Create a complete set of components for integration testing."""
    from engine import TradingEngine

    trading_engine = TradingEngine(
        sample_config,
        mock_data_manager,
        sample_signal_calculator,
        sample_risk_manager,
        mock_portfolio_manager,
    )

    return {
        "config": sample_config,
        "data_manager": mock_data_manager,
        "signal_calculator": sample_signal_calculator,
        "risk_manager": sample_risk_manager,
        "portfolio_manager": mock_portfolio_manager,
        "trading_engine": trading_engine,
    }


@pytest.fixture
def sample_price_data():
    """Create sample price data for testing."""
    return pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "High": [105.0, 106.0, 107.0, 108.0, 109.0],
            "Low": [95.0, 96.0, 97.0, 98.0, 99.0],
            "Close": [103.0, 104.0, 105.0, 106.0, 107.0],
            "Volume": [1000000, 1100000, 1200000, 1300000, 1400000],
        },
        index=pd.date_range("2023-01-01", periods=5),
    )


@pytest.fixture
def sample_trends_data():
    """Create sample Google Trends data for testing."""
    return pd.DataFrame(
        {"value": [75, 78, 80, 82, 85]}, index=pd.date_range("2023-01-01", periods=5)
    )
