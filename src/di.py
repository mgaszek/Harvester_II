"""
Dependency Injection container for Harvester II.
Provides factory functions for creating components with their dependencies injected.
"""

from config import Config
from data_manager import DataManager
from signals import SignalCalculator
from risk_manager import RiskManager
from portfolio import PortfolioManager
from engine import TradingEngine


def create_config(config_path: str = "config.json") -> Config:
    """Create and return a Config instance."""
    return Config(config_path)


def create_data_manager(config: Config) -> DataManager:
    """Create and return a DataManager instance."""
    return DataManager(config)


def create_signal_calculator(config: Config, data_manager: DataManager) -> SignalCalculator:
    """Create and return a SignalCalculator instance."""
    return SignalCalculator(config, data_manager)


def create_risk_manager(config: Config) -> RiskManager:
    """Create and return a RiskManager instance."""
    return RiskManager(config)


def create_portfolio_manager(config: Config, risk_manager: RiskManager, data_manager: DataManager, signal_calculator: SignalCalculator) -> PortfolioManager:
    """Create and return a PortfolioManager instance."""
    return PortfolioManager(config, risk_manager, data_manager, signal_calculator)


def create_trading_engine(config_path: str = "config.json") -> TradingEngine:
    """Create and return a TradingEngine instance with all dependencies injected."""
    config = create_config(config_path)
    data_manager = create_data_manager(config)
    signal_calculator = create_signal_calculator(config, data_manager)
    risk_manager = create_risk_manager(config)
    portfolio_manager = create_portfolio_manager(config, risk_manager, data_manager, signal_calculator)

    return TradingEngine(config, data_manager, signal_calculator, risk_manager, portfolio_manager)


def create_components(config_path: str = "config.json"):
    """Create and return all components as a dictionary."""
    config = create_config(config_path)
    data_manager = create_data_manager(config)
    signal_calculator = create_signal_calculator(config, data_manager)
    risk_manager = create_risk_manager(config)
    portfolio_manager = create_portfolio_manager(config, risk_manager, data_manager, signal_calculator)
    trading_engine = TradingEngine(config, data_manager, signal_calculator, risk_manager, portfolio_manager)

    return {
        'config': config,
        'data_manager': data_manager,
        'signal_calculator': signal_calculator,
        'risk_manager': risk_manager,
        'portfolio_manager': portfolio_manager,
        'trading_engine': trading_engine
    }
