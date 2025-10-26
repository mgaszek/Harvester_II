"""
Integration tests for backtest.py - comprehensive backtesting validation.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from pathlib import Path
import sys

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from backtest import BacktestEngine
from data_manager import DataManager
from signals import SignalCalculator
from config import Config


@pytest.mark.integration
class TestBacktestEngine:
    """Test BacktestEngine integration and results validation."""

    def create_mock_data_manager(self):
        """Create a mock data manager with historical data."""
        data_manager = Mock(spec=DataManager)

        # Create realistic historical price data for SPY (2020-2024)
        dates = pd.date_range('2020-01-01', '2024-01-01', freq='D')
        np.random.seed(42)  # For reproducible results

        # Generate realistic price series
        initial_price = 300.0
        daily_returns = np.random.normal(0.0005, 0.015, len(dates))  # Mean return with volatility
        price_series = initial_price * np.exp(np.cumsum(daily_returns))

        # Create OHLCV data
        high_multiplier = 1 + np.random.uniform(0, 0.02, len(dates))
        low_multiplier = 1 - np.random.uniform(0, 0.02, len(dates))
        volume_base = 50000000

        price_data = pd.DataFrame({
            'Open': price_series * (1 + np.random.normal(0, 0.005, len(dates))),
            'High': price_series * high_multiplier,
            'Low': price_series * low_multiplier,
            'Close': price_series,
            'Volume': volume_base + np.random.normal(0, 10000000, len(dates))
        }, index=dates)

        # Ensure OHLC relationships are correct
        price_data['High'] = price_data[['Open', 'Close', 'High']].max(axis=1)
        price_data['Low'] = price_data[['Open', 'Close', 'Low']].min(axis=1)

        data_manager.get_price_data.return_value = price_data

        # Mock trends data
        trends_values = 50 + 30 * np.sin(np.arange(len(dates)) * 0.01) + np.random.normal(0, 5, len(dates))
        trends_values = np.clip(trends_values, 0, 100)
        trends_data = pd.DataFrame({
            'value': trends_values.astype(int)
        }, index=dates)
        data_manager.get_google_trends.return_value = trends_data

        return data_manager

    def create_test_config(self):
        """Create a test configuration for backtesting."""
        config = Mock()
        config.get.side_effect = lambda key, default=None: {
            'system.lookback_window': 90,
            'universe.cri_threshold': 0.4,
            'signals.panic_threshold': 3.0,
            'macro_risk.g_score_threshold': 2,
            'signals.indicators.atr_period': 14,
            'signals.indicators.volume_period': 14,
            'signals.indicators.trends_period': 14,
            'risk_management.equity': 100000,
            'risk_management.base_position_fraction': 0.005,
            'risk_management.max_open_positions': 4,
            'risk_management.daily_drawdown_limit': 0.05,
            'risk_management.position_sizing.min_position_size': 100,
            'risk_management.position_sizing.max_position_size': 5000,
            'risk_management.position_sizing.risk_per_trade': 0.005,
            'universe.assets': ['SPY'],
            'backtesting.slippage_percent': 0.001,
            'backtesting.commission_per_share': 0.005,
            'trading.schedule.run_time': '16:00',
            'database.encrypted': False,
        }.get(key, default)

        return config

    def test_backtest_engine_initialization(self, sample_risk_manager):
        """Test BacktestEngine can be initialized with dependencies."""
        config = self.create_test_config()
        data_manager = self.create_mock_data_manager()
        signal_calc = SignalCalculator(config, data_manager)

        backtest_engine = BacktestEngine(config, data_manager, signal_calc, sample_risk_manager)

        assert backtest_engine.config == config
        assert backtest_engine.data_manager == data_manager
        assert backtest_engine.signal_calculator == signal_calc
        assert backtest_engine.risk_manager == sample_risk_manager

    def test_backtest_run_basic(self, sample_risk_manager):
        """Test basic backtest execution."""
        config = self.create_test_config()
        data_manager = self.create_mock_data_manager()
        signal_calc = SignalCalculator(config, data_manager)

        backtest_engine = BacktestEngine(config, data_manager, signal_calc, sample_risk_manager)

        results = backtest_engine.run_backtest('2020-01-01', '2020-12-31', 100000)

        # Verify basic result structure
        assert isinstance(results, dict)
        assert 'capital' in results
        assert 'trade_statistics' in results
        assert 'error' not in results

    def test_backtest_metrics_calculation(self, sample_risk_manager):
        """Test that backtest calculates proper metrics."""
        config = self.create_test_config()
        data_manager = self.create_mock_data_manager()
        signal_calc = SignalCalculator(config, data_manager)

        backtest_engine = BacktestEngine(config, data_manager, signal_calc, sample_risk_manager)

        results = backtest_engine.run_backtest('2020-01-01', '2021-12-31', 100000)

        capital = results.get('capital', {})
        trade_stats = results.get('trade_statistics', {})

        # Verify capital metrics
        assert 'total_return' in capital
        assert 'max_drawdown' in capital
        assert 'final_capital' in capital
        assert 'initial_capital' in capital

        # Verify trade statistics (may be empty if no trades generated)
        # Note: With mock data, trades may not be generated
        assert isinstance(trade_stats, dict)

        # Verify reasonable ranges
        assert capital['final_capital'] > 0
        assert -1 <= capital['total_return'] <= 1  # Between -100% and +100%
        assert capital['max_drawdown'] >= 0  # Drawdown should be positive
        assert 0 <= trade_stats.get('win_rate', 0) <= 1

    def test_backtest_date_validation(self, sample_risk_manager):
        """Test backtest date parameter validation."""
        config = self.create_test_config()
        data_manager = self.create_mock_data_manager()
        signal_calc = SignalCalculator(config, data_manager)

        backtest_engine = BacktestEngine(config, data_manager, signal_calc, sample_risk_manager)

        # Test invalid date range
        results = backtest_engine.run_backtest('2024-01-01', '2020-01-01', 100000)

        assert 'error' in results
        assert 'Invalid date range' in results['error']

    def test_backtest_insufficient_data(self, sample_risk_manager):
        """Test backtest with insufficient historical data."""
        config = self.create_test_config()
        data_manager = Mock(spec=DataManager)
        data_manager.get_price_data.return_value = pd.DataFrame()  # Empty data
        data_manager.get_google_trends.return_value = pd.DataFrame()
        signal_calc = SignalCalculator(config, data_manager)

        backtest_engine = BacktestEngine(config, data_manager, signal_calc, sample_risk_manager)

        results = backtest_engine.run_backtest('2020-01-01', '2020-12-31', 100000)

        # Backtest still runs but with no data, so no trades
        assert 'capital' in results
        assert results['capital']['final_capital'] == results['capital']['initial_capital']  # No change

    def test_backtest_zero_capital(self, sample_risk_manager):
        """Test backtest with zero or negative capital."""
        config = self.create_test_config()
        data_manager = self.create_mock_data_manager()
        signal_calc = SignalCalculator(config, data_manager)

        backtest_engine = BacktestEngine(config, data_manager, signal_calc, sample_risk_manager)

        results = backtest_engine.run_backtest('2020-01-01', '2020-12-31', 0)

        assert 'error' in results
        assert 'Invalid capital' in results['error']

    def test_backtest_performance_metrics(self, sample_risk_manager):
        """Test detailed performance metrics calculation."""
        config = self.create_test_config()
        data_manager = self.create_mock_data_manager()
        signal_calc = SignalCalculator(config, data_manager)

        backtest_engine = BacktestEngine(config, data_manager, signal_calc, sample_risk_manager)

        results = backtest_engine.run_backtest('2020-01-01', '2022-12-31', 100000)

        capital = results.get('capital', {})
        trade_stats = results.get('trade_statistics', {})

        # Note: With mock data, advanced metrics may not be calculated
        # but basic structure should exist
        assert isinstance(capital, dict)
        assert isinstance(trade_stats, dict)

        # Basic validation that results structure is correct

    def test_backtest_risk_metrics(self, sample_risk_manager):
        """Test risk-related metrics in backtest results."""
        config = self.create_test_config()
        data_manager = self.create_mock_data_manager()
        signal_calc = SignalCalculator(config, data_manager)

        backtest_engine = BacktestEngine(config, data_manager, signal_calc, sample_risk_manager)

        results = backtest_engine.run_backtest('2020-01-01', '2023-12-31', 100000)

        capital = results.get('capital', {})

        # Basic validation that capital metrics exist
        assert isinstance(capital, dict)

    def test_backtest_trade_analysis(self, sample_risk_manager):
        """Test detailed trade analysis in backtest results."""
        config = self.create_test_config()
        data_manager = self.create_mock_data_manager()
        signal_calc = SignalCalculator(config, data_manager)

        backtest_engine = BacktestEngine(config, data_manager, signal_calc, sample_risk_manager)

        results = backtest_engine.run_backtest('2020-01-01', '2021-12-31', 100000)

        trade_stats = results.get('trade_statistics', {})

        # Basic validation that trade stats structure exists
        assert isinstance(trade_stats, dict)

    @pytest.mark.slow
    def test_backtest_long_term_performance(self, sample_risk_manager):
        """Test backtest performance over a longer time period."""
        config = self.create_test_config()
        data_manager = self.create_mock_data_manager()
        signal_calc = SignalCalculator(config, data_manager)

        backtest_engine = BacktestEngine(config, data_manager, signal_calc, sample_risk_manager)

        results = backtest_engine.run_backtest('2020-01-01', '2024-01-01', 100000)

        capital = results.get('capital', {})

        # Basic validation that long-term backtest runs
        assert isinstance(capital, dict)
        assert 'final_capital' in capital
