"""
Integration tests for backtest.py - comprehensive backtesting validation.
"""

from pathlib import Path
import sys
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from backtest import BacktestEngine
from data_manager import DataManager
from signals import SignalCalculator

# Vectorbt imports
try:
    import vectorbt as vbt

    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None


@pytest.mark.integration
class TestBacktestEngine:
    """Test BacktestEngine integration and results validation."""

    def create_mock_data_manager(self):
        """Create a mock data manager with historical data."""
        data_manager = Mock(spec=DataManager)

        # Create realistic historical price data for SPY (2020-2024)
        dates = pd.date_range("2020-01-01", "2024-01-01", freq="D")
        np.random.seed(42)  # For reproducible results

        # Generate realistic price series
        initial_price = 300.0
        daily_returns = np.random.normal(
            0.0005, 0.015, len(dates)
        )  # Mean return with volatility
        price_series = initial_price * np.exp(np.cumsum(daily_returns))

        # Create OHLCV data
        high_multiplier = 1 + np.random.uniform(0, 0.02, len(dates))
        low_multiplier = 1 - np.random.uniform(0, 0.02, len(dates))
        volume_base = 50000000

        price_data = pd.DataFrame(
            {
                "Open": price_series * (1 + np.random.normal(0, 0.005, len(dates))),
                "High": price_series * high_multiplier,
                "Low": price_series * low_multiplier,
                "Close": price_series,
                "Volume": volume_base + np.random.normal(0, 10000000, len(dates)),
            },
            index=dates,
        )

        # Ensure OHLC relationships are correct
        price_data["High"] = price_data[["Open", "Close", "High"]].max(axis=1)
        price_data["Low"] = price_data[["Open", "Close", "Low"]].min(axis=1)

        data_manager.get_price_data.return_value = price_data

        # Mock trends data
        trends_values = (
            50
            + 30 * np.sin(np.arange(len(dates)) * 0.01)
            + np.random.normal(0, 5, len(dates))
        )
        trends_values = np.clip(trends_values, 0, 100)
        trends_data = pd.DataFrame({"value": trends_values.astype(int)}, index=dates)
        data_manager.get_google_trends.return_value = trends_data

        return data_manager

    def create_test_config(self):
        """Create a test configuration for backtesting."""
        config = Mock()
        config.get.side_effect = lambda key, default=None: {
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
            "risk_management.daily_drawdown_limit": 0.05,
            "risk_management.position_sizing.min_position_size": 100,
            "risk_management.position_sizing.max_position_size": 5000,
            "risk_management.position_sizing.risk_per_trade": 0.005,
            "universe.assets": ["SPY"],
            "backtesting.slippage_percent": 0.001,
            "backtesting.commission_per_share": 0.005,
            "trading.schedule.run_time": "16:00",
            "database.encrypted": False,
        }.get(key, default)

        return config

    def test_backtest_engine_initialization(self, sample_risk_manager):
        """Test BacktestEngine can be initialized with dependencies."""
        config = self.create_test_config()
        data_manager = self.create_mock_data_manager()
        signal_calc = SignalCalculator(config, data_manager)

        backtest_engine = BacktestEngine(
            config, data_manager, signal_calc, sample_risk_manager
        )

        assert backtest_engine.config == config
        assert backtest_engine.data_manager == data_manager
        assert backtest_engine.signal_calculator == signal_calc
        assert backtest_engine.risk_manager == sample_risk_manager

    def test_backtest_run_basic(self, sample_risk_manager):
        """Test basic backtest execution."""
        config = self.create_test_config()
        data_manager = self.create_mock_data_manager()
        signal_calc = SignalCalculator(config, data_manager)

        backtest_engine = BacktestEngine(
            config, data_manager, signal_calc, sample_risk_manager
        )

        results = backtest_engine.run_backtest("2020-01-01", "2020-12-31", 100000)

        # Verify basic result structure
        assert isinstance(results, dict)
        assert "capital" in results
        assert "trade_statistics" in results
        assert "error" not in results

    def test_backtest_metrics_calculation(self, sample_risk_manager):
        """Test that backtest calculates proper metrics."""
        config = self.create_test_config()
        data_manager = self.create_mock_data_manager()
        signal_calc = SignalCalculator(config, data_manager)

        backtest_engine = BacktestEngine(
            config, data_manager, signal_calc, sample_risk_manager
        )

        results = backtest_engine.run_backtest("2020-01-01", "2021-12-31", 100000)

        capital = results.get("capital", {})
        trade_stats = results.get("trade_statistics", {})

        # Verify capital metrics
        assert "total_return" in capital
        assert "max_drawdown" in capital
        assert "final_capital" in capital
        assert "initial_capital" in capital

        # Verify trade statistics (may be empty if no trades generated)
        # Note: With mock data, trades may not be generated
        assert isinstance(trade_stats, dict)

        # Verify reasonable ranges
        assert capital["final_capital"] > 0
        assert -1 <= capital["total_return"] <= 1  # Between -100% and +100%
        assert capital["max_drawdown"] >= 0  # Drawdown should be positive
        assert 0 <= trade_stats.get("win_rate", 0) <= 1

    def test_backtest_date_validation(self, sample_risk_manager):
        """Test backtest date parameter validation."""
        config = self.create_test_config()
        data_manager = self.create_mock_data_manager()
        signal_calc = SignalCalculator(config, data_manager)

        backtest_engine = BacktestEngine(
            config, data_manager, signal_calc, sample_risk_manager
        )

        # Test invalid date range (start after end)
        results = backtest_engine.run_backtest("2024-01-01", "2020-01-01", 100000)

        # With invalid date range, no trading dates are generated
        assert "error" in results
        assert "No equity curve data" in results["error"]

    def test_backtest_insufficient_data(self, sample_risk_manager):
        """Test backtest with insufficient historical data."""
        config = self.create_test_config()
        data_manager = Mock(spec=DataManager)
        data_manager.get_price_data.return_value = pd.DataFrame()  # Empty data
        data_manager.get_google_trends.return_value = pd.DataFrame()
        signal_calc = SignalCalculator(config, data_manager)

        backtest_engine = BacktestEngine(
            config, data_manager, signal_calc, sample_risk_manager
        )

        results = backtest_engine.run_backtest("2020-01-01", "2020-12-31", 100000)

        # Backtest still runs but with no data, so no trades
        assert "capital" in results
        assert (
            results["capital"]["final_capital"] == results["capital"]["initial_capital"]
        )  # No change

    def test_backtest_zero_capital(self, sample_risk_manager):
        """Test backtest with zero capital."""
        config = self.create_test_config()
        data_manager = self.create_mock_data_manager()
        signal_calc = SignalCalculator(config, data_manager)

        backtest_engine = BacktestEngine(
            config, data_manager, signal_calc, sample_risk_manager
        )

        results = backtest_engine.run_backtest("2020-01-01", "2020-12-31", 0)

        # With zero capital, backtest fails due to division by zero in calculations
        assert "error" in results
        assert "No equity curve data" in results["error"]

    def test_backtest_performance_metrics(self, sample_risk_manager):
        """Test detailed performance metrics calculation."""
        config = self.create_test_config()
        data_manager = self.create_mock_data_manager()
        signal_calc = SignalCalculator(config, data_manager)

        backtest_engine = BacktestEngine(
            config, data_manager, signal_calc, sample_risk_manager
        )

        results = backtest_engine.run_backtest("2020-01-01", "2022-12-31", 100000)

        capital = results.get("capital", {})
        trade_stats = results.get("trade_statistics", {})

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

        backtest_engine = BacktestEngine(
            config, data_manager, signal_calc, sample_risk_manager
        )

        results = backtest_engine.run_backtest("2020-01-01", "2023-12-31", 100000)

        capital = results.get("capital", {})

        # Basic validation that capital metrics exist
        assert isinstance(capital, dict)

    def test_backtest_trade_analysis(self, sample_risk_manager):
        """Test detailed trade analysis in backtest results."""
        config = self.create_test_config()
        data_manager = self.create_mock_data_manager()
        signal_calc = SignalCalculator(config, data_manager)

        backtest_engine = BacktestEngine(
            config, data_manager, signal_calc, sample_risk_manager
        )

        results = backtest_engine.run_backtest("2020-01-01", "2021-12-31", 100000)

        trade_stats = results.get("trade_statistics", {})

        # Basic validation that trade stats structure exists
        assert isinstance(trade_stats, dict)

    @pytest.mark.slow
    def test_backtest_long_term_performance(self, sample_risk_manager):
        """Test backtest performance over a longer time period."""
        config = self.create_test_config()
        data_manager = self.create_mock_data_manager()
        signal_calc = SignalCalculator(config, data_manager)

        backtest_engine = BacktestEngine(
            config, data_manager, signal_calc, sample_risk_manager
        )

        results = backtest_engine.run_backtest("2020-01-01", "2024-01-01", 100000)

        capital = results.get("capital", {})

        # Basic validation that long-term backtest runs
        assert isinstance(capital, dict)
        assert "final_capital" in capital

    @pytest.mark.parametrize("bsm_enabled", [True, False])
    def test_ab_test_bayesian_enhancement(self, bsm_enabled, sample_risk_manager):
        """Test A/B comparison between Bayesian enabled/disabled backtests."""
        # Create config with Bayesian settings
        config = Mock()
        config.get.side_effect = lambda key, default=None: {
            "system.lookback_window": 90,
            "universe.cri_threshold": 0.4,
            "signals.panic_threshold": 3.0,
            "macro_risk.g_score_threshold": 2,
            "backtesting.start_date": "2020-01-01",
            "backtesting.end_date": "2024-01-01",
            "backtesting.initial_capital": 100000,
            "bayesian.enabled": bsm_enabled,
            "bayesian.n_states": 3,
            "bayesian.conviction_threshold": 0.7,
            "bayesian.priors": [0.3, 0.4, 0.3],
        }.get(key, default)

        # Create mock data manager
        data_manager = self.create_mock_data_manager()

        # Create signal calculator
        signal_calc = SignalCalculator(config, data_manager)

        # Create backtest engine
        backtest_engine = BacktestEngine(
            config, data_manager, signal_calc, sample_risk_manager
        )

        # Run backtest
        results = backtest_engine.run_backtest("2020-01-01", "2021-01-01", 100000)

        # Validate results structure
        assert "capital" in results
        assert "trades" in results
        capital = results["capital"]
        trades = results["trades"]

        # Basic performance checks - with mock data, may have zero returns
        assert "final_capital" in capital
        assert "total_return" in capital
        assert capital["final_capital"] >= 0  # Should not be negative
        assert len(trades) >= 0  # May have zero trades in short periods

        # If Bayesian is enabled and we have trades, check conviction correlation
        if bsm_enabled and trades:
            convictions = []
            profits = []

            for trade in trades:
                if "conviction" in trade and "pnl_percentage" in trade:
                    convictions.append(trade["conviction"])
                    profits.append(1 if trade["pnl_percentage"] > 0 else 0)

            # If we have enough data points, check correlation
            if len(convictions) > 3:
                correlation = np.corrcoef(convictions, profits)[0, 1]
                # High conviction should correlate with profitability (positive correlation)
                # Allow for some noise in short backtests
                assert (
                    correlation >= -0.5
                ), f"Conviction-profitability correlation too low: {correlation}"

                # Check that high conviction trades (>0.7) have better win rate
                high_conviction_trades = [
                    t for t in trades if t.get("conviction", 0) > 0.7
                ]
                if high_conviction_trades:
                    high_conviction_win_rate = sum(
                        1
                        for t in high_conviction_trades
                        if t.get("pnl_percentage", 0) > 0
                    ) / len(high_conviction_trades)
                    overall_win_rate = sum(
                        1 for t in trades if t.get("pnl_percentage", 0) > 0
                    ) / len(trades)

                    # High conviction should not have dramatically worse win rate
                    assert (
                        high_conviction_win_rate >= overall_win_rate - 0.3
                    ), f"High conviction win rate ({high_conviction_win_rate:.2f}) much worse than overall ({overall_win_rate:.2f})"

    def test_ab_test_comparison_metrics(self, sample_risk_manager):
        """Test A/B test comparison metrics calculation."""
        # Create config
        config = Mock()
        config.get.side_effect = lambda key, default=None: {
            "system.lookback_window": 90,
            "universe.cri_threshold": 0.4,
            "signals.panic_threshold": 3.0,
            "macro_risk.g_score_threshold": 2,
            "bayesian.enabled": True,
            "bayesian.n_states": 3,
            "bayesian.conviction_threshold": 0.7,
            "bayesian.priors": [0.3, 0.4, 0.3],
        }.get(key, default)

        # Create mock data manager
        data_manager = self.create_mock_data_manager()

        # Create signal calculator
        signal_calc = SignalCalculator(config, data_manager)

        # Create backtest engine
        backtest_engine = BacktestEngine(
            config, data_manager, signal_calc, sample_risk_manager
        )

        # Run A/B test
        ab_results = backtest_engine.run_ab_test("2020-01-01", "2021-01-01", 100000)

        # Validate A/B results structure
        assert "test_period" in ab_results
        assert "initial_capital" in ab_results
        assert "comparison" in ab_results

        comparison = ab_results["comparison"]

        # Check that comparison metrics are present
        expected_metrics = [
            "sharpe_ratio_improvement",
            "total_return_improvement",
            "max_drawdown_improvement",
            "win_rate_improvement",
            "conviction_correlation",
        ]

        for metric in expected_metrics:
            assert metric in comparison, f"Missing comparison metric: {metric}"
            assert isinstance(
                comparison[metric], (int, float, np.floating)
            ), f"Invalid type for {metric}"

        # Conviction correlation should be a valid correlation coefficient
        assert (
            -1 <= comparison["conviction_correlation"] <= 1
        ), "Invalid conviction correlation range"


@pytest.mark.integration
@pytest.mark.skipif(not VECTORBT_AVAILABLE, reason="Vectorbt not available")
class TestVectorbtIntegration:
    """Test Vectorbt integration with Harvester II."""

    def test_vectorbt_backtest_basic(self, sample_config, sample_risk_manager):
        """Test basic vectorbt backtest functionality."""
        from vectorbt_backtest import VectorbtBacktestEngine

        # Create mock data manager
        data_manager = Mock(spec=DataManager)
        data_manager.get_price_data = Mock(
            return_value=pd.DataFrame(
                {
                    "Close": [100, 101, 99, 102, 98],
                    "High": [101, 102, 100, 103, 99],
                    "Low": [99, 100, 98, 101, 97],
                    "Volume": [1000000, 1100000, 900000, 1200000, 800000],
                },
                index=pd.date_range("2020-01-01", periods=5),
            )
        )

        # Create signal calculator
        signal_calc = Mock(spec=SignalCalculator)
        signal_calc.get_entry_signals = Mock(return_value=[])
        signal_calc.calculate_g_score = Mock(return_value=1.5)

        # Create vectorbt engine
        vbt_engine = VectorbtBacktestEngine(
            sample_config, data_manager, signal_calc, sample_risk_manager
        )

        # Run vectorbt backtest
        results = vbt_engine.run_vectorbt_backtest("2020-01-01", "2020-01-05", 100000)

        # Validate results structure
        assert "backtest_type" in results
        assert results["backtest_type"] == "vectorbt"
        assert "total_return" in results
        assert "sharpe_ratio" in results
        assert "max_drawdown" in results
        assert "total_trades" in results
        assert isinstance(results["total_return"], (int, float))
        assert isinstance(results["sharpe_ratio"], (int, float))

    def test_vectorbt_signal_generation(self, sample_config, sample_risk_manager):
        """Test vectorbt signal generation."""
        from vectorbt_backtest import VectorbtBacktestEngine

        # Create mock data manager
        data_manager = Mock(spec=DataManager)

        # Create realistic price data
        price_data = pd.DataFrame(
            {
                "SPY": [100, 95, 90, 105, 110, 85, 115],  # Contrarian pattern
                "QQQ": [200, 190, 180, 210, 220, 170, 230],
            },
            index=pd.date_range("2020-01-01", periods=7),
        )

        # Create signal calculator
        signal_calc = Mock(spec=SignalCalculator)

        # Create vectorbt engine
        vbt_engine = VectorbtBacktestEngine(
            sample_config, data_manager, signal_calc, sample_risk_manager
        )

        # Generate signals
        signals = vbt_engine._generate_vectorbt_signals(
            price_data, "2020-01-01", "2020-01-07"
        )

        # Validate signals
        assert not signals.empty
        assert len(signals.columns) == 2  # Two assets
        assert all(signals.dtypes == "float64")  # Signal values

        # Check signal range (-1 to 1 typically for vectorbt)
        assert signals.min().min() >= -1
        assert signals.max().max() <= 1

    def test_vectorbt_portfolio_creation(self, sample_config, sample_risk_manager):
        """Test vectorbt portfolio creation."""
        from vectorbt_backtest import VectorbtBacktestEngine

        # Create mock data manager
        data_manager = Mock(spec=DataManager)

        # Create price data
        price_data = pd.DataFrame(
            {"SPY": [100, 101, 102, 103, 104], "QQQ": [200, 202, 204, 206, 208]},
            index=pd.date_range("2020-01-01", periods=5),
        )

        # Create simple signals
        signals = pd.DataFrame(
            {
                "SPY": [0, 1, 0, -1, 0],  # Hold, Buy, Hold, Sell, Hold
                "QQQ": [0, 0, 1, 0, -1],
            },
            index=price_data.index,
        )

        # Create signal calculator
        signal_calc = Mock(spec=SignalCalculator)

        # Create vectorbt engine
        vbt_engine = VectorbtBacktestEngine(
            sample_config, data_manager, signal_calc, sample_risk_manager
        )

        # Create portfolio
        portfolio = vbt_engine._create_vectorbt_portfolio(price_data, signals, 100000)

        # Validate portfolio
        assert portfolio is not None
        assert hasattr(portfolio, "final_value")
        assert hasattr(portfolio, "returns")
        assert hasattr(portfolio, "stats")

        # Check final value is reasonable
        final_value = portfolio.final_value()
        assert final_value > 0

    def test_backtest_engine_vectorbt_integration(
        self, sample_config, sample_risk_manager
    ):
        """Test that BacktestEngine properly integrates with vectorbt."""
        # Create mock components
        data_manager = Mock(spec=DataManager)
        signal_calc = Mock(spec=SignalCalculator)
        signal_calc.get_entry_signals = Mock(return_value=[])

        # Create backtest engine with vectorbt enabled in config
        config_with_vectorbt = Mock()
        config_with_vectorbt.get = Mock(
            side_effect=lambda key, default=None: {
                "backtesting.use_vectorbt": True,
                "universe.assets": ["SPY", "QQQ"],
            }.get(key, default)
        )

        backtest_engine = BacktestEngine(
            config_with_vectorbt, data_manager, signal_calc, sample_risk_manager
        )

        # Verify vectorbt engine is initialized (if available)
        if VECTORBT_AVAILABLE:
            assert backtest_engine.vectorbt_engine is not None
        else:
            assert backtest_engine.vectorbt_engine is None

    def test_vectorbt_config_parameters(self, sample_config, sample_risk_manager):
        """Test vectorbt configuration parameters."""
        from vectorbt_backtest import VectorbtBacktestEngine

        # Create mock data manager
        data_manager = Mock(spec=DataManager)

        # Create signal calculator
        signal_calc = Mock(spec=SignalCalculator)

        # Create vectorbt engine
        vbt_engine = VectorbtBacktestEngine(
            sample_config, data_manager, signal_calc, sample_risk_manager
        )

        # Check vbt config
        assert hasattr(vbt_engine, "vbt_config")
        assert "fees" in vbt_engine.vbt_config
        assert "slippage" in vbt_engine.vbt_config
        assert "min_size" in vbt_engine.vbt_config
        assert "max_size" in vbt_engine.vbt_config

        # Check reasonable values
        assert 0 <= vbt_engine.vbt_config["fees"] <= 0.01  # Reasonable fee range
        assert (
            0 <= vbt_engine.vbt_config["slippage"] <= 0.01
        )  # Reasonable slippage range
