"""
Integration tests for Harvester II system components.
Tests API integrations with mocked responses.
"""

from pathlib import Path
import sys
from unittest.mock import Mock, patch

import pandas as pd
import pytest
import responses

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_manager import DataManager
from engine import TradingEngine
from signals import SignalCalculator


@pytest.mark.integration
class TestDataManagerIntegration:
    """Test DataManager integration with external APIs."""

    @responses.activate
    def test_get_price_data_yfinance_success(self, sample_config):
        """Test successful price data retrieval from Yahoo Finance."""
        # Mock yfinance response
        with patch("yfinance.Ticker") as mock_ticker:
            mock_instance = Mock()
            mock_ticker.return_value = mock_instance

            # Create sample price data
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

            mock_instance.history.return_value = price_data

            data_manager = DataManager(sample_config)
            result = data_manager.get_price_data("SPY", period="3d")

            assert not result.empty
            assert len(result) == 3
            assert all(
                col in result.columns
                for col in ["Open", "High", "Low", "Close", "Volume"]
            )

    @responses.activate
    def test_get_price_data_yfinance_failure(self, sample_config):
        """Test price data retrieval failure handling."""
        with patch("yfinance.Ticker") as mock_ticker:
            mock_instance = Mock()
            mock_ticker.return_value = mock_instance
            mock_instance.history.return_value = pd.DataFrame()  # Empty response

            data_manager = DataManager(sample_config)
            result = data_manager.get_price_data("INVALID", period="3d")

            assert result.empty

    def test_get_google_trends_success(self, sample_config):
        """Test successful Google Trends data retrieval."""
        with (
            patch("pytrends.request.TrendReq") as mock_trend_req,
            patch("pytrends.dailydata.get_daily_data") as mock_daily_data,
        ):
            # Mock the TrendReq instance
            mock_instance = Mock()
            mock_trend_req.return_value = mock_instance

            # Mock trends data response
            trends_data = pd.DataFrame(
                {
                    "SPY": [75, 78, 80, 82, 85],
                    "isPartial": [False, False, False, False, False],
                },
                index=pd.date_range("2023-01-01", periods=5),
            )

            mock_instance.interest_over_time.return_value = trends_data
            mock_daily_data.return_value = trends_data

            data_manager = DataManager(sample_config)
            result = data_manager.get_google_trends("SPY", timeframe="today 5-d")

            assert not result.empty
            assert "value" in result.columns
            assert len(result) == 5

    @responses.activate
    def test_get_google_trends_failure(self, sample_config):
        """Test Google Trends failure handling."""
        with patch("pytrends.request.TrendReq") as mock_trend_req:
            mock_instance = Mock()
            mock_trend_req.return_value = mock_instance
            mock_instance.interest_over_time.return_value = (
                pd.DataFrame()
            )  # Empty response

            data_manager = DataManager(sample_config)
            result = data_manager.get_google_trends("SPY", timeframe="today 5-d")

            assert result.empty

    @responses.activate
    def test_get_macro_indicator_vix_success(self, sample_config):
        """Test VIX macro indicator retrieval."""
        with patch("yfinance.Ticker") as mock_ticker:
            mock_instance = Mock()
            mock_ticker.return_value = mock_instance

            # Mock VIX data
            vix_data = pd.DataFrame(
                {"Close": [15.5, 16.2, 14.8, 17.1, 15.9]},
                index=pd.date_range("2023-01-01", periods=5),
            )

            mock_instance.history.return_value = vix_data

            data_manager = DataManager(sample_config)
            result = data_manager.get_macro_indicator("VIX")

            assert not result.empty
            assert "value" in result.columns
            assert len(result) == 5

    def test_get_universe_data_multiple_symbols(self, sample_config, mock_data_manager):
        """Test retrieving data for multiple universe symbols."""
        symbols = ["SPY", "QQQ", "AAPL"]

        # Mock successful data retrieval for all symbols
        def mock_get_price_data(symbol, **kwargs):
            return pd.DataFrame(
                {
                    "Open": [100.0, 101.0],
                    "High": [105.0, 106.0],
                    "Low": [95.0, 96.0],
                    "Close": [103.0, 104.0],
                    "Volume": [1000000, 1100000],
                },
                index=pd.date_range("2023-01-01", periods=2),
            )

        mock_data_manager.get_price_data.side_effect = mock_get_price_data

        result = mock_data_manager.get_universe_data(symbols, period="2d")

        assert len(result) == 3
        assert all(symbol in result for symbol in symbols)
        assert all(not df.empty for df in result.values())


@pytest.mark.integration
class TestSignalCalculatorIntegration:
    """Test SignalCalculator integration with DataManager."""

    def test_signal_calculator_with_data_manager(
        self, sample_config, mock_data_manager
    ):
        """Test SignalCalculator working with DataManager."""
        signal_calc = SignalCalculator(sample_config, mock_data_manager)

        # Test that signal calculator can get data from data manager
        cri = signal_calc.calculate_cri("SPY", pd.DataFrame(), pd.DataFrame())
        assert isinstance(cri, float)

        panic_score = signal_calc.calculate_panic_score(
            "SPY", pd.DataFrame(), pd.DataFrame()
        )
        assert isinstance(panic_score, float)

    def test_complete_signal_workflow(self, sample_config, mock_data_manager):
        """Test complete signal generation workflow."""
        signal_calc = SignalCalculator(sample_config, mock_data_manager)

        # Mock data retrieval methods
        mock_data_manager.get_price_data.return_value = pd.DataFrame(
            {
                "Open": [100.0] * 20,
                "High": [105.0] * 20,
                "Low": [95.0] * 20,
                "Close": [103.0] * 20,
                "Volume": [1000000] * 20,
            },
            index=pd.date_range("2023-01-01", periods=20),
        )

        mock_data_manager.get_google_trends.return_value = pd.DataFrame(
            {"value": [75] * 20}, index=pd.date_range("2023-01-01", periods=20)
        )

        # Test tradable universe generation
        universe = ["SPY", "QQQ"]
        tradable = signal_calc.get_tradable_universe(universe)

        assert isinstance(tradable, list)

        # Test entry signal generation
        signals = signal_calc.get_entry_signals(tradable)

        assert isinstance(signals, list)
        if signals:
            for signal in signals:
                assert "symbol" in signal
                assert "side" in signal


@pytest.mark.integration
class TestTradingEngineIntegration:
    """Test TradingEngine integration with all components."""

    def test_trading_engine_initialization(self, sample_components):
        """Test that TradingEngine can be initialized with all dependencies."""
        config = sample_components["config"]
        data_manager = sample_components["data_manager"]
        signal_calc = sample_components["signal_calculator"]
        risk_manager = sample_components["risk_manager"]
        portfolio_manager = sample_components["portfolio_manager"]

        engine = TradingEngine(
            config, data_manager, signal_calc, risk_manager, portfolio_manager
        )

        assert engine.config == config
        assert engine.data_manager == data_manager
        assert engine.signal_calculator == signal_calc
        assert engine.risk_manager == risk_manager
        assert engine.portfolio_manager == portfolio_manager

    def test_system_initialization_workflow(self, sample_components):
        """Test the complete system initialization workflow."""
        config = sample_components["config"]
        data_manager = sample_components["data_manager"]
        signal_calc = sample_components["signal_calculator"]
        risk_manager = sample_components["risk_manager"]
        portfolio_manager = sample_components["portfolio_manager"]

        engine = TradingEngine(
            config, data_manager, signal_calc, risk_manager, portfolio_manager
        )

        # Mock successful health checks
        data_manager.get_price_data.return_value = pd.DataFrame({"Close": [100.0]})
        data_manager.get_google_trends.return_value = pd.DataFrame({"value": [75]})
        data_manager.get_macro_indicator.return_value = pd.DataFrame({"value": [15.0]})

        # Mock risk manager
        risk_manager.can_open_new_position.return_value = True

        # Test initialization
        success = engine.initialize_system()

        assert success is True

    def test_trading_loop_execution(self, sample_components):
        """Test trading loop execution with mocked data."""
        config = sample_components["config"]
        data_manager = sample_components["data_manager"]
        signal_calc = sample_components["signal_calculator"]
        risk_manager = sample_components["risk_manager"]
        portfolio_manager = sample_components["portfolio_manager"]

        engine = TradingEngine(
            config, data_manager, signal_calc, risk_manager, portfolio_manager
        )

        # Mock all the data and methods needed for trading loop
        data_manager.get_price_data.return_value = pd.DataFrame(
            {"Close": [100.0, 101.0, 102.0]},
            index=pd.date_range("2023-01-01", periods=3),
        )

        signal_calc.calculate_g_score.return_value = 1.5
        signal_calc.get_entry_signals.return_value = []

        risk_manager.check_drawdown_limit.return_value = False
        risk_manager.can_open_new_position.return_value = True

        portfolio_manager.process_exit_signals.return_value = 0

        # Mock datetime to avoid universe update
        with patch("engine.datetime") as mock_datetime:
            mock_datetime.now.return_value = (
                engine.last_universe_update or mock_datetime.now()
            )

            # Execute trading loop
            engine.run_daily_trading_loop()

            # Verify methods were called
            assert signal_calc.calculate_g_score.called
            assert signal_calc.get_entry_signals.called


@pytest.mark.integration
@responses.activate
def test_end_to_end_data_flow(sample_config):
    """Test end-to-end data flow from APIs to signals."""
    # Mock yfinance
    with patch("yfinance.Ticker") as mock_ticker:
        mock_yf_instance = Mock()
        mock_ticker.return_value = mock_yf_instance

        price_data = pd.DataFrame(
            {
                "Open": [100.0, 101.0, 102.0, 103.0, 104.0],
                "High": [105.0, 106.0, 107.0, 108.0, 109.0],
                "Low": [95.0, 96.0, 97.0, 98.0, 99.0],
                "Close": [103.0, 104.0, 105.0, 106.0, 107.0],
                "Volume": [1000000, 1100000, 1200000, 1300000, 1400000],
            },
            index=pd.date_range("2023-01-01", periods=5),
        )

        mock_yf_instance.history.return_value = price_data

        # Mock Google Trends
        with (
            patch("pytrends.request.TrendReq") as mock_trend_req,
            patch("pytrends.dailydata.get_daily_data") as mock_daily_data,
        ):
            mock_trends_instance = Mock()
            mock_trend_req.return_value = mock_trends_instance

            trends_data = pd.DataFrame(
                {
                    "SPY": [75, 78, 80, 82, 85],
                    "isPartial": [False, False, False, False, False],
                },
                index=pd.date_range("2023-01-01", periods=5),
            )

            mock_trends_instance.interest_over_time.return_value = trends_data
            mock_daily_data.return_value = trends_data

            # Create components
            data_manager = DataManager(sample_config)
            signal_calc = SignalCalculator(sample_config, data_manager)

            # Test data retrieval
            price_result = data_manager.get_price_data("SPY", period="5d")
            trends_result = data_manager.get_google_trends("SPY", timeframe="today 5-d")

    assert not price_result.empty
    assert not trends_result.empty

    # Test signal calculation
    cri = signal_calc.calculate_cri("SPY", price_result, trends_result)
    panic_score = signal_calc.calculate_panic_score("SPY", price_result, trends_result)

    assert isinstance(cri, float)
    assert isinstance(panic_score, float)
    assert 0.0 <= abs(cri) <= 1.0
    assert panic_score >= 0.0


@pytest.mark.integration
class TestEndToEndIntegration:
    """End-to-end integration tests for the complete trading system."""

    def create_full_system(self):
        """Create a complete trading system for end-to-end testing."""
        from di import (
            create_config,
            create_data_manager,
            create_portfolio_manager,
            create_risk_manager,
            create_signal_calculator,
            create_trading_engine,
        )

        config = create_config("config.json")
        data_manager = create_data_manager(config)
        signal_calculator = create_signal_calculator(config, data_manager)
        risk_manager = create_risk_manager(config)
        portfolio_manager = create_portfolio_manager(
            config, risk_manager, data_manager, signal_calculator
        )
        trading_engine = create_trading_engine(config)

        return {
            "config": config,
            "data_manager": data_manager,
            "signal_calculator": signal_calculator,
            "risk_manager": risk_manager,
            "portfolio_manager": portfolio_manager,
            "trading_engine": trading_engine,
        }

    def test_end_to_end_backtest_to_live_parameters(self):
        """Test that backtest-optimized parameters work in live system."""
        system = self.create_full_system()
        engine = system["trading_engine"]

        # Run backtest with default parameters
        backtest_results = engine.run_backtest("2020-01-01", "2021-01-01", 100000)

        # Validate backtest completed successfully
        assert "error" not in backtest_results
        assert "capital" in backtest_results
        assert backtest_results["capital"]["final_capital"] > 0

        # Check that system can initialize for live trading
        init_success = engine.initialize_system()
        assert init_success, "System initialization failed"

        # Verify tradable universe was updated
        assert hasattr(engine, "tradable_universe")
        assert isinstance(engine.tradable_universe, list)

        # Check system health
        health_ok = engine.check_system_health()
        assert health_ok, "System health check failed"

    def test_parameter_bridging_backtest_to_live(self):
        """Test that parameters optimized in backtest transfer to live system."""
        system = self.create_full_system()
        config = system["config"]
        engine = system["trading_engine"]

        # Get original parameters
        original_panic_threshold = config.get("signals.panic_threshold")
        original_max_positions = config.get("risk_management.max_open_positions")

        # Simulate parameter optimization (in real scenario, this would come from optimization)
        optimized_params = {
            "signals.panic_threshold": 2.5,
            "risk_management.max_open_positions": 5,
        }

        # Apply optimized parameters to config
        for key, value in optimized_params.items():
            config._config_data[key] = value

        # Verify parameters were applied
        assert config.get("signals.panic_threshold") == 2.5
        assert config.get("risk_management.max_open_positions") == 5

        # Run backtest with optimized parameters
        backtest_results = engine.run_backtest("2020-01-01", "2021-01-01", 100000)

        # Validate backtest works with optimized parameters
        assert "error" not in backtest_results
        assert "capital" in backtest_results

    def test_system_resilience_under_stress(self):
        """Test system resilience when data sources are unavailable."""
        system = self.create_full_system()
        engine = system["trading_engine"]
        data_manager = system["data_manager"]

        # Mock data source failures
        original_get_price = data_manager.get_price_data
        original_get_trends = data_manager.get_google_trends

        def failing_price_data(*args, **kwargs):
            return pd.DataFrame()  # Return empty DataFrame

        def failing_trends_data(*args, **kwargs):
            return pd.DataFrame()  # Return empty DataFrame

        data_manager.get_price_data = failing_price_data
        data_manager.get_google_trends = failing_trends_data

        try:
            # System should handle data failures gracefully
            health_ok = engine.check_system_health()
            assert not health_ok, "System should detect data source failures"

            # Backtest should fail gracefully
            backtest_results = engine.run_backtest("2020-01-01", "2021-01-01", 100000)
            assert "error" in backtest_results, "Backtest should fail with no data"

        finally:
            # Restore original methods
            data_manager.get_price_data = original_get_price
            data_manager.get_google_trends = original_get_trends

    def test_configuration_validation_end_to_end(self):
        """Test that configuration validation works end-to-end."""
        system = self.create_full_system()
        config = system["config"]

        # Test valid configuration
        is_valid = config.validate()
        assert is_valid, "Valid configuration should pass validation"

        # Test invalid configuration (simulate missing required fields)
        original_universe = config._config_data.get("universe", {})
        config._config_data["universe"] = {}  # Remove required fields

        try:
            is_valid = config.validate()
            assert not is_valid, "Invalid configuration should fail validation"
        finally:
            # Restore original configuration
            if original_universe:
                config._config_data["universe"] = original_universe

    def test_bias_detection_integration(self):
        """Test bias detection works with actual backtest results."""
        system = self.create_full_system()
        engine = system["trading_engine"]

        # Run a backtest first
        backtest_results = engine.run_backtest("2020-01-01", "2021-01-01", 100000)
        assert "error" not in backtest_results

        # Run bias detection on the results
        bias_results = engine.detect_backtest_biases(backtest_results)

        # Validate bias detection structure
        assert isinstance(bias_results, dict)
        assert "error" not in bias_results

        # Check for expected bias analysis fields
        expected_fields = [
            "look_ahead_bias",
            "survivorship_bias",
            "overfitting_analysis",
            "recommendations",
        ]
        for field in expected_fields:
            assert field in bias_results, f"Missing bias analysis field: {field}"

    def test_walk_forward_validation_end_to_end(self):
        """Test complete walk-forward validation process."""
        system = self.create_full_system()
        engine = system["trading_engine"]

        # Run walk-forward validation
        wf_results = engine.run_walk_forward_validation(
            start_date="2020-01-01",
            end_date="2022-01-01",
            train_window_months=12,
            test_window_months=3,
            step_months=3,
        )

        # Validate walk-forward results structure
        assert "error" not in wf_results
        assert "summary" in wf_results
        assert "fold_results" in wf_results

        summary = wf_results["summary"]
        assert "total_folds" in summary
        assert "overfitting_rate" in summary
        assert "recommendation" in summary

    def test_survivor_free_backtest_integration(self):
        """Test survivor-free backtest with actual data filtering."""
        system = self.create_full_system()
        engine = system["trading_engine"]

        # Run survivor-free backtest
        sf_results = engine.run_survivor_free_backtest(
            "2020-01-01", "2021-01-01", 100000
        )

        # Validate survivor-free results
        assert "error" not in sf_results
        assert "survivor_analysis" in sf_results
        assert "capital" in sf_results

        survivor_analysis = sf_results["survivor_analysis"]
        assert "original_universe_size" in survivor_analysis
        assert "survivor_universe_size" in survivor_analysis
        assert "survival_rate" in survivor_analysis

    def test_a_b_test_bayesian_enhancement(self):
        """Test A/B testing of Bayesian enhancement end-to-end."""
        system = self.create_full_system()
        engine = system["trading_engine"]

        # Run A/B test
        ab_results = engine.run_ab_test("2020-01-01", "2021-01-01", 100000)

        # Validate A/B test results structure
        assert "error" not in ab_results
        assert "test_period" in ab_results
        assert "comparison" in ab_results

        comparison = ab_results["comparison"]
        expected_metrics = [
            "sharpe_ratio_improvement",
            "total_return_improvement",
            "max_drawdown_improvement",
            "win_rate_improvement",
            "conviction_correlation",
        ]

        for metric in expected_metrics:
            assert metric in comparison, f"Missing comparison metric: {metric}"

    def test_hyperparameter_optimization_workflow(self):
        """Test complete hyperparameter optimization workflow."""
        system = self.create_full_system()
        engine = system["trading_engine"]

        # Run hyperparameter optimization (with reduced trials for testing)
        # Temporarily modify config for faster testing
        original_n_trials = engine.config.get("optimization.n_trials")
        engine.config._config_data["optimization"]["n_trials"] = 5

        try:
            opt_results = engine.run_hyperparameter_optimization(
                "2020-01-01", "2021-01-01", 100000
            )

            # Validate optimization results structure
            if "error" not in opt_results:
                assert "best_parameters" in opt_results
                assert "best_sharpe_ratio" in opt_results
                assert "optimization_trials" in opt_results
                assert "final_backtest_results" in opt_results

                best_params = opt_results["best_parameters"]
                assert isinstance(best_params, dict)
                assert len(best_params) > 0, "Should have optimized parameters"
        finally:
            # Restore original config
            if original_n_trials:
                engine.config._config_data["optimization"]["n_trials"] = (
                    original_n_trials
                )

    def test_system_monitoring_and_metrics(self):
        """Test system monitoring and metrics collection."""
        system = self.create_full_system()
        engine = system["trading_engine"]

        # Get system status
        status = engine.get_system_status()

        # Validate status structure
        assert "system_status" in status
        assert "portfolio" in status
        assert "risk_management" in status
        assert "macro_risk" in status
        assert "tradable_universe" in status

        # Get Prometheus metrics
        metrics = engine.get_metrics()

        # Should return metrics string or error message
        assert isinstance(metrics, str)
        assert len(metrics) > 0

        # Update metrics
        engine.update_metrics()  # Should not raise exception

    def test_configuration_bridging(self):
        """Test that configuration changes properly bridge between components."""
        system = self.create_full_system()
        config = system["config"]
        signal_calc = system["signal_calculator"]
        risk_manager = system["risk_manager"]

        # Test that config changes affect component behavior
        original_threshold = config.get("signals.panic_threshold")

        # Change panic threshold
        config._config_data["signals"]["panic_threshold"] = 2.0

        # Verify change is reflected
        assert config.get("signals.panic_threshold") == 2.0

        # Test risk management parameter bridging
        original_max_pos = config.get("risk_management.max_open_positions")
        config._config_data["risk_management"]["max_open_positions"] = 3

        assert config.get("risk_management.max_open_positions") == 3
