"""
Integration tests for Harvester II system components.
Tests API integrations with mocked responses.
"""

import pytest
import pandas as pd
import responses
import yfinance as yf
from unittest.mock import patch, Mock
from pathlib import Path
import sys

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_manager import DataManager
from signals import SignalCalculator
from engine import TradingEngine
from di import create_components


@pytest.mark.integration
class TestDataManagerIntegration:
    """Test DataManager integration with external APIs."""

    @responses.activate
    def test_get_price_data_yfinance_success(self, sample_config):
        """Test successful price data retrieval from Yahoo Finance."""
        # Mock yfinance response
        with patch('yfinance.Ticker') as mock_ticker:
            mock_instance = Mock()
            mock_ticker.return_value = mock_instance

            # Create sample price data
            price_data = pd.DataFrame({
                'Open': [100.0, 101.0, 102.0],
                'High': [105.0, 106.0, 107.0],
                'Low': [95.0, 96.0, 97.0],
                'Close': [103.0, 104.0, 105.0],
                'Volume': [1000000, 1100000, 1200000]
            }, index=pd.date_range('2023-01-01', periods=3))

            mock_instance.history.return_value = price_data

            data_manager = DataManager(sample_config)
            result = data_manager.get_price_data('SPY', period='3d')

            assert not result.empty
            assert len(result) == 3
            assert all(col in result.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])

    @responses.activate
    def test_get_price_data_yfinance_failure(self, sample_config):
        """Test price data retrieval failure handling."""
        with patch('yfinance.Ticker') as mock_ticker:
            mock_instance = Mock()
            mock_ticker.return_value = mock_instance
            mock_instance.history.return_value = pd.DataFrame()  # Empty response

            data_manager = DataManager(sample_config)
            result = data_manager.get_price_data('INVALID', period='3d')

            assert result.empty

    def test_get_google_trends_success(self, sample_config):
        """Test successful Google Trends data retrieval."""
        with patch('pytrends.request.TrendReq') as mock_trend_req, \
             patch('pytrends.dailydata.get_daily_data') as mock_daily_data:

            # Mock the TrendReq instance
            mock_instance = Mock()
            mock_trend_req.return_value = mock_instance

            # Mock trends data response
            trends_data = pd.DataFrame({
                'SPY': [75, 78, 80, 82, 85],
                'isPartial': [False, False, False, False, False]
            }, index=pd.date_range('2023-01-01', periods=5))

            mock_instance.interest_over_time.return_value = trends_data
            mock_daily_data.return_value = trends_data

            data_manager = DataManager(sample_config)
            result = data_manager.get_google_trends('SPY', timeframe='today 5-d')

            assert not result.empty
            assert 'value' in result.columns
            assert len(result) == 5

    @responses.activate
    def test_get_google_trends_failure(self, sample_config):
        """Test Google Trends failure handling."""
        with patch('pytrends.request.TrendReq') as mock_trend_req:
            mock_instance = Mock()
            mock_trend_req.return_value = mock_instance
            mock_instance.interest_over_time.return_value = pd.DataFrame()  # Empty response

            data_manager = DataManager(sample_config)
            result = data_manager.get_google_trends('SPY', timeframe='today 5-d')

            assert result.empty

    @responses.activate
    def test_get_macro_indicator_vix_success(self, sample_config):
        """Test VIX macro indicator retrieval."""
        with patch('yfinance.Ticker') as mock_ticker:
            mock_instance = Mock()
            mock_ticker.return_value = mock_instance

            # Mock VIX data
            vix_data = pd.DataFrame({
                'Close': [15.5, 16.2, 14.8, 17.1, 15.9]
            }, index=pd.date_range('2023-01-01', periods=5))

            mock_instance.history.return_value = vix_data

            data_manager = DataManager(sample_config)
            result = data_manager.get_macro_indicator('VIX')

            assert not result.empty
            assert 'value' in result.columns
            assert len(result) == 5

    def test_get_universe_data_multiple_symbols(self, sample_config, mock_data_manager):
        """Test retrieving data for multiple universe symbols."""
        symbols = ['SPY', 'QQQ', 'AAPL']

        # Mock successful data retrieval for all symbols
        def mock_get_price_data(symbol, **kwargs):
            return pd.DataFrame({
                'Open': [100.0, 101.0],
                'High': [105.0, 106.0],
                'Low': [95.0, 96.0],
                'Close': [103.0, 104.0],
                'Volume': [1000000, 1100000]
            }, index=pd.date_range('2023-01-01', periods=2))

        mock_data_manager.get_price_data.side_effect = mock_get_price_data

        result = mock_data_manager.get_universe_data(symbols, period='2d')

        assert len(result) == 3
        assert all(symbol in result for symbol in symbols)
        assert all(not df.empty for df in result.values())


@pytest.mark.integration
class TestSignalCalculatorIntegration:
    """Test SignalCalculator integration with DataManager."""

    def test_signal_calculator_with_data_manager(self, sample_config, mock_data_manager):
        """Test SignalCalculator working with DataManager."""
        signal_calc = SignalCalculator(sample_config, mock_data_manager)

        # Test that signal calculator can get data from data manager
        cri = signal_calc.calculate_cri('SPY', pd.DataFrame(), pd.DataFrame())
        assert isinstance(cri, float)

        panic_score = signal_calc.calculate_panic_score('SPY', pd.DataFrame(), pd.DataFrame())
        assert isinstance(panic_score, float)

    def test_complete_signal_workflow(self, sample_config, mock_data_manager):
        """Test complete signal generation workflow."""
        signal_calc = SignalCalculator(sample_config, mock_data_manager)

        # Mock data retrieval methods
        mock_data_manager.get_price_data.return_value = pd.DataFrame({
            'Open': [100.0] * 20,
            'High': [105.0] * 20,
            'Low': [95.0] * 20,
            'Close': [103.0] * 20,
            'Volume': [1000000] * 20
        }, index=pd.date_range('2023-01-01', periods=20))

        mock_data_manager.get_google_trends.return_value = pd.DataFrame({
            'value': [75] * 20
        }, index=pd.date_range('2023-01-01', periods=20))

        # Test tradable universe generation
        universe = ['SPY', 'QQQ']
        tradable = signal_calc.get_tradable_universe(universe)

        assert isinstance(tradable, list)

        # Test entry signal generation
        signals = signal_calc.get_entry_signals(tradable)

        assert isinstance(signals, list)
        if signals:
            for signal in signals:
                assert 'symbol' in signal
                assert 'side' in signal


@pytest.mark.integration
class TestTradingEngineIntegration:
    """Test TradingEngine integration with all components."""

    def test_trading_engine_initialization(self, sample_components):
        """Test that TradingEngine can be initialized with all dependencies."""
        config = sample_components['config']
        data_manager = sample_components['data_manager']
        signal_calc = sample_components['signal_calculator']
        risk_manager = sample_components['risk_manager']
        portfolio_manager = sample_components['portfolio_manager']

        engine = TradingEngine(config, data_manager, signal_calc, risk_manager, portfolio_manager)

        assert engine.config == config
        assert engine.data_manager == data_manager
        assert engine.signal_calculator == signal_calc
        assert engine.risk_manager == risk_manager
        assert engine.portfolio_manager == portfolio_manager

    def test_system_initialization_workflow(self, sample_components):
        """Test the complete system initialization workflow."""
        config = sample_components['config']
        data_manager = sample_components['data_manager']
        signal_calc = sample_components['signal_calculator']
        risk_manager = sample_components['risk_manager']
        portfolio_manager = sample_components['portfolio_manager']

        engine = TradingEngine(config, data_manager, signal_calc, risk_manager, portfolio_manager)

        # Mock successful health checks
        data_manager.get_price_data.return_value = pd.DataFrame({'Close': [100.0]})
        data_manager.get_google_trends.return_value = pd.DataFrame({'value': [75]})
        data_manager.get_macro_indicator.return_value = pd.DataFrame({'value': [15.0]})

        # Mock risk manager
        risk_manager.can_open_new_position.return_value = True

        # Test initialization
        success = engine.initialize_system()

        assert success is True

    def test_trading_loop_execution(self, sample_components):
        """Test trading loop execution with mocked data."""
        config = sample_components['config']
        data_manager = sample_components['data_manager']
        signal_calc = sample_components['signal_calculator']
        risk_manager = sample_components['risk_manager']
        portfolio_manager = sample_components['portfolio_manager']

        engine = TradingEngine(config, data_manager, signal_calc, risk_manager, portfolio_manager)

        # Mock all the data and methods needed for trading loop
        data_manager.get_price_data.return_value = pd.DataFrame({
            'Close': [100.0, 101.0, 102.0]
        }, index=pd.date_range('2023-01-01', periods=3))

        signal_calc.calculate_g_score.return_value = 1.5
        signal_calc.get_entry_signals.return_value = []

        risk_manager.check_drawdown_limit.return_value = False
        risk_manager.can_open_new_position.return_value = True

        portfolio_manager.process_exit_signals.return_value = 0

        # Mock datetime to avoid universe update
        with patch('engine.datetime') as mock_datetime:
            mock_datetime.now.return_value = engine.last_universe_update or mock_datetime.now()

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
    with patch('yfinance.Ticker') as mock_ticker:
        mock_yf_instance = Mock()
        mock_ticker.return_value = mock_yf_instance

        price_data = pd.DataFrame({
            'Open': [100.0, 101.0, 102.0, 103.0, 104.0],
            'High': [105.0, 106.0, 107.0, 108.0, 109.0],
            'Low': [95.0, 96.0, 97.0, 98.0, 99.0],
            'Close': [103.0, 104.0, 105.0, 106.0, 107.0],
            'Volume': [1000000, 1100000, 1200000, 1300000, 1400000]
        }, index=pd.date_range('2023-01-01', periods=5))

        mock_yf_instance.history.return_value = price_data

        # Mock Google Trends
        with patch('pytrends.request.TrendReq') as mock_trend_req, \
             patch('pytrends.dailydata.get_daily_data') as mock_daily_data:
            mock_trends_instance = Mock()
            mock_trend_req.return_value = mock_trends_instance

            trends_data = pd.DataFrame({
                'SPY': [75, 78, 80, 82, 85],
                'isPartial': [False, False, False, False, False]
            }, index=pd.date_range('2023-01-01', periods=5))

            mock_trends_instance.interest_over_time.return_value = trends_data
            mock_daily_data.return_value = trends_data

            # Create components
            data_manager = DataManager(sample_config)
            signal_calc = SignalCalculator(sample_config, data_manager)

            # Test data retrieval
            price_result = data_manager.get_price_data('SPY', period='5d')
            trends_result = data_manager.get_google_trends('SPY', timeframe='today 5-d')

    assert not price_result.empty
    assert not trends_result.empty

    # Test signal calculation
    cri = signal_calc.calculate_cri('SPY', price_result, trends_result)
    panic_score = signal_calc.calculate_panic_score('SPY', price_result, trends_result)

    assert isinstance(cri, float)
    assert isinstance(panic_score, float)
    assert 0.0 <= abs(cri) <= 1.0
    assert panic_score >= 0.0
