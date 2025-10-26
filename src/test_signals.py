"""
Unit tests for signals.py - CRI and Panic Score calculations.
"""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest


@pytest.mark.unit
class TestSignalCalculator:
    """Test SignalCalculator core functionality."""

    def test_calculate_cri_perfect_correlation(
        self, sample_signal_calculator, sample_price_data, sample_trends_data
    ):
        """Test CRI calculation with perfect correlation."""
        # Create perfectly correlated data - price and trends with same % changes (need >= 10 points)
        base_price = 100.0
        price_values = []
        trends_values = []
        for i in range(
            12
        ):  # Create 12 data points to ensure we have enough after pct_change
            change = (i + 1) * 0.01  # 1%, 2%, 3%, etc.
            price_values.append(base_price * (1 + change))
            trends_values.append(base_price * (1 + change))  # Perfect correlation

        price_values = np.array(price_values)
        trends_values = np.array(trends_values)

        # Mock the align_data_by_date method
        with patch.object(
            sample_signal_calculator, "_align_data_by_date"
        ) as mock_align:
            aligned_data = pd.DataFrame(
                {"price": price_values, "trends": trends_values}
            )
            mock_align.return_value = aligned_data

            cri = sample_signal_calculator.calculate_cri(
                "SPY", sample_price_data, sample_trends_data
            )

            # Should return high CRI value (close to 1.0 in absolute value)
            assert isinstance(cri, float)
            assert 0.5 <= abs(cri) <= 1.0

    def test_calculate_cri_no_correlation(self, sample_signal_calculator):
        """Test CRI calculation with no correlation."""
        # Create uncorrelated data
        price_data = pd.DataFrame({"Close": [100.0, 110.0, 105.0, 115.0, 120.0]})
        trends_data = pd.DataFrame({"value": [50, 60, 55, 65, 70]})

        with patch.object(
            sample_signal_calculator, "_align_data_by_date"
        ) as mock_align:
            aligned_data = pd.DataFrame(
                {
                    "price": [100.0, 110.0, 105.0, 115.0, 120.0],
                    "trends": [50, 60, 55, 65, 70],
                }
            )
            mock_align.return_value = aligned_data

            cri = sample_signal_calculator.calculate_cri("SPY", price_data, trends_data)

            assert isinstance(cri, float)
            assert abs(cri) <= 1.0

    def test_calculate_cri_insufficient_data(self, sample_signal_calculator):
        """Test CRI calculation with insufficient data."""
        # Data with less than 10 points
        price_data = pd.DataFrame({"Close": [100.0, 101.0]})
        trends_data = pd.DataFrame({"value": [50, 51]})

        cri = sample_signal_calculator.calculate_cri("SPY", price_data, trends_data)
        assert cri == 0.0

    def test_calculate_cri_empty_data(self, sample_signal_calculator):
        """Test CRI calculation with empty data."""
        empty_price = pd.DataFrame()
        empty_trends = pd.DataFrame()

        cri = sample_signal_calculator.calculate_cri("SPY", empty_price, empty_trends)
        assert cri == 0.0

    def test_calculate_panic_score_normal_conditions(
        self, sample_signal_calculator, sample_price_data, sample_trends_data
    ):
        """Test panic score calculation under normal market conditions."""
        panic_score = sample_signal_calculator.calculate_panic_score(
            "SPY", sample_price_data, sample_trends_data
        )

        assert isinstance(panic_score, float)
        assert panic_score >= 0.0  # Panic score should be non-negative

    def test_calculate_panic_score_high_volatility(self, sample_signal_calculator):
        """Test panic score with high volatility data."""
        # Create high volatility price data
        high_vol_price = pd.DataFrame(
            {
                "Open": [100.0, 120.0, 80.0, 150.0, 50.0],
                "High": [130.0, 140.0, 90.0, 160.0, 70.0],
                "Low": [90.0, 100.0, 70.0, 120.0, 30.0],
                "Close": [120.0, 80.0, 150.0, 50.0, 100.0],
                "Volume": [5000000, 6000000, 7000000, 8000000, 9000000],
            },
            index=pd.date_range("2023-01-01", periods=5),
        )

        trends_data = pd.DataFrame(
            {"value": [80, 85, 90, 95, 100]},
            index=pd.date_range("2023-01-01", periods=5),
        )

        panic_score = sample_signal_calculator.calculate_panic_score(
            "SPY", high_vol_price, trends_data
        )

        assert isinstance(panic_score, float)
        assert panic_score >= 0.0

    def test_calculate_panic_score_empty_data(self, sample_signal_calculator):
        """Test panic score calculation with empty data."""
        empty_data = pd.DataFrame()

        panic_score = sample_signal_calculator.calculate_panic_score(
            "SPY", empty_data, empty_data
        )
        assert panic_score == 0.0

    def test_calculate_g_score(self, sample_signal_calculator, mock_data_manager):
        """Test G-Score calculation."""
        # Mock macro data
        mock_data_manager.get_macro_indicator.side_effect = (
            lambda indicator, **kwargs: {
                "VIX": pd.DataFrame(
                    {"value": [15.0, 16.0, 14.0]},
                    index=pd.date_range("2023-01-01", periods=3),
                ),
                "SPY": pd.DataFrame(
                    {"value": [400.0, 405.0, 410.0]},
                    index=pd.date_range("2023-01-01", periods=3),
                ),
                "USO": pd.DataFrame(
                    {"value": [70.0, 68.0, 72.0]},
                    index=pd.date_range("2023-01-01", periods=3),
                ),
            }.get(indicator, pd.DataFrame())
        )

        g_score = sample_signal_calculator.calculate_g_score()

        assert isinstance(g_score, float)
        assert g_score >= 0.0

    def test_get_tradable_universe(self, sample_signal_calculator, mock_data_manager):
        """Test tradable universe filtering."""
        # Mock data for universe assets
        mock_data_manager.get_price_data.side_effect = (
            lambda symbol, **kwargs: pd.DataFrame(
                {
                    "Open": [100.0, 101.0, 102.0],
                    "High": [105.0, 106.0, 107.0],
                    "Low": [95.0, 96.0, 97.0],
                    "Close": [103.0, 104.0, 105.0],
                    "Volume": [1000000, 1100000, 1200000],
                },
                index=pd.date_range("2023-01-01", periods=3),
            )
        )

        mock_data_manager.get_google_trends.side_effect = (
            lambda keyword, **kwargs: pd.DataFrame(
                {"value": [75, 78, 80]}, index=pd.date_range("2023-01-01", periods=3)
            )
        )

        universe = ["SPY", "QQQ", "AAPL"]
        tradable = sample_signal_calculator.get_tradable_universe(universe)

        assert isinstance(tradable, list)
        assert all(symbol in universe for symbol in tradable)

    def test_get_entry_signals(self, sample_signal_calculator, mock_data_manager):
        """Test entry signal generation."""
        # Mock data for tradable assets
        mock_data_manager.get_price_data.side_effect = (
            lambda symbol, **kwargs: pd.DataFrame(
                {
                    "Open": [100.0, 101.0, 102.0, 103.0, 104.0],
                    "High": [105.0, 106.0, 107.0, 108.0, 109.0],
                    "Low": [95.0, 96.0, 97.0, 98.0, 99.0],
                    "Close": [103.0, 104.0, 105.0, 106.0, 107.0],
                    "Volume": [1000000, 1100000, 1200000, 1300000, 1400000],
                },
                index=pd.date_range("2023-01-01", periods=5),
            )
        )

        mock_data_manager.get_google_trends.side_effect = (
            lambda keyword, **kwargs: pd.DataFrame(
                {"value": [75, 78, 80, 82, 85]},
                index=pd.date_range("2023-01-01", periods=5),
            )
        )

        tradable_assets = ["SPY", "QQQ"]
        signals = sample_signal_calculator.get_entry_signals(tradable_assets)

        assert isinstance(signals, list)
        if signals:  # If signals were generated
            for signal in signals:
                assert isinstance(signal, dict)
                assert "symbol" in signal
                assert "side" in signal
                assert signal["symbol"] in tradable_assets

    def test_validate_symbol_valid(self, sample_signal_calculator):
        """Test symbol validation with valid symbols."""
        # Should not raise exception
        sample_signal_calculator._validate_symbol("SPY")
        sample_signal_calculator._validate_symbol("AAPL")
        sample_signal_calculator._validate_symbol("BTC-USD")

    def test_validate_symbol_invalid(self, sample_signal_calculator):
        """Test symbol validation with invalid symbols."""
        with pytest.raises(ValueError, match="Symbol must be uppercase"):
            sample_signal_calculator._validate_symbol("spy")

        with pytest.raises(ValueError, match="Symbol cannot be empty"):
            sample_signal_calculator._validate_symbol("")

        with pytest.raises(ValueError, match="Symbol too long"):
            sample_signal_calculator._validate_symbol("A" * 11)

        with pytest.raises(ValueError, match="contains invalid characters"):
            sample_signal_calculator._validate_symbol("SPY@123")

    def test_get_tradable_universe_invalid_input(self, sample_signal_calculator):
        """Test tradable universe with invalid input."""
        with pytest.raises(ValueError, match="Universe must be a list"):
            sample_signal_calculator.get_tradable_universe("not_a_list")

        with pytest.raises(ValueError, match="Symbol must be uppercase"):
            sample_signal_calculator.get_tradable_universe(["spy"])

    def test_get_entry_signals_invalid_input(self, sample_signal_calculator):
        """Test entry signals with invalid input."""
        with pytest.raises(ValueError, match="Tradable assets must be a list"):
            sample_signal_calculator.get_entry_signals("not_a_list")

        with pytest.raises(ValueError, match="Symbol must be uppercase"):
            sample_signal_calculator.get_entry_signals(["spy"])

    def test_bayesian_state_machine_integration(
        self, sample_signal_calculator, mock_data_manager
    ):
        """Test Bayesian State Machine integration enhances panic scores."""
        # Mock Bayesian State Machine with high conviction
        mock_bsm = Mock()
        mock_bsm.prepare_features.return_value = np.array([[2.0, 1.5, 2.5, 2, -0.02]])
        mock_bsm.assess_conviction.return_value = {
            "conviction_level": "high",
            "should_trade": True,
            "confidence": 0.9,
            "state": "panic",
            "method": "hmm",
        }
        sample_signal_calculator.bayesian_state_machine = mock_bsm

        # Mock data for high panic scenario
        mock_data_manager.get_price_data.side_effect = (
            lambda symbol, **kwargs: pd.DataFrame(
                {
                    "Open": [100.0, 101.0, 102.0, 103.0, 104.0] * 20,  # 100 data points
                    "High": [105.0, 106.0, 107.0, 108.0, 109.0] * 20,
                    "Low": [95.0, 96.0, 97.0, 98.0, 99.0] * 20,
                    "Close": [103.0, 104.0, 105.0, 106.0, 107.0] * 20,
                    "Volume": [1000000, 1100000, 1200000, 1300000, 1400000] * 20,
                },
                index=pd.date_range("2023-01-01", periods=100),
            )
        )

        mock_data_manager.get_google_trends.side_effect = (
            lambda keyword, **kwargs: pd.DataFrame(
                {"value": [75, 78, 80, 82, 85] * 20},
                index=pd.date_range("2023-01-01", periods=100),
            )
        )

        tradable_assets = ["SPY"]
        signals = sample_signal_calculator.get_entry_signals(tradable_assets)

        # Should generate signal due to high Bayesian conviction
        assert len(signals) == 1
        signal = signals[0]
        assert signal["conviction_level"] == "high"
        assert signal["confidence"] == 0.9
        assert "enhanced_panic_score" in signal
        assert "panic_score" in signal
        assert signal["assessment_method"] == "hmm"

    def test_bayesian_state_machine_low_conviction(
        self, sample_signal_calculator, mock_data_manager
    ):
        """Test Bayesian State Machine with low conviction doesn't generate signals."""
        # Mock Bayesian State Machine with low conviction
        mock_bsm = Mock()
        mock_bsm.prepare_features.return_value = np.array([[0.1, 0.0, 0.2, 0, 0.01]])
        mock_bsm.assess_conviction.return_value = {
            "conviction_level": "low",
            "should_trade": False,
            "confidence": 0.2,
            "state": "calm",
            "method": "hmm",
        }
        sample_signal_calculator.bayesian_state_machine = mock_bsm

        # Mock data for calm market
        mock_data_manager.get_price_data.side_effect = (
            lambda symbol, **kwargs: pd.DataFrame(
                {
                    "Open": [100.0, 101.0, 102.0, 103.0, 104.0] * 20,
                    "High": [105.0, 106.0, 107.0, 108.0, 109.0] * 20,
                    "Low": [95.0, 96.0, 97.0, 98.0, 99.0] * 20,
                    "Close": [103.0, 104.0, 105.0, 106.0, 107.0] * 20,
                    "Volume": [1000000, 1100000, 1200000, 1300000, 1400000] * 20,
                },
                index=pd.date_range("2023-01-01", periods=100),
            )
        )

        mock_data_manager.get_google_trends.side_effect = (
            lambda keyword, **kwargs: pd.DataFrame(
                {"value": [75, 78, 80, 82, 85] * 20},
                index=pd.date_range("2023-01-01", periods=100),
            )
        )

        tradable_assets = ["SPY"]
        signals = sample_signal_calculator.get_entry_signals(tradable_assets)

        # Should not generate signals due to low conviction (enhanced score below threshold)
        assert len(signals) == 0

    def test_bayesian_state_machine_fallback(
        self, sample_signal_calculator, mock_data_manager
    ):
        """Test fallback to rule-based logic when Bayesian State Machine fails."""
        # Mock Bayesian State Machine to raise exception
        mock_bsm = Mock()
        mock_bsm.prepare_features.side_effect = Exception("HMM error")
        sample_signal_calculator.bayesian_state_machine = mock_bsm

        # Mock data that would normally generate a signal (high panic score)
        mock_data_manager.get_price_data.side_effect = (
            lambda symbol, **kwargs: pd.DataFrame(
                {
                    "Open": [100.0, 101.0, 102.0, 103.0, 104.0] * 20,  # 100 data points
                    "High": [105.0, 106.0, 107.0, 108.0, 109.0] * 20,
                    "Low": [95.0, 96.0, 97.0, 98.0, 99.0] * 20,
                    "Close": [103.0, 104.0, 105.0, 106.0, 107.0] * 20,
                    "Volume": [1000000, 1100000, 1200000, 1300000, 1400000] * 20,
                },
                index=pd.date_range("2023-01-01", periods=100),
            )
        )

        mock_data_manager.get_google_trends.side_effect = (
            lambda keyword, **kwargs: pd.DataFrame(
                {"value": [75, 78, 80, 82, 85] * 20},
                index=pd.date_range("2023-01-01", periods=100),
            )
        )

        tradable_assets = ["SPY"]
        signals = sample_signal_calculator.get_entry_signals(tradable_assets)

        # Should generate signal using rule-based fallback
        assert len(signals) == 1
        signal = signals[0]
        assert signal["assessment_method"] == "rules"
        assert "enhanced_panic_score" in signal
        assert (
            signal["enhanced_panic_score"] == signal["panic_score"]
        )  # No enhancement applied
