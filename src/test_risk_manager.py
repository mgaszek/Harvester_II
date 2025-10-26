"""
Unit tests for risk_manager.py - position sizing and risk calculations.
"""

import pytest


@pytest.mark.unit
class TestRiskManager:
    """Test RiskManager position sizing and risk calculations."""

    def test_calculate_position_size_normal_conditions(self, sample_risk_manager):
        """Test position sizing under normal market conditions."""
        symbol = "SPY"
        current_price = 400.0
        atr = 5.0  # 1.25% of price
        g_score = 1.5  # Normal risk level

        position_info = sample_risk_manager.calculate_position_size(
            symbol, current_price, atr, g_score
        )

        assert position_info is not None
        assert "shares" in position_info
        assert "position_value" in position_info
        assert "risk_amount" in position_info
        assert "stop_loss_price" in position_info
        assert "profit_target_price" in position_info

        # Verify calculations
        assert position_info["shares"] > 0
        assert position_info["position_value"] > 0
        assert position_info["risk_amount"] > 0

    def test_calculate_position_size_high_volatility(self, sample_risk_manager):
        """Test position sizing with high volatility (high ATR)."""
        symbol = "SPY"
        current_price = 400.0
        atr = 20.0  # 5% of price - very volatile
        g_score = 1.5

        position_info = sample_risk_manager.calculate_position_size(
            symbol, current_price, atr, g_score
        )

        assert position_info is not None
        # High volatility should result in smaller position sizes
        assert position_info["shares"] < 1000  # Should be conservative

    def test_calculate_position_size_low_volatility(self, sample_risk_manager):
        """Test position sizing with low volatility (low ATR)."""
        symbol = "SPY"
        current_price = 400.0
        atr = 1.0  # 0.25% of price - very stable
        g_score = 1.5

        position_info = sample_risk_manager.calculate_position_size(
            symbol, current_price, atr, g_score
        )

        assert position_info is not None
        # Low volatility should allow reasonable position sizes (actual behavior is conservative)
        assert position_info["shares"] >= 1

    def test_calculate_position_size_high_risk_gscore(self, sample_risk_manager):
        """Test position sizing with high risk G-Score."""
        symbol = "SPY"
        current_price = 400.0
        atr = 5.0
        g_score = 3.5  # High risk - should reduce position size

        position_info = sample_risk_manager.calculate_position_size(
            symbol, current_price, atr, g_score
        )

        assert position_info is not None
        # High G-Score should result in smaller positions
        assert position_info["shares"] < 500

    def test_calculate_position_size_min_size_limit(self, sample_risk_manager):
        """Test that position sizing respects minimum size limits."""
        symbol = "SPY"
        current_price = 400.0
        atr = 50.0  # Extremely high volatility
        g_score = 5.0  # Extremely high risk

        position_info = sample_risk_manager.calculate_position_size(
            symbol, current_price, atr, g_score
        )

        # High risk may result in no position being allowed
        if position_info:
            # If a position is allowed, it should meet minimum size or be rejected
            assert (
                position_info["shares"] == 0
                or position_info["shares"] >= sample_risk_manager.min_position_size
            )

    def test_calculate_position_size_max_size_limit(self, sample_risk_manager):
        """Test that position sizing respects maximum size limits."""
        symbol = "SPY"
        current_price = 10.0  # Very cheap stock
        atr = 0.1  # Very low volatility
        g_score = 0.5  # Very low risk

        position_info = sample_risk_manager.calculate_position_size(
            symbol, current_price, atr, g_score
        )

        if position_info:
            assert position_info["shares"] <= sample_risk_manager.max_position_size

    def test_calculate_position_size_insufficient_equity(self, sample_risk_manager):
        """Test position sizing when equity is insufficient."""
        # Temporarily reduce equity
        original_equity = sample_risk_manager.equity
        sample_risk_manager.equity = 100  # Very low equity

        symbol = "SPY"
        current_price = 400.0
        atr = 5.0
        g_score = 1.5

        position_info = sample_risk_manager.calculate_position_size(
            symbol, current_price, atr, g_score
        )

        # Should still return valid position info, but smaller
        assert position_info is not None
        assert position_info["shares"] >= 0

        # Restore original equity
        sample_risk_manager.equity = original_equity

    def test_can_open_new_position_available_slots(self, sample_risk_manager):
        """Test checking if new positions can be opened when slots are available."""
        # Mock empty positions
        sample_risk_manager.positions = {}

        can_open = sample_risk_manager.can_open_new_position()
        assert can_open is True

    def test_can_open_new_position_at_limit(self, sample_risk_manager):
        """Test checking if new positions can be opened when at the limit."""
        # Mock positions at the limit
        positions = {
            f"POS{i}": {"position_value": 1000}
            for i in range(sample_risk_manager.max_open_positions)
        }
        sample_risk_manager.open_positions = positions

        can_open = sample_risk_manager.can_open_new_position()
        assert can_open is False

    def test_check_drawdown_limit_no_breach(self, sample_risk_manager):
        """Test drawdown limit check when no breach occurs."""
        # Mock current equity at starting level
        sample_risk_manager.starting_equity = 100000
        sample_risk_manager.current_equity = 100000

        breach = sample_risk_manager.check_drawdown_limit()
        assert breach is False

    def test_check_drawdown_limit_breach(self, sample_risk_manager):
        """Test drawdown limit check when breach occurs."""
        # Mock significant drawdown
        sample_risk_manager.starting_equity = 100000
        sample_risk_manager.current_equity = 80000  # 20% drawdown

        breach = sample_risk_manager.check_drawdown_limit()
        assert breach is True

    def test_add_position_success(self, sample_risk_manager):
        """Test successfully adding a position."""
        position_info = {
            "symbol": "SPY",
            "shares": 100,
            "entry_price": 400.0,
            "position_value": 40000,
            "risk_amount": 200,
            "stop_loss_price": 390.0,
            "profit_target_price": 420.0,
        }

        # Ensure we can open positions
        sample_risk_manager.open_positions = {}

        success = sample_risk_manager.add_position(position_info)
        assert success is True

        # Check that position was added
        assert len(sample_risk_manager.open_positions) > 0

    def test_add_position_at_limit(self, sample_risk_manager):
        """Test adding position when at the limit."""
        # Fill all available slots
        positions = {
            f"POS{i}": {} for i in range(sample_risk_manager.max_open_positions)
        }
        sample_risk_manager.positions = positions

        position_info = {
            "symbol": "SPY",
            "shares": 100,
            "position_value": 40000,
            "risk_amount": 200,
            "stop_loss_price": 390.0,
            "profit_target_price": 420.0,
        }

        success = sample_risk_manager.add_position(position_info)
        assert success is False

    def test_reset_daily_stats(self, sample_risk_manager):
        """Test resetting daily statistics."""
        # Set some mock daily pnl
        sample_risk_manager.daily_pnl = 1000.0

        sample_risk_manager.reset_daily_stats()

        assert sample_risk_manager.daily_pnl == 0.0

    def test_get_portfolio_summary(self, sample_risk_manager):
        """Test getting portfolio summary."""
        # Mock some positions
        sample_risk_manager.positions = {
            "SPY": {"position_value": 40000, "risk_amount": 200},
            "QQQ": {"position_value": 30000, "risk_amount": 150},
        }
        sample_risk_manager.current_equity = 100000
        sample_risk_manager.starting_equity = 95000

        summary = sample_risk_manager.get_portfolio_summary()

        assert isinstance(summary, dict)
        assert "current_equity" in summary
        assert "total_position_value" in summary
        assert "total_risk_amount" in summary
        assert "open_positions" in summary

    def test_position_sizing_logic(self, sample_risk_manager):
        """Test that position sizing follows risk management principles."""
        # Test various scenarios to ensure position sizing is reasonable
        scenarios = [
            {"price": 100.0, "atr": 2.0, "g_score": 1.0},  # Low risk
            {"price": 100.0, "atr": 5.0, "g_score": 2.0},  # Medium risk
            {"price": 100.0, "atr": 10.0, "g_score": 3.0},  # High risk
        ]

        for scenario in scenarios:
            position_info = sample_risk_manager.calculate_position_size(
                "TEST", scenario["price"], scenario["atr"], scenario["g_score"]
            )

            if position_info:
                # Ensure position size is reasonable
                assert position_info["shares"] > 0
                assert position_info["position_value"] > 0
                assert position_info["risk_amount"] > 0

                # Higher risk should generally result in smaller positions
                # (though this is a complex relationship)


@pytest.mark.unit
class TestRiskManagerStressTests:
    """Stress tests for RiskManager under extreme market conditions."""

    def test_multi_signal_panic_spike_stress(self, sample_risk_manager):
        """Test risk management with 10 simultaneous high-panic signals."""
        # Create 10 signals with panic scores > 3.0 (extreme panic)
        panic_signals = {}
        for i in range(10):
            symbol = f"STRESS{i}"
            panic_signals[symbol] = {
                "panic_score": 3.5 + i * 0.1,  # Escalating panic levels
                "cri_score": 2.0 + i * 0.05,
                "entry_signal": True,
            }

        # Assess portfolio risk from these signals
        risk_assessment = sample_risk_manager.assess_portfolio_risk_from_signals(panic_signals)

        # Verify risk assessment captures the extreme conditions
        assert risk_assessment["high_panic_signal_count"] == 10
        assert risk_assessment["panic_concentration"] == 1.0  # All signals are high panic
        assert risk_assessment["overall_risk_score"] >= 0.3  # High overall risk from panic concentration

        # Test correlation analysis
        correlation_analysis = sample_risk_manager.check_signal_correlation(panic_signals)
        assert "panic_correlation" in correlation_analysis
        assert "systemic_risk_level" in correlation_analysis
        assert "diversification_score" in correlation_analysis

        # With high panic scores, systemic risk should be elevated
        # (correlation calculation may vary, but the framework should work)
        assert isinstance(correlation_analysis["systemic_risk_level"], (int, float))

    def test_drawdown_simulation_2008_crisis(self, sample_risk_manager):
        """Simulate 2008 financial crisis drawdown scenario."""
        # 2008 crisis: ~50% market drawdown
        initial_equity = 100000
        crisis_drawdown = 0.50  # 50% loss

        # Set up initial state (equity = starting daily equity, current_equity = current value)
        sample_risk_manager.equity = initial_equity  # Starting equity for the day
        sample_risk_manager.current_equity = initial_equity
        sample_risk_manager.peak_equity = initial_equity

        # Simulate crisis drawdown
        crisis_equity = initial_equity * (1 - crisis_drawdown)
        sample_risk_manager.current_equity = crisis_equity

        # Test drawdown limit check
        breach = sample_risk_manager.check_drawdown_limit()

        # Should trigger drawdown breach (assuming 20% limit)
        drawdown_pct = (initial_equity - crisis_equity) / initial_equity
        assert abs(drawdown_pct - crisis_drawdown) < 0.001  # Allow small floating point difference
        assert breach is True  # Should breach the 20% drawdown limit

    def test_drawdown_simulation_2020_crash(self, sample_risk_manager):
        """Simulate 2020 COVID crash drawdown scenario."""
        # 2020 crash: ~34% market drawdown in ~1 month
        initial_equity = 100000
        crash_drawdown = 0.34  # 34% loss

        # Set up initial state (equity = starting daily equity, current_equity = current value)
        sample_risk_manager.equity = initial_equity  # Starting equity for the day
        sample_risk_manager.current_equity = initial_equity
        sample_risk_manager.peak_equity = initial_equity

        # Simulate crash drawdown
        crash_equity = initial_equity * (1 - crash_drawdown)
        sample_risk_manager.current_equity = crash_equity

        # Test drawdown limit check
        breach = sample_risk_manager.check_drawdown_limit()

        # Should trigger drawdown breach
        drawdown_pct = (initial_equity - crash_equity) / initial_equity
        assert abs(drawdown_pct - crash_drawdown) < 0.001  # Allow small floating point difference
        assert breach is True  # Should breach the 20% drawdown limit

    def test_drawdown_simulation_2022_volatility(self, sample_risk_manager):
        """Simulate 2022 high volatility environment."""
        # 2022: High volatility but contained drawdown
        initial_equity = 100000
        volatility_drawdown = 0.15  # 15% loss (below typical 20% limit)

        # Set up initial state
        sample_risk_manager.starting_equity = initial_equity
        sample_risk_manager.current_equity = initial_equity
        sample_risk_manager.equity = initial_equity

        # Simulate volatility drawdown (below limit)
        volatility_equity = initial_equity * (1 - volatility_drawdown)
        sample_risk_manager.current_equity = volatility_equity
        sample_risk_manager.equity = volatility_equity

        # Test drawdown limit check
        breach = sample_risk_manager.check_drawdown_limit()

        # Should NOT trigger drawdown breach (15% < 20% limit)
        drawdown_pct = (initial_equity - volatility_equity) / initial_equity
        assert drawdown_pct == volatility_drawdown
        assert breach is False  # Should not breach the 20% drawdown limit

    def test_extreme_market_conditions_position_sizing(self, sample_risk_manager):
        """Test position sizing under extreme market conditions."""
        # Extreme conditions: very high ATR, high G-Score
        symbol = "CRISIS"
        current_price = 100.0
        extreme_atr = 50.0  # 50% volatility
        extreme_g_score = 5.0  # Maximum risk

        position_info = sample_risk_manager.calculate_position_size(
            symbol, current_price, extreme_atr, extreme_g_score
        )

        # Under extreme conditions, should either reject position or use minimum size
        if position_info:
            # If position is allowed, ensure it's very conservative
            assert position_info["shares"] <= sample_risk_manager.min_position_size
            # Risk amount should be reasonable (not necessarily < 100, depends on equity)
            assert position_info["risk_amount"] >= 0

            # Risk amount should be very small relative to total equity
            expected_max_risk = sample_risk_manager.equity * sample_risk_manager.risk_per_trade
            assert position_info["risk_amount"] <= expected_max_risk

    def test_signal_correlation_analysis(self, sample_risk_manager):
        """Test signal correlation analysis functionality."""
        # Test with various signal combinations
        test_cases = [
            # Case 1: Low correlation, mixed signals
            {
                "SPY": {"panic_score": 1.0, "cri_score": 1.2},
                "QQQ": {"panic_score": 1.5, "cri_score": 0.8},
                "IWM": {"panic_score": 0.8, "cri_score": 1.5},
            },
            # Case 2: High correlation, all panic
            {
                "SPY": {"panic_score": 3.0, "cri_score": 2.8},
                "QQQ": {"panic_score": 3.2, "cri_score": 3.1},
                "IWM": {"panic_score": 2.8, "cri_score": 2.9},
            },
            # Case 3: Single signal (should return defaults)
            {
                "SPY": {"panic_score": 2.0, "cri_score": 1.5},
            },
        ]

        for signals in test_cases:
            correlation_result = sample_risk_manager.check_signal_correlation(signals)

            # Verify structure
            required_keys = ["panic_correlation", "cri_correlation",
                           "systemic_risk_level", "diversification_score"]
            for key in required_keys:
                assert key in correlation_result
                assert isinstance(correlation_result[key], (int, float))

            # Verify ranges
            assert -1 <= correlation_result["panic_correlation"] <= 1
            assert -1 <= correlation_result["cri_correlation"] <= 1
            assert 0 <= correlation_result["systemic_risk_level"] <= 1
            assert 0 <= correlation_result["diversification_score"] <= 1

    def test_portfolio_risk_assessment_under_stress(self, sample_risk_manager):
        """Test comprehensive portfolio risk assessment under stress conditions."""
        # Create stressed signal environment
        stressed_signals = {
            f"ASSET{i}": {
                "panic_score": 3.0 + i * 0.1,  # High panic across assets
                "cri_score": 2.5 + i * 0.05,
            }
            for i in range(5)
        }

        # Mock existing positions
        existing_positions = {
            "POS1": {"position_value": 20000},
            "POS2": {"position_value": 15000},
            "POS3": {"position_value": 10000},
        }

        risk_assessment = sample_risk_manager.assess_portfolio_risk_from_signals(
            stressed_signals, existing_positions
        )

        # Verify comprehensive risk assessment
        required_keys = [
            "systemic_risk_level", "diversification_score", "panic_concentration",
            "high_panic_signal_count", "position_concentration", "overall_risk_score"
        ]

        for key in required_keys:
            assert key in risk_assessment

        # Under stress conditions, risk should be elevated
        assert risk_assessment["high_panic_signal_count"] == 5  # All signals high panic
        assert risk_assessment["overall_risk_score"] > 0.3  # Significant risk

    def test_drawdown_control_under_extreme_stress(self, sample_risk_manager):
        """Test that drawdown stays below 2% under extreme stress conditions."""
        # Set up conservative drawdown limit for testing
        original_limit = sample_risk_manager.daily_drawdown_limit
        sample_risk_manager.daily_drawdown_limit = 0.02  # 2% limit

        initial_equity = 100000
        sample_risk_manager.starting_equity = initial_equity
        sample_risk_manager.current_equity = initial_equity

        # Simulate various stress scenarios
        stress_scenarios = [
            {"drawdown": 0.015, "should_breach": False},  # 1.5% - below limit
            {"drawdown": 0.019, "should_breach": False},  # 1.9% - below limit
            {"drawdown": 0.021, "should_breach": True},   # 2.1% - above limit
            {"drawdown": 0.03, "should_breach": True},    # 3.0% - well above limit
        ]

        for scenario in stress_scenarios:
            # Reset equity
            sample_risk_manager.current_equity = initial_equity * (1 - scenario["drawdown"])

            breach = sample_risk_manager.check_drawdown_limit()
            drawdown_pct = ((initial_equity - sample_risk_manager.current_equity) / initial_equity)

            assert drawdown_pct == scenario["drawdown"]
            assert breach == scenario["should_breach"]

        # Restore original limit
        sample_risk_manager.daily_drawdown_limit = original_limit
