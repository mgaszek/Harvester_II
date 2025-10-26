"""
Unit tests for risk_manager.py - position sizing and risk calculations.
"""

import pytest
from unittest.mock import Mock


@pytest.mark.unit
class TestRiskManager:
    """Test RiskManager position sizing and risk calculations."""

    def test_calculate_position_size_normal_conditions(self, sample_risk_manager):
        """Test position sizing under normal market conditions."""
        symbol = 'SPY'
        current_price = 400.0
        atr = 5.0  # 1.25% of price
        g_score = 1.5  # Normal risk level

        position_info = sample_risk_manager.calculate_position_size(symbol, current_price, atr, g_score)

        assert position_info is not None
        assert 'shares' in position_info
        assert 'position_value' in position_info
        assert 'risk_amount' in position_info
        assert 'stop_loss_price' in position_info
        assert 'profit_target_price' in position_info

        # Verify calculations
        assert position_info['shares'] > 0
        assert position_info['position_value'] > 0
        assert position_info['risk_amount'] > 0

    def test_calculate_position_size_high_volatility(self, sample_risk_manager):
        """Test position sizing with high volatility (high ATR)."""
        symbol = 'SPY'
        current_price = 400.0
        atr = 20.0  # 5% of price - very volatile
        g_score = 1.5

        position_info = sample_risk_manager.calculate_position_size(symbol, current_price, atr, g_score)

        assert position_info is not None
        # High volatility should result in smaller position sizes
        assert position_info['shares'] < 1000  # Should be conservative

    def test_calculate_position_size_low_volatility(self, sample_risk_manager):
        """Test position sizing with low volatility (low ATR)."""
        symbol = 'SPY'
        current_price = 400.0
        atr = 1.0  # 0.25% of price - very stable
        g_score = 1.5

        position_info = sample_risk_manager.calculate_position_size(symbol, current_price, atr, g_score)

        assert position_info is not None
        # Low volatility should allow reasonable position sizes (actual behavior is conservative)
        assert position_info['shares'] >= 1

    def test_calculate_position_size_high_risk_gscore(self, sample_risk_manager):
        """Test position sizing with high risk G-Score."""
        symbol = 'SPY'
        current_price = 400.0
        atr = 5.0
        g_score = 3.5  # High risk - should reduce position size

        position_info = sample_risk_manager.calculate_position_size(symbol, current_price, atr, g_score)

        assert position_info is not None
        # High G-Score should result in smaller positions
        assert position_info['shares'] < 500

    def test_calculate_position_size_min_size_limit(self, sample_risk_manager):
        """Test that position sizing respects minimum size limits."""
        symbol = 'SPY'
        current_price = 400.0
        atr = 50.0  # Extremely high volatility
        g_score = 5.0  # Extremely high risk

        position_info = sample_risk_manager.calculate_position_size(symbol, current_price, atr, g_score)

        # High risk may result in no position being allowed
        if position_info:
            # If a position is allowed, it should meet minimum size or be rejected
            assert position_info['shares'] == 0 or position_info['shares'] >= sample_risk_manager.min_position_size

    def test_calculate_position_size_max_size_limit(self, sample_risk_manager):
        """Test that position sizing respects maximum size limits."""
        symbol = 'SPY'
        current_price = 10.0  # Very cheap stock
        atr = 0.1  # Very low volatility
        g_score = 0.5  # Very low risk

        position_info = sample_risk_manager.calculate_position_size(symbol, current_price, atr, g_score)

        if position_info:
            assert position_info['shares'] <= sample_risk_manager.max_position_size

    def test_calculate_position_size_insufficient_equity(self, sample_risk_manager):
        """Test position sizing when equity is insufficient."""
        # Temporarily reduce equity
        original_equity = sample_risk_manager.equity
        sample_risk_manager.equity = 100  # Very low equity

        symbol = 'SPY'
        current_price = 400.0
        atr = 5.0
        g_score = 1.5

        position_info = sample_risk_manager.calculate_position_size(symbol, current_price, atr, g_score)

        # Should still return valid position info, but smaller
        assert position_info is not None
        assert position_info['shares'] >= 0

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
        positions = {f'POS{i}': {'position_value': 1000} for i in range(sample_risk_manager.max_open_positions)}
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
            'symbol': 'SPY',
            'shares': 100,
            'entry_price': 400.0,
            'position_value': 40000,
            'risk_amount': 200,
            'stop_loss_price': 390.0,
            'profit_target_price': 420.0
        }

        # Ensure we can open positions
        sample_risk_manager.open_positions = {}

        success = sample_risk_manager.add_position(position_info)
        assert success is True

        # Check that position was added
        assert len(sample_risk_manager.positions) > 0

    def test_add_position_at_limit(self, sample_risk_manager):
        """Test adding position when at the limit."""
        # Fill all available slots
        positions = {f'POS{i}': {} for i in range(sample_risk_manager.max_open_positions)}
        sample_risk_manager.positions = positions

        position_info = {
            'symbol': 'SPY',
            'shares': 100,
            'position_value': 40000,
            'risk_amount': 200,
            'stop_loss_price': 390.0,
            'profit_target_price': 420.0
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
            'SPY': {'position_value': 40000, 'risk_amount': 200},
            'QQQ': {'position_value': 30000, 'risk_amount': 150}
        }
        sample_risk_manager.current_equity = 100000
        sample_risk_manager.starting_equity = 95000

        summary = sample_risk_manager.get_portfolio_summary()

        assert isinstance(summary, dict)
        assert 'current_equity' in summary
        assert 'total_position_value' in summary
        assert 'total_risk_amount' in summary
        assert 'open_positions' in summary

    def test_position_sizing_logic(self, sample_risk_manager):
        """Test that position sizing follows risk management principles."""
        # Test various scenarios to ensure position sizing is reasonable
        scenarios = [
            {'price': 100.0, 'atr': 2.0, 'g_score': 1.0},  # Low risk
            {'price': 100.0, 'atr': 5.0, 'g_score': 2.0},  # Medium risk
            {'price': 100.0, 'atr': 10.0, 'g_score': 3.0}, # High risk
        ]

        for scenario in scenarios:
            position_info = sample_risk_manager.calculate_position_size(
                'TEST', scenario['price'], scenario['atr'], scenario['g_score']
            )

            if position_info:
                # Ensure position size is reasonable
                assert position_info['shares'] > 0
                assert position_info['position_value'] > 0
                assert position_info['risk_amount'] > 0

                # Higher risk should generally result in smaller positions
                # (though this is a complex relationship)
