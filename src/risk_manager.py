"""
Risk management module for Harvester II trading system.
Handles position sizing, drawdown limits, and risk controls.
"""

from datetime import datetime
import logging

import pandas as pd

# Config is now injected via constructor


class RiskManager:
    """Manages risk controls, position sizing, and drawdown limits."""

    def __init__(self, config):
        """Initialize risk manager with injected configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Risk parameters from config
        self.equity = self.config.get("risk_management.equity", 100000)
        self.base_position_fraction = self.config.get(
            "risk_management.base_position_fraction", 0.005
        )
        self.max_open_positions = self.config.get(
            "risk_management.max_open_positions", 4
        )
        self.daily_drawdown_limit = self.config.get(
            "risk_management.daily_drawdown_limit", 0.02
        )

        # Position sizing limits
        self.min_position_size = self.config.get(
            "risk_management.position_sizing.min_position_size", 100
        )
        self.max_position_size = self.config.get(
            "risk_management.position_sizing.max_position_size", 5000
        )
        self.risk_per_trade = self.config.get(
            "risk_management.position_sizing.risk_per_trade", 0.005
        )

        # ATR-based stop loss and profit targets
        self.atr_profit_multiplier = self.config.get(
            "risk_management.atr_profit_target_multiplier", 2.0
        )
        self.atr_stop_multiplier = self.config.get(
            "risk_management.atr_stop_loss_multiplier", 1.0
        )

        # Trailing stops configuration
        trailing_config = self.config.get("risk_management.trailing_stops", {})
        self.trailing_stops_enabled = trailing_config.get("enabled", True)
        self.trailing_atr_multiplier = trailing_config.get("atr_multiplier", 1.5)
        self.trailing_min_distance = trailing_config.get("min_distance", 0.02)
        self.trailing_update_freq = trailing_config.get("update_frequency", "daily")

        # Macro risk adjustments
        self.g_score_threshold = self.config.get("macro_risk.g_score_threshold", 2)
        self.high_risk_multiplier = self.config.get(
            "macro_risk.position_size_multiplier.high_risk", 0.5
        )
        self.normal_risk_multiplier = self.config.get(
            "macro_risk.position_size_multiplier.normal_risk", 1.0
        )

        # State tracking
        self.current_equity = self.equity
        self.daily_pnl = 0.0
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        self.peak_equity = self.equity

        # Position tracking
        self.open_positions: dict[str, dict] = {}
        self.position_history: list[dict] = []

        self.logger.info(f"Risk Manager initialized with ${self.equity:,.2f} equity")

    def calculate_position_size(
        self, symbol: str, entry_price: float, atr: float, g_score: float = 0.0
    ) -> dict[str, float]:
        """
        Calculate position size based on risk parameters and macro conditions.

        Args:
            symbol: Asset symbol
            entry_price: Entry price for the position
            atr: Average True Range for stop loss calculation
            g_score: Current G-Score for macro risk adjustment

        Returns:
            Dictionary with position sizing details
        """
        try:
            # Base position size
            base_size = self.current_equity * self.base_position_fraction

            # Apply macro risk multiplier
            if g_score >= self.g_score_threshold:
                risk_multiplier = self.high_risk_multiplier
                self.logger.info(
                    f"High macro risk detected (G-Score: {g_score}), reducing position size"
                )
            else:
                risk_multiplier = self.normal_risk_multiplier

            adjusted_size = base_size * risk_multiplier

            # Apply position size limits
            position_size = max(
                self.min_position_size, min(adjusted_size, self.max_position_size)
            )

            # Calculate shares/units
            shares = int(position_size / entry_price)
            actual_position_value = shares * entry_price

            # Calculate stop loss and profit target levels
            stop_loss_price = entry_price - (atr * self.atr_stop_multiplier)
            profit_target_price = entry_price + (atr * self.atr_profit_multiplier)

            # Calculate risk amount
            risk_amount = actual_position_value - (shares * stop_loss_price)
            risk_percentage = risk_amount / self.current_equity

            position_info = {
                "symbol": symbol,
                "shares": shares,
                "entry_price": entry_price,
                "position_value": actual_position_value,
                "stop_loss_price": stop_loss_price,
                "profit_target_price": profit_target_price,
                "risk_amount": risk_amount,
                "risk_percentage": risk_percentage,
                "atr": atr,
                "g_score": g_score,
                "risk_multiplier": risk_multiplier,
                "timestamp": datetime.now(),
            }

            self.logger.info(
                f"Position sizing for {symbol}: {shares} shares, "
                f"${actual_position_value:,.2f} value, "
                f"{risk_percentage:.2%} risk"
            )

            return position_info

        except Exception as e:
            self.logger.error(f"Failed to calculate position size for {symbol}: {e}")
            return {}

    def check_drawdown_limit(self) -> bool:
        """
        Check if daily drawdown limit has been exceeded.

        Returns:
            True if drawdown limit exceeded (should stop trading)
        """
        try:
            # Calculate current drawdown
            if self.current_equity > self.peak_equity:
                self.peak_equity = self.current_equity
                self.current_drawdown = 0.0
            else:
                self.current_drawdown = (
                    self.peak_equity - self.current_equity
                ) / self.peak_equity

            # Update maximum drawdown
            self.max_drawdown = max(self.current_drawdown, self.max_drawdown)

            # Check daily drawdown limit
            daily_drawdown_amount = self.daily_drawdown_limit * self.equity
            current_drawdown_amount = self.equity - self.current_equity

            if current_drawdown_amount >= daily_drawdown_amount:
                self.logger.critical(
                    f"DAILY DRAWDOWN LIMIT EXCEEDED: "
                    f"${current_drawdown_amount:,.2f} >= ${daily_drawdown_amount:,.2f}"
                )
                return True

            self.logger.debug(
                f"Drawdown check: {self.current_drawdown:.2%} "
                f"(limit: {self.daily_drawdown_limit:.2%})"
            )

            return False

        except Exception as e:
            self.logger.error(f"Failed to check drawdown limit: {e}")
            return True  # Fail safe - stop trading on error

    def can_open_new_position(self) -> bool:
        """
        Check if we can open a new position based on current limits.

        Returns:
            True if new position can be opened
        """
        try:
            # Check drawdown limit first
            if self.check_drawdown_limit():
                return False

            # Check maximum open positions
            if len(self.open_positions) >= self.max_open_positions:
                self.logger.info(
                    f"Maximum open positions reached: {len(self.open_positions)}"
                )
                return False

            # Check available equity
            total_position_value = sum(
                pos["position_value"] for pos in self.open_positions.values()
            )
            available_equity = self.current_equity - total_position_value

            if available_equity < self.min_position_size:
                self.logger.info(
                    f"Insufficient equity for new position: ${available_equity:,.2f}"
                )
                return False

            return True

        except Exception as e:
            self.logger.error(f"Failed to check if new position can be opened: {e}")
            return False

    def add_position(self, position_info: dict[str, float]) -> bool:
        """
        Add a new position to the portfolio.

        Args:
            position_info: Position information from calculate_position_size

        Returns:
            True if position added successfully
        """
        try:
            symbol = position_info["symbol"]

            if symbol in self.open_positions:
                self.logger.warning(f"Position already exists for {symbol}")
                return False

            if not self.can_open_new_position():
                self.logger.warning(
                    f"Cannot add position for {symbol} - limits exceeded"
                )
                return False

            # Add position with trailing stop initialization
            position_info["trailing_stop"] = position_info[
                "stop_loss_price"
            ]  # Initialize trailing stop
            self.open_positions[symbol] = position_info

            # Update equity (subtract position value)
            self.current_equity -= position_info["position_value"]

            self.logger.info(
                f"Added position for {symbol}: "
                f"{position_info['shares']} shares at ${position_info['entry_price']:.2f}"
            )

            return True

        except Exception as e:
            self.logger.error(f"Failed to add position: {e}")
            return False

    def remove_position(
        self, symbol: str, exit_price: float, exit_reason: str = "manual"
    ) -> dict[str, float]:
        """
        Remove a position from the portfolio and calculate P&L.

        Args:
            symbol: Asset symbol
            exit_price: Exit price for the position
            exit_reason: Reason for exit (stop_loss, profit_target, manual)

        Returns:
            Dictionary with trade results
        """
        try:
            if symbol not in self.open_positions:
                self.logger.warning(f"No position found for {symbol}")
                return {}

            position = self.open_positions[symbol]

            # Calculate P&L
            shares = position["shares"]
            entry_price = position["entry_price"]
            position_value = position["position_value"]

            exit_value = shares * exit_price
            pnl = exit_value - position_value
            pnl_percentage = pnl / position_value

            # Update equity
            self.current_equity += exit_value

            # Create trade record
            trade_record = {
                "symbol": symbol,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "shares": shares,
                "entry_value": position_value,
                "exit_value": exit_value,
                "pnl": pnl,
                "pnl_percentage": pnl_percentage,
                "exit_reason": exit_reason,
                "entry_time": position["timestamp"],
                "exit_time": datetime.now(),
                "duration": datetime.now() - position["timestamp"],
                "atr": position["atr"],
                "g_score": position["g_score"],
            }

            # Add to history and remove from open positions
            self.position_history.append(trade_record)
            del self.open_positions[symbol]

            # Update daily P&L
            self.daily_pnl += pnl

            self.logger.info(
                f"Closed position for {symbol}: "
                f"${pnl:,.2f} P&L ({pnl_percentage:.2%}) - {exit_reason}"
            )

            return trade_record

        except Exception as e:
            self.logger.error(f"Failed to remove position for {symbol}: {e}")
            return {}

    def check_exit_signals(
        self, current_prices: dict[str, float]
    ) -> list[dict[str, float]]:
        """
        Check all open positions for exit signals (stop loss or profit target).

        Args:
            current_prices: Dictionary mapping symbols to current prices

        Returns:
            List of positions that should be exited
        """
        exits = []

        try:
            for symbol, position in self.open_positions.items():
                if symbol not in current_prices:
                    self.logger.warning(f"No current price for {symbol}")
                    continue

                current_price = current_prices[symbol]
                stop_loss = position["stop_loss_price"]
                profit_target = position["profit_target_price"]

                # Check exit conditions
                if current_price <= stop_loss:
                    exits.append(
                        {
                            "symbol": symbol,
                            "exit_price": current_price,
                            "exit_reason": "stop_loss",
                            "position": position,
                        }
                    )
                elif current_price >= profit_target:
                    exits.append(
                        {
                            "symbol": symbol,
                            "exit_price": current_price,
                            "exit_reason": "profit_target",
                            "position": position,
                        }
                    )

            if exits:
                self.logger.info(f"Found {len(exits)} exit signals")

            return exits

        except Exception as e:
            self.logger.error(f"Failed to check exit signals: {e}")
            return []

    def get_portfolio_summary(self) -> dict[str, any]:
        """
        Get current portfolio summary.

        Returns:
            Dictionary with portfolio statistics
        """
        try:
            total_position_value = sum(
                pos["position_value"] for pos in self.open_positions.values()
            )
            total_risk_amount = sum(
                pos["risk_amount"] for pos in self.open_positions.values()
            )

            # Calculate unrealized P&L
            unrealized_pnl = 0.0
            for symbol, position in self.open_positions.items():
                # This would need current prices - placeholder for now
                unrealized_pnl += 0.0

            summary = {
                "current_equity": self.current_equity,
                "total_equity": self.equity,
                "daily_pnl": self.daily_pnl,
                "current_drawdown": self.current_drawdown,
                "max_drawdown": self.max_drawdown,
                "open_positions": len(self.open_positions),
                "max_positions": self.max_open_positions,
                "total_position_value": total_position_value,
                "total_risk_amount": total_risk_amount,
                "available_equity": self.current_equity - total_position_value,
                "can_open_new": self.can_open_new_position(),
                "drawdown_limit_exceeded": self.check_drawdown_limit(),
            }

            return summary

        except Exception as e:
            self.logger.error(f"Failed to get portfolio summary: {e}")
            return {}

    def reset_daily_stats(self) -> None:
        """Reset daily statistics (call at start of new trading day)."""
        self.daily_pnl = 0.0
        self.logger.info("Daily statistics reset")

    def update_trailing_stops(
        self, current_prices: dict[str, float], atr_values: dict[str, float]
    ) -> None:
        """
        Update trailing stops for all open positions based on current ATR.

        Args:
            current_prices: Dictionary of current prices by symbol
            atr_values: Dictionary of current ATR values by symbol
        """
        if not self.trailing_stops_enabled:
            return

        try:
            updated_stops = 0

            for symbol, position in self.open_positions.items():
                if symbol not in current_prices or symbol not in atr_values:
                    continue

                current_price = current_prices[symbol]
                current_atr = atr_values[symbol]

                # Calculate potential new trailing stop
                if position["side"] == "BUY":
                    # For long positions, trailing stop moves up with price
                    trailing_distance = current_atr * self.trailing_atr_multiplier
                    potential_stop = current_price - trailing_distance

                    # Ensure minimum distance from current price
                    min_stop_distance = current_price * self.trailing_min_distance
                    potential_stop = min(
                        potential_stop, current_price - min_stop_distance
                    )

                    # Only update if new stop is higher than current stop
                    if potential_stop > position.get(
                        "trailing_stop", position["stop_loss_price"]
                    ):
                        position["trailing_stop"] = potential_stop
                        position["stop_loss_price"] = (
                            potential_stop  # Update the actual stop
                        )
                        updated_stops += 1
                        self.logger.debug(
                            f"Updated trailing stop for {symbol}: ${potential_stop:.2f}"
                        )

                else:  # SELL (short positions)
                    # For short positions, trailing stop moves down with price
                    trailing_distance = current_atr * self.trailing_atr_multiplier
                    potential_stop = current_price + trailing_distance

                    # Ensure minimum distance from current price
                    min_stop_distance = current_price * self.trailing_min_distance
                    potential_stop = max(
                        potential_stop, current_price + min_stop_distance
                    )

                    # Only update if new stop is lower than current stop
                    if potential_stop < position.get(
                        "trailing_stop", position["stop_loss_price"]
                    ):
                        position["trailing_stop"] = potential_stop
                        position["stop_loss_price"] = (
                            potential_stop  # Update the actual stop
                        )
                        updated_stops += 1
                        self.logger.debug(
                            f"Updated trailing stop for {symbol}: ${potential_stop:.2f}"
                        )

            if updated_stops > 0:
                self.logger.info(
                    f"Updated trailing stops for {updated_stops} positions"
                )

        except Exception as e:
            self.logger.error(f"Failed to update trailing stops: {e}")

    def get_trailing_stop_info(self, symbol: str) -> dict[str, float] | None:
        """
        Get trailing stop information for a position.

        Args:
            symbol: Symbol to get trailing stop info for

        Returns:
            Dictionary with trailing stop information or None if position not found
        """
        try:
            if symbol not in self.open_positions:
                return None

            position = self.open_positions[symbol]
            return {
                "current_stop": position["stop_loss_price"],
                "trailing_stop": position.get("trailing_stop"),
                "entry_price": position["entry_price"],
                "current_price": position.get("current_price"),
                "atr": position.get("atr", 0),
            }

        except Exception as e:
            self.logger.error(f"Failed to get trailing stop info for {symbol}: {e}")
            return None

    def get_position_history(self) -> pd.DataFrame:
        """
        Get position history as DataFrame.

        Returns:
            DataFrame with trade history
        """
        if not self.position_history:
            return pd.DataFrame()

        return pd.DataFrame(self.position_history)


# RiskManager is now created via dependency injection in di.py
