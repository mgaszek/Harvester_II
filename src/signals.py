"""
Signal calculation module for Harvester II trading system.
Implements CRI (Crowd-Reactivity Index), Panic Score, and G-Score calculations.
"""

import logging

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

try:
    import polars as pl

    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    pl = None

# Dependencies are now injected via constructor
from data_processing import DataProcessor
from utils import (
    calculate_z_score,
    validate_symbol,
    validate_tradable_assets,
    validate_universe,
)


class SignalCalculator:
    """Calculates trading signals based on volatility, volume, and attention metrics."""

    def __init__(self, config, data_manager, bayesian_state_machine=None):
        """Initialize signal calculator with injected dependencies."""
        self.config = config
        self.data_manager = data_manager
        self.bayesian_state_machine = bayesian_state_machine
        self.logger = logging.getLogger(__name__)

        # Get parameters from config
        self.lookback_window = self.config.get("system.lookback_window", 90)
        self.cri_threshold = self.config.get("universe.cri_threshold", 0.4)
        self.panic_threshold = self.config.get("signals.panic_threshold", 3.0)
        self.g_score_threshold = self.config.get("macro_risk.g_score_threshold", 2)

        # Log Bayesian State Machine status
        if self.bayesian_state_machine:
            self.logger.info(
                "Bayesian State Machine loaded for signal conviction assessment"
            )
        else:
            self.logger.warning(
                "Bayesian State Machine not available - using rule-based logic"
            )

        # Indicator periods
        self.atr_period = self.config.get("signals.indicators.atr_period", 14)
        self.volume_period = self.config.get("signals.indicators.volume_period", 14)
        self.trends_period = self.config.get("signals.indicators.trends_period", 14)

        # Panic Score weights
        panic_weights = self.config.get("signals.panic_score_weights", {})
        self.volatility_weight = panic_weights.get("volatility_weight", 1.0)
        self.volume_weight = panic_weights.get("volume_weight", 1.0)
        self.trends_weight = panic_weights.get("trends_weight", 0.8)

        # Performance optimization settings
        self.use_polars = self.config.get("performance.use_polars", POLARS_AVAILABLE)

    # _validate_symbol method removed - now using centralized validate_symbol from utils

    def _pct_change(self, series, periods: int = 1):
        """Calculate percentage change using pandas or polars."""
        if self.use_polars and isinstance(series, pl.Series):
            return series.pct_change(periods)
        return series.pct_change(periods)

    def _mean(self, data):
        """Calculate mean using pandas or polars."""
        if self.use_polars and isinstance(data, pl.Series):
            return data.mean()
        return data.mean()

    def _std(self, data):
        """Calculate standard deviation using pandas or polars."""
        if self.use_polars and isinstance(data, pl.Series):
            return data.std()
        return data.std()

    def _tail(self, df, n: int = 5):
        """Get last n rows using pandas or polars."""
        if self.use_polars and isinstance(df, pl.DataFrame):
            return df.tail(n)
        return df.tail(n)

    def _dropna(self, series):
        """Drop NA values using pandas or polars."""
        if self.use_polars and isinstance(series, pl.Series):
            return series.drop_nulls()
        return series.dropna()

    def calculate_cri(
        self, symbol: str, price_data: pd.DataFrame, trends_data: pd.DataFrame
    ) -> float:
        """
        Calculate Crowd-Reactivity Index (CRI) for a symbol.

        Args:
            symbol: Asset symbol
            price_data: Price data DataFrame
            trends_data: Google Trends data DataFrame

        Returns:
            CRI value (absolute correlation between price changes and trends)
        """
        validate_symbol(symbol)

        try:
            if price_data.empty or trends_data.empty:
                return 0.0

            # Align data by date
            aligned_data = self._align_data_by_date(price_data, trends_data)
            if aligned_data.empty:
                return 0.0

            # Calculate daily price changes
            price_changes = self._dropna(self._pct_change(aligned_data["price"]))

            # Calculate daily trends changes
            trends_changes = self._dropna(self._pct_change(aligned_data["trends"]))

            # Ensure same length
            min_length = min(len(price_changes), len(trends_changes))
            if min_length < 10:  # Need minimum data points
                return 0.0

            price_changes = price_changes.iloc[-min_length:]
            trends_changes = trends_changes.iloc[-min_length:]

            # Calculate correlation
            correlation, _ = pearsonr(price_changes, trends_changes)

            # Return absolute correlation as CRI
            cri = abs(correlation) if not np.isnan(correlation) else 0.0

            self.logger.debug(f"CRI for {symbol}: {cri:.4f}")
            return cri

        except Exception as e:
            self.logger.error(f"Failed to calculate CRI for {symbol}: {e}")
            return 0.0

    def calculate_panic_score(
        self, symbol: str, price_data: pd.DataFrame, trends_data: pd.DataFrame
    ) -> float:
        """
        Calculate Panic Score for entry signal.

        Args:
            symbol: Asset symbol
            price_data: Price data with technical indicators
            trends_data: Google Trends data

        Returns:
            Panic Score (sum of z-scores)
        """
        validate_symbol(symbol)

        try:
            if price_data.empty:
                return 0.0

            # Calculate technical indicators if not present
            if "ATR" not in price_data.columns:
                price_data = self.data_manager.calculate_technical_indicators(
                    price_data
                )

            # Get recent data for z-score calculation
            recent_data = self._tail(price_data, self.lookback_window)

            if len(recent_data) < 30:  # Need minimum data
                return 0.0

            # Calculate z-scores
            volatility_z = self._calculate_z_score(
                recent_data["ATR"].iloc[-1], recent_data["ATR"]
            )

            volume_z = self._calculate_z_score(
                recent_data["Volume"].iloc[-1], recent_data["Volume"]
            )

            # Trends z-score
            trends_z = 0.0
            if not trends_data.empty:
                recent_trends = self._tail(trends_data, self.lookback_window)
                if len(recent_trends) >= 30:
                    trends_z = self._calculate_z_score(
                        recent_trends["value"].iloc[-1], recent_trends["value"]
                    )

            # Calculate weighted Panic Score
            panic_score = (
                volatility_z * self.volatility_weight
                + volume_z * self.volume_weight
                + trends_z * self.trends_weight
            )

            self.logger.debug(
                f"Panic Score for {symbol}: {panic_score:.4f} "
                f"(vol: {volatility_z:.2f}, vol: {volume_z:.2f}, trends: {trends_z:.2f})"
            )

            return panic_score

        except Exception as e:
            self.logger.error(f"Failed to calculate Panic Score for {symbol}: {e}")
            return 0.0

    def calculate_g_score(self) -> float:
        """
        Calculate Geopolitical Score (G-Score) for macro risk assessment.

        Returns:
            G-Score (0-3, higher = more risk)
        """
        try:
            score = 0.0

            # Get macro indicators
            vix_data = self.data_manager.get_macro_indicator("VIX")
            spy_data = self.data_manager.get_macro_indicator("SPY")
            oil_data = self.data_manager.get_macro_indicator("USO")

            # VIX threshold check
            if not vix_data.empty:
                current_vix = vix_data["value"].iloc[-1]
                vix_threshold = self.config.get(
                    "macro_risk.indicators.vix_threshold", 25
                )
                if current_vix > vix_threshold:
                    score += 1
                    self.logger.debug(
                        f"VIX above threshold: {current_vix:.2f} > {vix_threshold}"
                    )

            # SPY 7-day return check
            if not spy_data.empty:
                spy_7d_return = self._calculate_period_return(spy_data, 7)
                spy_threshold = self.config.get(
                    "macro_risk.indicators.spy_return_threshold", -0.05
                )
                if spy_7d_return < spy_threshold:
                    score += 1
                    self.logger.debug(
                        f"SPY 7d return below threshold: {spy_7d_return:.4f} < {spy_threshold}"
                    )

            # Oil 7-day return check
            if not oil_data.empty:
                oil_7d_return = self._calculate_period_return(oil_data, 7)
                oil_threshold = self.config.get(
                    "macro_risk.indicators.oil_return_threshold", 0.10
                )
                if oil_7d_return > oil_threshold:
                    score += 1
                    self.logger.debug(
                        f"Oil 7d return above threshold: {oil_7d_return:.4f} > {oil_threshold}"
                    )

            self.logger.info(f"G-Score calculated: {score:.1f}")
            return score

        except Exception as e:
            self.logger.error(f"Failed to calculate G-Score: {e}")
            return 0.0

    def get_tradable_universe(self, universe: list[str]) -> list[str]:
        """
        Filter universe to only include assets with high CRI.

        Args:
            universe: List of asset symbols

        Returns:
            List of tradable assets (high CRI)
        """
        # Use centralized validation
        validate_universe(universe)

        tradable_assets = []

        for symbol in universe:
            try:
                # Get price and trends data
                price_data = self.data_manager.get_price_data(symbol, period="3mo")
                trends_data, trends_latency = self.data_manager.get_google_trends(
                    symbol, timeframe="today 3-m"
                )

                if price_data.empty:
                    self.logger.warning(f"No price data for {symbol}")
                    continue

                # Check for stale trends data (>24 hours)
                stale_data_penalty = 0.0
                if trends_latency > 24:  # More than 24 hours old
                    stale_data_penalty = 0.5  # Reduce CRI by 50%
                    self.logger.warning(
                        f"Using stale trends data for {symbol} ({trends_latency:.1f}h old), "
                        f"applying {stale_data_penalty*100:.0f}% penalty"
                    )

                # Calculate CRI with stale data penalty
                cri = self.calculate_cri(symbol, price_data, trends_data)
                cri = max(0, cri * (1 - stale_data_penalty))  # Apply penalty

                if cri >= self.cri_threshold:
                    tradable_assets.append(symbol)
                    self.logger.info(
                        f"{symbol} added to tradable universe (CRI: {cri:.4f})"
                    )
                else:
                    self.logger.debug(
                        f"{symbol} excluded (CRI: {cri:.4f} < {self.cri_threshold})"
                    )

            except Exception as e:
                self.logger.error(f"Failed to evaluate {symbol}: {e}")

        self.logger.info(
            f"Tradable universe: {len(tradable_assets)}/{len(universe)} assets"
        )
        return tradable_assets

    def get_entry_signals(self, tradable_assets: list[str]) -> list[dict[str, any]]:
        """
        Get entry signals for tradable assets using Bayesian State Machine conviction assessment.

        Args:
            tradable_assets: List of tradable asset symbols

        Returns:
            List of entry signal dictionaries with conviction levels
        """
        # Use centralized validation
        validate_tradable_assets(tradable_assets)

        signals = []

        for symbol in tradable_assets:
            try:
                # Get recent data
                price_data = self.data_manager.get_price_data(symbol, period="3mo")
                trends_data, trends_latency = self.data_manager.get_google_trends(
                    symbol, timeframe="today 3-m"
                )

                if price_data.empty:
                    continue

                # Check for stale trends data and reduce conviction
                conviction_multiplier = 1.0
                if trends_latency > 24:  # More than 24 hours old
                    conviction_multiplier = 0.0  # Zero conviction for stale data
                    self.logger.warning(
                        f"Stale trends data for {symbol} ({trends_latency:.1f}h old), "
                        f"setting conviction to zero"
                    )

                # Calculate base Panic Score
                panic_score = self.calculate_panic_score(
                    symbol, price_data, trends_data
                )

                # Calculate additional features for state machine
                price_change_5d = self._calculate_period_return(price_data, 5)
                g_score = self.calculate_g_score()  # Get current macro risk score

                # Extract z-score components from panic score calculation
                volatility_z = 0.0
                volume_z = 0.0
                trends_z = 0.0

                try:
                    # Recalculate individual z-scores for state machine features
                    recent_prices = self._tail(price_data, self.lookback_window)
                    if len(recent_prices) >= 30:
                        volatility_z = self._calculate_z_score(
                            recent_prices["Close"].pct_change().std() * np.sqrt(252),
                            recent_prices["Close"].pct_change().rolling(30).std()
                            * np.sqrt(252),
                        )

                        volume_z = self._calculate_z_score(
                            recent_prices["Volume"].iloc[-1], recent_prices["Volume"]
                        )

                    if not trends_data.empty:
                        recent_trends = self._tail(trends_data, self.lookback_window)
                        if len(recent_trends) >= 30:
                            trends_z = self._calculate_z_score(
                                recent_trends["value"].iloc[-1], recent_trends["value"]
                            )
                except Exception as e:
                    self.logger.debug(
                        f"Could not calculate all z-scores for {symbol}: {e}"
                    )

                # Enhanced Panic Score with Bayesian conviction multiplier
                enhanced_panic_score = panic_score
                conviction_level = "low"
                confidence = 0.0
                market_state = "unknown"
                assessment_method = "rules"

                # Apply stale data penalty to conviction multiplier
                final_conviction_multiplier = conviction_multiplier

                if self.bayesian_state_machine:
                    try:
                        features = self.bayesian_state_machine.prepare_features(
                            volatility_z, volume_z, trends_z, g_score, price_change_5d
                        )
                        conviction = self.bayesian_state_machine.assess_conviction(
                            features
                        )

                        # Enhance panic score with Bayesian conviction
                        bayesian_multiplier = conviction["confidence"]
                        final_conviction_multiplier = conviction_multiplier * bayesian_multiplier
                        enhanced_panic_score = panic_score * final_conviction_multiplier

                        conviction_level = conviction["conviction_level"]
                        confidence = conviction["confidence"]
                        market_state = conviction.get("state", "unknown")
                        assessment_method = f"{conviction['method']}_stale_penalty" if conviction_multiplier < 1.0 else conviction["method"]

                        self.logger.debug(
                            f"Enhanced panic score for {symbol}: {panic_score:.2f} -> {enhanced_panic_score:.2f} "
                            f"(stale_mult: {conviction_multiplier:.2f}, bayesian_mult: {bayesian_multiplier:.2f})"
                        )

                    except Exception as e:
                        self.logger.warning(
                            f"Bayesian enhancement failed for {symbol}, using base score: {e}"
                        )
                        # Fall back to base panic score with stale penalty
                        enhanced_panic_score = panic_score * conviction_multiplier
                        assessment_method = "rules_stale_penalty" if conviction_multiplier < 1.0 else "rules"
                else:
                    # No Bayesian, just apply stale penalty
                    enhanced_panic_score = panic_score * conviction_multiplier
                    assessment_method = "rules_stale_penalty" if conviction_multiplier < 1.0 else "rules"

                # Generate signal based on enhanced panic score
                if enhanced_panic_score > self.panic_threshold:
                    signal = {
                        "symbol": symbol,
                        "panic_score": panic_score,  # Original score
                        "enhanced_panic_score": enhanced_panic_score,  # Bayesian-enhanced score
                        "price_change_5d": price_change_5d,
                        "side": "BUY"
                        if price_change_5d < 0
                        else "SELL",  # Contrarian logic
                        "current_price": price_data["Close"].iloc[-1],
                        "timestamp": pd.Timestamp.now(),
                        "conviction_level": conviction_level,
                        "confidence": confidence,
                        "market_state": market_state,
                        "assessment_method": assessment_method,
                    }

                    signals.append(signal)
                    self.logger.info(
                        f"Entry signal for {symbol}: {signal['side']} "
                        f"(Enhanced Panic: {enhanced_panic_score:.2f}, Base: {panic_score:.2f}, "
                        f"Conviction: {conviction_level})"
                    )

            except Exception as e:
                self.logger.error(f"Failed to get entry signal for {symbol}: {e}")

        return signals

    def _align_data_by_date(self, price_data, trends_data):
        """Align price and trends data by date."""
        try:
            if DataProcessor.get_backend() == "polars":
                # Polars implementation - simplified for now
                return DataProcessor.create_dataframe({})
            # Pandas implementation
            # Convert indices to date if needed
            price_dates = pd.to_datetime(price_data.index).date
            trends_dates = pd.to_datetime(trends_data.index).date

            # Find common dates
            common_dates = set(price_dates) & set(trends_dates)

            if not common_dates:
                return pd.DataFrame()

            # Create aligned DataFrame
            aligned_data = []
            for date in sorted(common_dates):
                price_idx = price_dates.tolist().index(date)
                trends_idx = trends_dates.tolist().index(date)

                aligned_data.append(
                    {
                        "date": date,
                        "price": price_data["Close"].iloc[price_idx],
                        "trends": trends_data["value"].iloc[trends_idx],
                    }
                )

            return pd.DataFrame(aligned_data).set_index("date")

        except Exception as e:
            self.logger.error(f"Failed to align data by date: {e}")
            return DataProcessor.create_dataframe({})

    def _calculate_z_score(
        self, current_value: float, historical_values: pd.Series
    ) -> float:
        """Calculate z-score for current value against historical data using centralized function."""
        try:
            return calculate_z_score(current_value, historical_values)
        except Exception as e:
            self.logger.error(f"Failed to calculate z-score: {e}")
            return 0.0

    def _calculate_period_return(self, data: pd.DataFrame, periods: int) -> float:
        """Calculate return over specified number of periods."""
        try:
            if len(data) < periods + 1:
                return 0.0

            current_value = data["value"].iloc[-1]
            past_value = data["value"].iloc[-(periods + 1)]

            return (current_value - past_value) / past_value

        except Exception as e:
            self.logger.error(f"Failed to calculate {periods}-period return: {e}")
            return 0.0


# SignalCalculator is now created via dependency injection in di.py
