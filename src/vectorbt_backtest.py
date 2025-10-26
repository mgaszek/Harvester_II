"""
Vectorbt-based backtesting engine for Harvester II.
Provides high-performance vectorized backtesting using vectorbt library.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta

# Vectorbt imports
try:
    import vectorbt as vbt
    VECTORBT_AVAILABLE = True
except ImportError:
    vbt = None
    VECTORBT_AVAILABLE = False

# Dependencies are now injected via constructor
from utils import calculate_performance_metrics


class VectorbtBacktestEngine:
    """
    Vectorbt-powered backtesting engine for Harvester II.

    Leverages vectorbt's vectorized operations for high-performance backtesting
    while maintaining compatibility with existing signal generation and risk management.
    """

    def __init__(self, config, data_manager, signal_calculator, risk_manager):
        """Initialize vectorbt backtesting engine with injected dependencies."""
        self.config = config
        self.data_manager = data_manager
        self.signal_calculator = signal_calculator
        self.risk_manager = risk_manager
        self.logger = logging.getLogger(__name__)

        if not VECTORBT_AVAILABLE:
            self.logger.warning("Vectorbt not available - VectorbtBacktestEngine disabled")
            return

        # Vectorbt configuration
        self.vbt_config = {
            'fees': self.config.get('trading.execution.commission_per_trade', 0.001),
            'slippage': self.config.get('trading.execution.slippage_tolerance', 0.001),
            'min_size': self.config.get('risk_management.position_sizing.min_position_size', 100),
            'max_size': self.config.get('risk_management.position_sizing.max_position_size', 5000)
        }

        self.logger.info("Vectorbt backtesting engine initialized")

    def run_vectorbt_backtest(self, start_date: str, end_date: str,
                            initial_capital: float = 100000) -> Dict[str, Any]:
        """
        Run vectorbt-powered backtest.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            initial_capital: Starting capital

        Returns:
            Dictionary with vectorbt backtest results
        """
        if not VECTORBT_AVAILABLE:
            return {'error': 'Vectorbt not available'}

        try:
            self.logger.info(f"Starting vectorbt backtest: {start_date} to {end_date}")

            # Get price data for all assets
            universe = self.config.get('universe.assets', [])
            price_data = self._get_price_data_vectorbt(universe, start_date, end_date)

            if price_data.empty:
                return {'error': 'No price data available for backtest period'}

            # Generate signals for vectorbt
            signals = self._generate_vectorbt_signals(price_data, start_date, end_date)

            if signals.empty:
                return {'error': 'No signals generated for backtest period'}

            # Create vectorbt portfolio
            portfolio = self._create_vectorbt_portfolio(price_data, signals, initial_capital)

            # Analyze results
            results = self._analyze_vectorbt_results(portfolio, start_date, end_date, initial_capital)

            self.logger.info("Vectorbt backtest completed successfully")
            return results

        except Exception as e:
            self.logger.error(f"Vectorbt backtest failed: {e}")
            return {'error': str(e)}

    def _get_price_data_vectorbt(self, universe: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get price data in vectorbt format (multi-asset DataFrame).

        Args:
            universe: List of asset symbols
            start_date: Start date
            end_date: End date

        Returns:
            Multi-asset price DataFrame
        """
        try:
            # Get price data for each asset
            price_dfs = []
            valid_assets = []

            for symbol in universe:
                try:
                    # Get extended data for proper indicators
                    extended_start = pd.to_datetime(start_date) - timedelta(days=365)

                    ticker = vbt.YFData.download(
                        symbol,
                        start=extended_start.strftime('%Y-%m-%d'),
                        end=end_date,
                        auto_adjust=True
                    )

                    if not ticker.empty and len(ticker) > 100:  # Require minimum data
                        # Calculate technical indicators
                        ticker = self._add_technical_indicators(ticker)
                        price_dfs.append(ticker[['Close']].rename(columns={'Close': symbol}))
                        valid_assets.append(symbol)

                except Exception as e:
                    self.logger.debug(f"Failed to load {symbol} for vectorbt: {e}")
                    continue

            if not price_dfs:
                return pd.DataFrame()

            # Combine all assets into multi-asset DataFrame
            combined_data = pd.concat(price_dfs, axis=1)
            combined_data = combined_data.dropna(how='all')  # Remove rows with all NaN

            # Filter to backtest period
            period_mask = (combined_data.index >= start_date) & (combined_data.index <= end_date)
            filtered_data = combined_data[period_mask]

            self.logger.info(f"Loaded price data for {len(valid_assets)} assets: {valid_assets[:5]}...")
            return filtered_data

        except Exception as e:
            self.logger.error(f"Failed to get vectorbt price data: {e}")
            return pd.DataFrame()

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators required for signal generation."""
        try:
            # Add basic indicators needed for signals
            close = df['Close']

            # Simple moving averages for basic trend analysis
            df['SMA_20'] = close.rolling(20).mean()
            df['SMA_50'] = close.rolling(50).mean()

            # Volatility (ATR proxy)
            high_low = df['High'] - df['Low']
            high_close = (df['High'] - close.shift(1)).abs()
            low_close = (df['Low'] - close.shift(1)).abs()
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['ATR'] = true_range.rolling(14).mean()

            # Volume indicators
            df['Volume_SMA'] = df['Volume'].rolling(20).mean()

            return df

        except Exception as e:
            self.logger.debug(f"Failed to add technical indicators: {e}")
            return df

    def _generate_vectorbt_signals(self, price_data: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Generate trading signals compatible with vectorbt format.

        Args:
            price_data: Multi-asset price DataFrame
            start_date: Backtest start date
            end_date: Backtest end date

        Returns:
            DataFrame with entry/exit signals for each asset
        """
        try:
            signals_dict = {}

            for symbol in price_data.columns:
                try:
                    # Get individual asset data
                    asset_data = price_data[[symbol]].dropna()
                    asset_prices = asset_data[symbol]

                    # Generate signals using existing signal calculator
                    # We'll create a simplified version that works with vectorbt's format

                    # Calculate basic signals (this would normally use the full signal calculator)
                    signals = self._calculate_basic_signals(asset_prices, symbol)

                    signals_dict[symbol] = signals

                except Exception as e:
                    self.logger.debug(f"Failed to generate signals for {symbol}: {e}")
                    continue

            if not signals_dict:
                return pd.DataFrame()

            # Combine signals into DataFrame
            signals_df = pd.DataFrame(signals_dict)

            # Filter to backtest period
            period_mask = (signals_df.index >= start_date) & (signals_df.index <= end_date)
            filtered_signals = signals_df[period_mask]

            self.logger.info(f"Generated signals for {len(signals_dict)} assets")
            return filtered_signals

        except Exception as e:
            self.logger.error(f"Failed to generate vectorbt signals: {e}")
            return pd.DataFrame()

    def _calculate_basic_signals(self, prices: pd.Series, symbol: str) -> pd.Series:
        """
        Calculate basic entry/exit signals for vectorbt.

        This is a simplified version - in production, this would integrate
        with the full SignalCalculator for CRI, Panic Score, etc.
        """
        try:
            # Calculate simple contrarian signals based on recent performance
            returns_5d = prices.pct_change(5)
            returns_20d = prices.pct_change(20)

            # Entry signal: Recent decline (contrarian)
            entry_signal = returns_5d < -0.05  # 5%+ decline in 5 days

            # Exit signal: Recent recovery or overbought
            exit_signal = (returns_5d > 0.03) | (prices > prices.rolling(50).mean() * 1.1)

            # Combine into single signal series (-1=short, 0=hold, 1=long)
            signals = pd.Series(0, index=prices.index)
            signals[entry_signal] = 1  # Long on decline
            signals[exit_signal] = 0   # Exit positions

            return signals

        except Exception as e:
            self.logger.debug(f"Failed to calculate basic signals for {symbol}: {e}")
            return pd.Series(0, index=prices.index)

    def _create_vectorbt_portfolio(self, price_data: pd.DataFrame, signals: pd.DataFrame,
                                 initial_capital: float):
        """
        Create vectorbt portfolio from signals and price data.

        Args:
            price_data: Multi-asset price DataFrame
            signals: Multi-asset signals DataFrame
            initial_capital: Starting capital

        Returns:
            Vectorbt portfolio object
        """
        try:
            # For multi-asset portfolios, we need to create individual portfolios
            # and combine them, or use vectorbt's multi-asset approach
            portfolios = []

            for symbol in price_data.columns:
                try:
                    # Get data for this symbol
                    symbol_prices = price_data[symbol]
                    symbol_signals = signals[symbol]

                    # Convert signals to boolean arrays with explicit types
                    entries = pd.Series(symbol_signals > 0, index=symbol_signals.index, dtype=bool)
                    exits = pd.Series(symbol_signals == 0, index=symbol_signals.index, dtype=bool)
                    short_entries = pd.Series(symbol_signals < 0, index=symbol_signals.index, dtype=bool)
                    short_exits = pd.Series(symbol_signals == 0, index=symbol_signals.index, dtype=bool)

                    # Ensure price data is float64
                    symbol_prices = pd.Series(symbol_prices, index=symbol_prices.index, dtype=float)

                    # Create portfolio for this symbol
                    symbol_portfolio = vbt.Portfolio.from_signals(
                        close=symbol_prices,
                        entries=entries,
                        exits=exits,
                        short_entries=short_entries,
                        short_exits=short_exits,
                        size=1.0,  # Full position size for this symbol
                        size_type='percent',
                        init_cash=initial_capital / len(price_data.columns),  # Split capital
                        fees=self.vbt_config['fees'],
                        slippage=self.vbt_config['slippage']
                    )

                    portfolios.append(symbol_portfolio)

                except Exception as e:
                    self.logger.warning(f"Failed to create portfolio for {symbol}: {e}")
                    continue

            if not portfolios:
                return None

            # For now, return the first portfolio (simplified approach)
            # In a full implementation, we'd combine portfolios properly
            return portfolios[0]

        except Exception as e:
            self.logger.error(f"Failed to create vectorbt portfolio: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _analyze_vectorbt_results(self, portfolio, start_date: str, end_date: str,
                                initial_capital: float) -> Dict[str, Any]:
        """
        Analyze vectorbt portfolio results.

        Args:
            portfolio: Vectorbt portfolio object
            start_date: Backtest start date
            end_date: Backtest end date
            initial_capital: Starting capital

        Returns:
            Dictionary with analysis results
        """
        try:
            # Get basic statistics
            stats = portfolio.stats()

            # Get returns and drawdowns
            returns = portfolio.returns()
            drawdown = portfolio.drawdown()

            # Get trade records
            trades = portfolio.trades.records if hasattr(portfolio.trades, 'records') else []

            # Calculate performance metrics
            final_value = portfolio.final_value()
            total_return = (final_value - initial_capital) / initial_capital
            max_drawdown = drawdown.min()

            # Calculate Sharpe ratio (assuming 0% risk-free rate)
            if len(returns) > 0 and returns.std() > 0:
                sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
            else:
                sharpe_ratio = 0.0

            # Calculate win rate
            winning_trades = sum(1 for trade in trades if trade['Return'] > 0) if trades else 0
            total_trades = len(trades) if trades else 0
            win_rate = winning_trades / total_trades if total_trades > 0 else 0

            # Create results dictionary
            results = {
                'backtest_type': 'vectorbt',
                'period': f"{start_date} to {end_date}",
                'initial_capital': initial_capital,
                'final_value': float(final_value),
                'total_return': float(total_return),
                'sharpe_ratio': float(sharpe_ratio),
                'max_drawdown': float(max_drawdown),
                'win_rate': float(win_rate),
                'total_trades': total_trades,
                'equity_curve': portfolio.value().tolist(),
                'returns': returns.tolist() if hasattr(returns, 'tolist') else [],
                'trades': [{
                    'entry_time': str(trade['Entry Date']) if 'Entry Date' in trade else None,
                    'exit_time': str(trade['Exit Date']) if 'Exit Date' in trade else None,
                    'symbol': trade.get('Symbol', 'Unknown'),
                    'entry_price': float(trade.get('Entry Price', 0)),
                    'exit_price': float(trade.get('Exit Price', 0)),
                    'pnl': float(trade.get('PnL', 0)),
                    'return_pct': float(trade.get('Return', 0))
                } for trade in trades] if trades else [],
                'vectorbt_stats': {
                    'total_return': stats.get('Total Return [%]', 0),
                    'sharpe_ratio': stats.get('Sharpe Ratio', 0),
                    'max_drawdown': stats.get('Max Drawdown [%]', 0),
                    'win_rate': stats.get('Win Rate [%]', 0),
                    'total_trades': stats.get('Total Trades', 0)
                }
            }

            return results

        except Exception as e:
            self.logger.error(f"Failed to analyze vectorbt results: {e}")
            return {'error': str(e)}

    def compare_with_custom_backtest(self, start_date: str, end_date: str,
                                   initial_capital: float = 100000) -> Dict[str, Any]:
        """
        Compare vectorbt results with custom backtest engine.

        Args:
            start_date: Start date
            end_date: End date
            initial_capital: Starting capital

        Returns:
            Comparison results
        """
        try:
            # Run vectorbt backtest
            vbt_result = self.run_vectorbt_backtest(start_date, end_date, initial_capital)

            # Run custom backtest (would need to import the original engine)
            from backtest import BacktestEngine
            custom_engine = BacktestEngine(self.config, self.data_manager,
                                         self.signal_calculator, self.risk_manager)
            custom_result = custom_engine.run_backtest(start_date, end_date, initial_capital)

            # Compare results
            comparison = {}
            if 'error' not in vbt_result and 'error' not in custom_result:
                vbt_return = vbt_result.get('total_return', 0)
                custom_return = custom_result.get('capital', {}).get('total_return', 0)

                comparison = {
                    'vectorbt_return': vbt_return,
                    'custom_return': custom_return,
                    'return_difference': vbt_return - custom_return,
                    'vectorbt_sharpe': vbt_result.get('sharpe_ratio', 0),
                    'custom_sharpe': custom_result.get('capital', {}).get('sharpe_ratio', 0),
                    'vectorbt_trades': vbt_result.get('total_trades', 0),
                    'custom_trades': custom_result.get('trade_statistics', {}).get('total_trades', 0)
                }

            return {
                'vectorbt_results': vbt_result,
                'custom_results': custom_result,
                'comparison': comparison
            }

        except Exception as e:
            self.logger.error(f"Backtest comparison failed: {e}")
            return {'error': str(e)}


# Global VectorbtBacktestEngine instance
_vectorbt_engine: Optional[VectorbtBacktestEngine] = None


def get_vectorbt_backtest_engine(config, data_manager, signal_calculator, risk_manager) -> Optional[VectorbtBacktestEngine]:
    """
    Get the global VectorbtBacktestEngine instance.

    Args:
        config: Configuration object
        data_manager: DataManager instance
        signal_calculator: SignalCalculator instance
        risk_manager: RiskManager instance

    Returns:
        VectorbtBacktestEngine instance or None if not available
    """
    global _vectorbt_engine
    if _vectorbt_engine is None and VECTORBT_AVAILABLE:
        _vectorbt_engine = VectorbtBacktestEngine(config, data_manager, signal_calculator, risk_manager)
    return _vectorbt_engine
