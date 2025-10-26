"""
Enhanced backtesting engine for Harvester II trading system.
Provides comprehensive historical simulation with realistic trade execution.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
import yfinance as yf
from pytrends.request import TrendReq
from pandas.tseries.holiday import USFederalHolidayCalendar

# Dependencies are now injected via constructor
from utils import calculate_performance_metrics


class BacktestEngine:
    """Comprehensive backtesting engine for Harvester II system."""
    
    def __init__(self, config, data_manager, signal_calculator, risk_manager):
        """Initialize backtesting engine with injected dependencies."""
        self.config = config
        self.data_manager = data_manager
        self.signal_calculator = signal_calculator
        self.risk_manager = risk_manager
        self.logger = logging.getLogger(__name__)
        
        # Backtest state
        self.initial_capital = 100000
        self.current_capital = 100000
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        self.daily_stats = []
        self.tradable_universe = []  # Initialize tradable universe
        
        # Performance tracking
        self.peak_equity = 100000
        self.max_drawdown = 0.0
        
        # Data cache for backtest
        self.price_data_cache = {}
        self.trends_data_cache = {}

        # Historical data paths
        self.historical_trends_path = self.config.get('backtesting.historical_trends_csv', None)
        
    def run_backtest(self, start_date: str, end_date: str, 
                    initial_capital: float = 100000) -> Dict[str, Any]:
        """
        Run comprehensive backtest of the trading system.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            initial_capital: Starting capital
            
        Returns:
            Dictionary with detailed backtest results
        """
        try:
            self.logger.info(f"Starting backtest: {start_date} to {end_date}")
            
            # Initialize backtest state
            self.initial_capital = initial_capital
            self.current_capital = initial_capital
            self.peak_equity = initial_capital
            self.positions = {}
            self.trades = []
            self.equity_curve = []
            self.daily_stats = []
            
            # Pre-load all required data
            self._preload_historical_data(start_date, end_date)
            
            # Generate trading dates (weekdays only)
            trading_dates = self._generate_trading_dates(start_date, end_date)
            
            # Run simulation day by day
            for i, current_date in enumerate(trading_dates):
                self._simulate_trading_day(current_date, i)
            
            # Calculate final results
            results = self._calculate_backtest_results(start_date, end_date)
            
            self.logger.info("Backtest completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Backtest failed: {e}")
            return {'error': str(e)}
    
    def _preload_historical_data(self, start_date: str, end_date: str) -> None:
        """Pre-load all historical data needed for backtest."""
        try:
            self.logger.info("Pre-loading historical data...")
            
            # Get universe of assets
            universe = self.config.get('universe.assets', [])
            
            # Load price data for all assets
            for symbol in universe:
                try:
                    # Get extended data to ensure we have enough for calculations
                    extended_start = pd.to_datetime(start_date) - timedelta(days=365)
                    
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(
                        start=extended_start.strftime('%Y-%m-%d'),
                        end=end_date,
                        auto_adjust=True
                    )
                    
                    if not data.empty:
                        # Calculate technical indicators
                        data = self._calculate_technical_indicators(data)
                        self.price_data_cache[symbol] = data
                        self.logger.debug(f"Loaded {len(data)} days of data for {symbol}")
                    else:
                        self.logger.warning(f"No data available for {symbol}")
                        
                except Exception as e:
                    self.logger.warning(f"Failed to load data for {symbol}: {e}")
            
            # Load Google Trends data (simplified - would need historical trends data)
            self._preload_trends_data(universe, start_date, end_date)
            
            self.logger.info(f"Pre-loaded data for {len(self.price_data_cache)} assets")
            
        except Exception as e:
            self.logger.error(f"Failed to preload historical data: {e}")
    
    def _preload_trends_data(self, universe: List[str], start_date: str, end_date: str) -> None:
        """Pre-load Google Trends data (simulated implementation for backtesting)."""
        try:
            self.logger.info("Generating simulated Google Trends data for backtesting...")
            
            # Generate realistic trends data based on price patterns
            for symbol in universe:
                if symbol in self.price_data_cache:
                    price_data = self.price_data_cache[symbol]
                    
                    # Create realistic trends simulation
                    trends_simulated = self._generate_realistic_trends_data(price_data)
                    
                    trends_df = pd.DataFrame({
                        'value': trends_simulated
                    }, index=price_data.index)
                    
                    self.trends_data_cache[symbol] = trends_df
                    self.logger.debug(f"Generated trends data for {symbol}")
                    
        except Exception as e:
            self.logger.warning(f"Failed to preload trends data: {e}")
    
    def _generate_realistic_trends_data(self, price_data: pd.DataFrame) -> pd.Series:
        """Generate realistic Google Trends simulation based on price data."""
        try:
            # Base trends level
            base_trends = pd.Series(50, index=price_data.index)
            
            # Add volatility-based spikes
            volatility = price_data['Close'].pct_change().rolling(14).std()
            volatility_spikes = (volatility * 200).fillna(0)
            
            # Add volume-based interest
            volume_ma = price_data['Volume'].rolling(14).mean()
            volume_ratio = price_data['Volume'] / volume_ma
            volume_interest = ((volume_ratio - 1) * 20).fillna(0)
            
            # Add momentum-based trends
            momentum = price_data['Close'].pct_change(5).fillna(0)
            momentum_trends = (abs(momentum) * 100).fillna(0)
            
            # Combine all factors
            trends = base_trends + volatility_spikes + volume_interest + momentum_trends
            
            # Add some random noise for realism
            noise = np.random.normal(0, 5, len(trends))
            trends += noise
            
            # Ensure values are in reasonable range (0-100)
            trends = trends.clip(0, 100)
            
            # Smooth the data
            trends = trends.rolling(3, center=True).mean().fillna(trends)
            
            return trends
            
        except Exception as e:
            self.logger.error(f"Failed to generate trends data: {e}")
            # Return simple volatility-based trends as fallback
            volatility = price_data['Close'].pct_change().rolling(14).std()
            return (volatility * 100).fillna(50).clip(0, 100)
    
    def _calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for price data."""
        try:
            result = data.copy()
            
            # ATR
            high_low = data['High'] - data['Low']
            high_close = np.abs(data['High'] - data['Close'].shift())
            low_close = np.abs(data['Low'] - data['Close'].shift())
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            result['ATR'] = true_range.rolling(window=14).mean()
            
            # Volume moving average
            result['Volume_MA'] = data['Volume'].rolling(window=14).mean()
            
            # Returns
            result['Returns'] = data['Close'].pct_change()
            result['Returns_5d'] = data['Close'].pct_change(5)
            
            # Volatility
            result['Volatility'] = result['Returns'].rolling(window=14).std()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to calculate technical indicators: {e}")
            return data
    
    def _generate_trading_dates(self, start_date: str, end_date: str) -> List[datetime]:
        """Generate list of trading dates (weekdays only, excluding holidays)."""
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        date_range = pd.date_range(start=start, end=end, freq='D')

        # Filter for weekdays only (Monday=0, Friday=4)
        weekdays = [d for d in date_range if d.weekday() < 5]

        # Get US federal holidays
        calendar = USFederalHolidayCalendar()
        holidays = calendar.holidays(start=start, end=end)

        # Filter out holidays
        trading_dates = [d for d in weekdays if d not in holidays]

        self.logger.info(f"Generated {len(trading_dates)} trading dates from {start_date} to {end_date} "
                        f"(excluded {len(weekdays) - len(trading_dates)} holidays)")

        return trading_dates
    
    def _simulate_trading_day(self, current_date: datetime, day_index: int) -> None:
        """Simulate trading for a single day."""
        try:
            # Update tradable universe weekly
            if day_index % 7 == 0:
                self._update_tradable_universe_backtest(current_date)
            
            # Process exit signals first
            self._process_exit_signals_backtest(current_date)

            # Update trailing stops if enabled
            if hasattr(self.risk_manager, 'trailing_stops_enabled') and self.risk_manager.trailing_stops_enabled:
                # Get current prices and ATR values for trailing stops update
                current_prices = {}
                atr_values = {}

                for symbol in self.positions.keys():
                    price_data = self.price_data_cache.get(symbol)
                    if price_data is not None:
                        try:
                            price_index_naive = price_data.index.tz_localize(None) if price_data.index.tz else price_data.index
                            price_subset = price_data[price_index_naive <= current_date]
                            if not price_subset.empty:
                                current_prices[symbol] = price_subset['Close'].iloc[-1]
                                if 'ATR' in price_subset.columns and not price_subset['ATR'].isna().all():
                                    atr_values[symbol] = price_subset['ATR'].iloc[-1]
                        except Exception as e:
                            self.logger.debug(f"Error getting price/ATR for trailing stops {symbol}: {e}")

                # Update trailing stops
                self.risk_manager.update_trailing_stops(current_prices, atr_values)

            # Check if we can open new positions
            if len(self.positions) >= self.config.get('risk_management.max_open_positions', 4):
                self._record_daily_stats(current_date)
                return

            # Calculate G-Score for macro risk
            g_score = self._calculate_g_score_backtest(current_date)
            
            # Get entry signals
            entry_signals = self._get_entry_signals_backtest(current_date)
            
            # Process entry signals
            for signal in entry_signals:
                if len(self.positions) >= self.config.get('risk_management.max_open_positions', 4):
                    break
                
                self._execute_entry_backtest(signal, current_date, g_score)
            
            # Record daily statistics
            self._record_daily_stats(current_date)
            
        except Exception as e:
            self.logger.error(f"Error simulating trading day {current_date}: {e}")
    
    def _update_tradable_universe_backtest(self, current_date: datetime) -> None:
        """Update tradable universe for backtest."""
        try:
            universe = self.config.get('universe.assets', [])
            tradable_assets = []
            
            for symbol in universe:
                if symbol not in self.price_data_cache:
                    continue
                
                price_data = self.price_data_cache[symbol]
                trends_data = self.trends_data_cache.get(symbol, pd.DataFrame())
                
                # Filter data up to current date (handle timezone issues)
                try:
                    # Convert timezone-aware index to naive for comparison
                    price_index_naive = price_data.index.tz_localize(None) if price_data.index.tz else price_data.index
                    price_subset = price_data[price_index_naive <= current_date]
                    
                    if not trends_data.empty:
                        trends_index_naive = trends_data.index.tz_localize(None) if trends_data.index.tz else trends_data.index
                        trends_subset = trends_data[trends_index_naive <= current_date]
                    else:
                        trends_subset = pd.DataFrame()
                except Exception as e:
                    self.logger.warning(f"Date filtering error for {symbol}: {e}")
                    continue
                
                if len(price_subset) < 90:  # Need minimum data for CRI calculation
                    continue
                
                # Calculate CRI
                cri = self._calculate_cri_backtest(symbol, price_subset, trends_subset)
                
                if cri >= self.config.get('universe.cri_threshold', 0.4):
                    tradable_assets.append(symbol)
            
            self.tradable_universe = tradable_assets
            
        except Exception as e:
            self.logger.error(f"Failed to update tradable universe: {e}")
    
    def _calculate_cri_backtest(self, symbol: str, price_data: pd.DataFrame, 
                               trends_data: pd.DataFrame) -> float:
        """Calculate CRI for backtest."""
        try:
            if price_data.empty or trends_data.empty:
                return 0.0
            
            # Align data by date
            aligned_data = self._align_data_by_date_backtest(price_data, trends_data)
            if aligned_data.empty:
                return 0.0
            
            # Calculate daily price changes
            price_changes = aligned_data['price'].pct_change().dropna()
            trends_changes = aligned_data['trends'].pct_change().dropna()
            
            # Ensure same length
            min_length = min(len(price_changes), len(trends_changes))
            if min_length < 10:
                return 0.0
            
            price_changes = price_changes.iloc[-min_length:]
            trends_changes = trends_changes.iloc[-min_length:]
            
            # Calculate correlation
            correlation = price_changes.corr(trends_changes)
            
            return abs(correlation) if not pd.isna(correlation) else 0.0
            
        except Exception as e:
            self.logger.error(f"Failed to calculate CRI for {symbol}: {e}")
            return 0.0
    
    def _align_data_by_date_backtest(self, price_data: pd.DataFrame, 
                                    trends_data: pd.DataFrame) -> pd.DataFrame:
        """Align price and trends data by date for backtest."""
        try:
            # Find common dates
            common_dates = set(price_data.index.date) & set(trends_data.index.date)
            
            if not common_dates:
                return pd.DataFrame()
            
            # Create aligned DataFrame
            aligned_data = []
            for date in sorted(common_dates):
                price_idx = price_data.index.date.tolist().index(date)
                trends_idx = trends_data.index.date.tolist().index(date)
                
                aligned_data.append({
                    'date': date,
                    'price': price_data['Close'].iloc[price_idx],
                    'trends': trends_data['value'].iloc[trends_idx]
                })
            
            return pd.DataFrame(aligned_data).set_index('date')
            
        except Exception as e:
            self.logger.error(f"Failed to align data by date: {e}")
            return pd.DataFrame()

    def _load_historical_trends_data(self, keyword: str) -> Optional[pd.DataFrame]:
        """
        Load historical trends data from CSV file if available.

        Args:
            keyword: Search keyword for trends data

        Returns:
            DataFrame with historical trends data or None if not available
        """
        if not self.historical_trends_path:
            return None

        try:
            # Load CSV file
            trends_df = pd.read_csv(self.historical_trends_path)

            # Validate required columns
            if 'date' not in trends_df.columns or keyword not in trends_df.columns:
                self.logger.warning(f"Historical trends CSV missing required columns: date, {keyword}")
                return None

            # Convert date column and set as index
            trends_df['date'] = pd.to_datetime(trends_df['date'])
            trends_df = trends_df.set_index('date')

            # Extract data for the specific keyword
            keyword_data = trends_df[keyword].copy()
            keyword_data.name = 'value'

            # Ensure we have a proper DataFrame structure
            result_df = keyword_data.to_frame()

            self.logger.info(f"Loaded historical trends data for {keyword}: {len(result_df)} records")
            return result_df

        except FileNotFoundError:
            self.logger.warning(f"Historical trends CSV file not found: {self.historical_trends_path}")
            return None
        except Exception as e:
            self.logger.error(f"Error loading historical trends data: {e}")
            return None

    def _get_trends_data_backtest(self, keyword: str, current_date: datetime) -> pd.DataFrame:
        """
        Get trends data for backtest, using historical CSV if available, otherwise pytrends.

        Args:
            keyword: Search keyword
            current_date: Current backtest date

        Returns:
            DataFrame with trends data up to current_date
        """
        # Try loading historical data first
        historical_data = self._load_historical_trends_data(keyword)
        if historical_data is not None:
            # Filter data up to current date
            try:
                historical_filtered = historical_data[historical_data.index <= current_date]
                if not historical_filtered.empty:
                    return historical_filtered
            except Exception as e:
                self.logger.warning(f"Error filtering historical trends data: {e}")

        # Fall back to pytrends if historical data not available
        try:
            trends_data = self.data_manager.get_google_trends(keyword, timeframe="today 12-m")
            if trends_data is not None and not trends_data.empty:
                return trends_data
        except Exception as e:
            self.logger.error(f"Error fetching trends data from pytrends: {e}")

        # Return empty DataFrame if no data available
        return pd.DataFrame()

    def _calculate_g_score_backtest(self, current_date: datetime) -> float:
        """Calculate G-Score for backtest."""
        try:
            score = 0.0
            
            # VIX check - try multiple symbols
            vix_symbols = ['VIX', '^VIX', 'VIXCLS']  # Different VIX symbols
            vix_found = False
            
            for vix_symbol in vix_symbols:
                vix_data = self.price_data_cache.get(vix_symbol)
                if vix_data is not None:
                    try:
                        vix_index_naive = vix_data.index.tz_localize(None) if vix_data.index.tz else vix_data.index
                        vix_subset = vix_data[vix_index_naive <= current_date]
                        if not vix_subset.empty:
                            current_vix = vix_subset['Close'].iloc[-1]
                            if current_vix > self.config.get('macro_risk.indicators.vix_threshold', 25):
                                score += 1
                            vix_found = True
                            break
                    except Exception as e:
                        self.logger.debug(f"VIX data filtering error for {vix_symbol}: {e}")
                        continue
            
            # If no VIX data available, use SPY volatility as proxy
            if not vix_found:
                spy_data = self.price_data_cache.get('SPY')
                if spy_data is not None:
                    try:
                        spy_index_naive = spy_data.index.tz_localize(None) if spy_data.index.tz else spy_data.index
                        spy_subset = spy_data[spy_index_naive <= current_date]
                        if len(spy_subset) >= 14:
                            spy_volatility = spy_subset['Close'].pct_change().rolling(14).std().iloc[-1]
                            # Convert volatility to VIX-like scale (multiply by ~20)
                            vix_proxy = spy_volatility * 20
                            if vix_proxy > self.config.get('macro_risk.indicators.vix_threshold', 25):
                                score += 1
                    except Exception as e:
                        self.logger.debug(f"SPY volatility calculation error: {e}")
            
            # SPY 7-day return check
            spy_data = self.price_data_cache.get('SPY')
            if spy_data is not None:
                try:
                    spy_index_naive = spy_data.index.tz_localize(None) if spy_data.index.tz else spy_data.index
                    spy_subset = spy_data[spy_index_naive <= current_date]
                    if len(spy_subset) >= 7:
                        spy_7d_return = (spy_subset['Close'].iloc[-1] / spy_subset['Close'].iloc[-8]) - 1
                        if spy_7d_return < self.config.get('macro_risk.indicators.spy_return_threshold', -0.05):
                            score += 1
                except Exception as e:
                    self.logger.debug(f"SPY return calculation error: {e}")
            
            # Oil 7-day return check
            oil_data = self.price_data_cache.get('USO')
            if oil_data is not None:
                try:
                    oil_index_naive = oil_data.index.tz_localize(None) if oil_data.index.tz else oil_data.index
                    oil_subset = oil_data[oil_index_naive <= current_date]
                    if len(oil_subset) >= 7:
                        oil_7d_return = (oil_subset['Close'].iloc[-1] / oil_subset['Close'].iloc[-8]) - 1
                        if oil_7d_return > self.config.get('macro_risk.indicators.oil_return_threshold', 0.10):
                            score += 1
                except Exception as e:
                    self.logger.debug(f"Oil return calculation error: {e}")
            
            return score
            
        except Exception as e:
            self.logger.error(f"Failed to calculate G-Score: {e}")
            return 0.0
    
    def _get_entry_signals_backtest(self, current_date: datetime) -> List[Dict[str, Any]]:
        """Get entry signals for backtest."""
        signals = []
        
        try:
            for symbol in self.tradable_universe:
                if symbol in self.positions:
                    continue
                
                price_data = self.price_data_cache.get(symbol)
                trends_data = self._get_trends_data_backtest(symbol, current_date)
                
                if price_data is None:
                    continue
                
                # Filter data up to current date (handle timezone issues)
                try:
                    price_index_naive = price_data.index.tz_localize(None) if price_data.index.tz else price_data.index
                    price_subset = price_data[price_index_naive <= current_date]
                    
                    if not trends_data.empty:
                        trends_index_naive = trends_data.index.tz_localize(None) if trends_data.index.tz else trends_data.index
                        trends_subset = trends_data[trends_index_naive <= current_date]
                    else:
                        trends_subset = pd.DataFrame()
                except Exception as e:
                    self.logger.debug(f"Date filtering error for {symbol}: {e}")
                    continue
                
                if len(price_subset) < 90:
                    continue
                
                # Calculate Panic Score
                panic_score = self._calculate_panic_score_backtest(symbol, price_subset, trends_subset)
                
                if panic_score > self.config.get('signals.panic_threshold', 3.0):
                    # Determine trade direction
                    if len(price_subset) >= 5:
                        price_change_5d = (price_subset['Close'].iloc[-1] / price_subset['Close'].iloc[-6]) - 1
                    else:
                        price_change_5d = 0.0
                    
                    signal = {
                        'symbol': symbol,
                        'panic_score': panic_score,
                        'price_change_5d': price_change_5d,
                        'side': 'BUY' if price_change_5d < 0 else 'SELL',
                        'current_price': price_subset['Close'].iloc[-1],
                        'timestamp': current_date
                    }
                    
                    signals.append(signal)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Failed to get entry signals: {e}")
            return []
    
    def _calculate_panic_score_backtest(self, symbol: str, price_data: pd.DataFrame, 
                                       trends_data: pd.DataFrame) -> float:
        """Calculate Panic Score for backtest."""
        try:
            if price_data.empty:
                return 0.0
            
            # Get recent data for z-score calculation
            lookback_window = self.config.get('system.lookback_window', 90)
            recent_data = price_data.tail(lookback_window)
            
            if len(recent_data) < 30:
                return 0.0
            
            # Calculate z-scores
            volatility_z = self._calculate_z_score(
                recent_data['ATR'].iloc[-1], 
                recent_data['ATR']
            )
            
            volume_z = self._calculate_z_score(
                recent_data['Volume'].iloc[-1], 
                recent_data['Volume']
            )
            
            # Trends z-score
            trends_z = 0.0
            if not trends_data.empty:
                recent_trends = trends_data.tail(lookback_window)
                if len(recent_trends) >= 30:
                    trends_z = self._calculate_z_score(
                        recent_trends['value'].iloc[-1],
                        recent_trends['value']
                    )
            
            # Sum z-scores for Panic Score
            panic_score = volatility_z + volume_z + trends_z
            
            return panic_score
            
        except Exception as e:
            self.logger.error(f"Failed to calculate Panic Score for {symbol}: {e}")
            return 0.0
    
    def _calculate_z_score(self, current_value: float, series: pd.Series) -> float:
        """Calculate z-score for current value against historical data."""
        try:
            if len(series) < 2:
                return 0.0
            
            mean_val = series.mean()
            std_val = series.std()
            
            if std_val == 0:
                return 0.0
            
            z_score = (current_value - mean_val) / std_val
            return z_score if not pd.isna(z_score) else 0.0
            
        except Exception as e:
            self.logger.error(f"Failed to calculate z-score: {e}")
            return 0.0
    
    def _execute_entry_backtest(self, signal: Dict[str, Any], current_date: datetime, 
                               g_score: float) -> None:
        """Execute entry order for backtest."""
        try:
            symbol = signal['symbol']
            entry_price = signal['current_price']
            side = signal['side']
            
            # Get ATR for position sizing
            price_data = self.price_data_cache[symbol]
            try:
                price_index_naive = price_data.index.tz_localize(None) if price_data.index.tz else price_data.index
                price_subset = price_data[price_index_naive <= current_date]
            except Exception as e:
                self.logger.debug(f"Date filtering error for {symbol}: {e}")
                return
            
            if len(price_subset) < 14:
                return
            
            atr = price_subset['ATR'].iloc[-1]
            if pd.isna(atr) or atr <= 0:
                return
            
            # Calculate position size
            base_position_fraction = self.config.get('risk_management.base_position_fraction', 0.005)
            
            # Apply macro risk multiplier
            if g_score >= self.config.get('macro_risk.g_score_threshold', 2):
                risk_multiplier = self.config.get('macro_risk.position_size_multiplier.high_risk', 0.5)
            else:
                risk_multiplier = self.config.get('macro_risk.position_size_multiplier.normal_risk', 1.0)
            
            position_value = self.current_capital * base_position_fraction * risk_multiplier
            
            # Apply position size limits
            min_position_size = self.config.get('risk_management.position_sizing.min_position_size', 100)
            max_position_size = self.config.get('risk_management.position_sizing.max_position_size', 5000)
            
            position_value = max(min_position_size, min(position_value, max_position_size))
            
            # Apply slippage to entry price (realistic trading costs)
            slippage_pct = self.config.get('trading.execution.slippage_tolerance', 0.001)
            slippage_adjustment = entry_price * slippage_pct
            fill_price = entry_price + slippage_adjustment if side == 'BUY' else entry_price - slippage_adjustment

            # Apply commission per trade
            commission_per_trade = self.config.get('trading.execution.commission_per_trade', 1.0)

            # Calculate shares
            shares = int(position_value / fill_price)
            actual_position_value = shares * fill_price + commission_per_trade
            
            # Calculate stop loss and profit target
            atr_stop_multiplier = self.config.get('risk_management.atr_stop_loss_multiplier', 1.0)
            atr_profit_multiplier = self.config.get('risk_management.atr_profit_target_multiplier', 2.0)
            
            if side == 'BUY':
                stop_loss_price = entry_price - (atr * atr_stop_multiplier)
                profit_target_price = entry_price + (atr * atr_profit_multiplier)
            else:  # SELL
                stop_loss_price = entry_price + (atr * atr_stop_multiplier)
                profit_target_price = entry_price - (atr * atr_profit_multiplier)
            
            # Create position
            position = {
                'symbol': symbol,
                'shares': shares,
                'entry_price': entry_price,
                'fill_price': fill_price,
                'commission': commission_per_trade,
                'slippage': slippage_adjustment,
                'entry_date': current_date,
                'side': side,
                'stop_loss_price': stop_loss_price,
                'profit_target_price': profit_target_price,
                'atr': atr,
                'g_score': g_score,
                'position_value': actual_position_value
            }

            # Add position and deduct capital (including commissions)
            self.positions[symbol] = position
            self.current_capital -= actual_position_value

            self.logger.debug(f"Opened {side} position for {symbol}: {shares} shares at ${fill_price:.2f} "
                            f"(slippage: ${slippage_adjustment:.2f}, commission: ${commission_per_trade:.2f})")
            
        except Exception as e:
            self.logger.error(f"Failed to execute entry for {signal['symbol']}: {e}")
    
    def _process_exit_signals_backtest(self, current_date: datetime) -> None:
        """Process exit signals for backtest."""
        try:
            positions_to_close = []
            
            for symbol, position in self.positions.items():
                # Get current price
                price_data = self.price_data_cache[symbol]
                try:
                    price_index_naive = price_data.index.tz_localize(None) if price_data.index.tz else price_data.index
                    price_subset = price_data[price_index_naive <= current_date]
                except Exception as e:
                    self.logger.debug(f"Date filtering error for {symbol}: {e}")
                    continue
                
                if price_subset.empty:
                    continue
                
                current_price = price_subset['Close'].iloc[-1]
                
                # Check exit conditions
                should_exit = False
                exit_reason = ""
                
                if position['side'] == 'BUY':
                    if current_price <= position['stop_loss_price']:
                        should_exit = True
                        exit_reason = "stop_loss"
                    elif current_price >= position['profit_target_price']:
                        should_exit = True
                        exit_reason = "profit_target"
                else:  # SELL
                    if current_price >= position['stop_loss_price']:
                        should_exit = True
                        exit_reason = "stop_loss"
                    elif current_price <= position['profit_target_price']:
                        should_exit = True
                        exit_reason = "profit_target"
                
                if should_exit:
                    positions_to_close.append((symbol, current_price, exit_reason))
            
            # Close positions
            for symbol, exit_price, exit_reason in positions_to_close:
                self._close_position_backtest(symbol, exit_price, exit_reason, current_date)
                
        except Exception as e:
            self.logger.error(f"Failed to process exit signals: {e}")
    
    def _close_position_backtest(self, symbol: str, exit_price: float, 
                                exit_reason: str, current_date: datetime) -> None:
        """Close position for backtest."""
        try:
            if symbol not in self.positions:
                return
            
            position = self.positions[symbol]
            
            # Apply slippage to exit price (realistic trading costs)
            slippage_pct = self.config.get('trading.execution.slippage_tolerance', 0.001)
            slippage_adjustment = exit_price * slippage_pct

            # Apply commission per trade
            commission_per_trade = self.config.get('trading.execution.commission_per_trade', 1.0)

            # Calculate P&L with realistic execution
            shares = position['shares']
            entry_price = position['entry_price']
            entry_value = position['position_value']

            # Exit price after slippage
            fill_exit_price = exit_price - slippage_adjustment if position['side'] == 'BUY' else exit_price + slippage_adjustment

            exit_value = shares * fill_exit_price - commission_per_trade  # Deduct commission
            pnl = exit_value - entry_value
            pnl_percentage = pnl / entry_value

            # Update capital
            self.current_capital += exit_value
            
            # Create trade record
            trade_record = {
                'symbol': symbol,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'fill_exit_price': fill_exit_price,
                'exit_slippage': slippage_adjustment,
                'exit_commission': commission_per_trade,
                'shares': shares,
                'entry_date': position['entry_date'],
                'exit_date': current_date,
                'entry_value': entry_value,
                'exit_value': exit_value,
                'pnl': pnl,
                'pnl_percentage': pnl_percentage,
                'exit_reason': exit_reason,
                'duration_days': (current_date - position['entry_date']).days,
                'atr': position['atr'],
                'g_score': position['g_score']
            }
            
            # Add to trades
            self.trades.append(trade_record)
            
            # Remove position
            del self.positions[symbol]
            
            self.logger.debug(f"Closed {symbol}: ${pnl:.2f} P&L ({pnl_percentage:.2%}) - {exit_reason} "
                            f"(exit price: ${fill_exit_price:.2f}, slippage: ${slippage_adjustment:.2f}, commission: ${commission_per_trade:.2f})")
            
        except Exception as e:
            self.logger.error(f"Failed to close position {symbol}: {e}")
    
    def _record_daily_stats(self, current_date: datetime) -> None:
        """Record daily statistics."""
        try:
            # Calculate total portfolio value
            total_position_value = sum(pos['position_value'] for pos in self.positions.values())
            total_equity = self.current_capital + total_position_value
            
            # Update peak equity and drawdown
            if total_equity > self.peak_equity:
                self.peak_equity = total_equity
            
            current_drawdown = (self.peak_equity - total_equity) / self.peak_equity
            if current_drawdown > self.max_drawdown:
                self.max_drawdown = current_drawdown
            
            # Record daily stats
            daily_stat = {
                'date': current_date,
                'equity': total_equity,
                'cash': self.current_capital,
                'positions_value': total_position_value,
                'open_positions': len(self.positions),
                'drawdown': current_drawdown
            }
            
            self.daily_stats.append(daily_stat)
            self.equity_curve.append({
                'date': current_date,
                'equity': total_equity
            })
            
        except Exception as e:
            self.logger.error(f"Failed to record daily stats: {e}")
    
    def _calculate_backtest_results(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """Calculate comprehensive backtest results."""
        try:
            if not self.equity_curve:
                return {'error': 'No equity curve data'}
            
            equity_df = pd.DataFrame(self.equity_curve)
            equity_df['date'] = pd.to_datetime(equity_df['date'])
            equity_df = equity_df.set_index('date')
            
            # Basic metrics
            total_return = (equity_df['equity'].iloc[-1] - self.initial_capital) / self.initial_capital
            
            # Calculate performance metrics
            performance_metrics = calculate_performance_metrics(equity_df['equity'])
            
            # Trade analysis
            trades_df = pd.DataFrame(self.trades) if self.trades else pd.DataFrame()
            
            trade_stats = {}
            if not trades_df.empty:
                trade_stats = {
                    'total_trades': len(trades_df),
                    'winning_trades': len(trades_df[trades_df['pnl'] > 0]),
                    'losing_trades': len(trades_df[trades_df['pnl'] <= 0]),
                    'win_rate': len(trades_df[trades_df['pnl'] > 0]) / len(trades_df),
                    'avg_trade_pnl': trades_df['pnl'].mean(),
                    'avg_winning_trade': trades_df[trades_df['pnl'] > 0]['pnl'].mean() if len(trades_df[trades_df['pnl'] > 0]) > 0 else 0,
                    'avg_losing_trade': trades_df[trades_df['pnl'] <= 0]['pnl'].mean() if len(trades_df[trades_df['pnl'] <= 0]) > 0 else 0,
                    'avg_trade_duration': trades_df['duration_days'].mean(),
                    'profit_factor': abs(trades_df[trades_df['pnl'] > 0]['pnl'].sum() / trades_df[trades_df['pnl'] <= 0]['pnl'].sum()) if trades_df[trades_df['pnl'] <= 0]['pnl'].sum() != 0 else float('inf')
                }
            
            # Results
            results = {
                'backtest_period': {
                    'start_date': start_date,
                    'end_date': end_date,
                    'trading_days': len(equity_df)
                },
                'capital': {
                    'initial_capital': self.initial_capital,
                    'final_capital': equity_df['equity'].iloc[-1],
                    'total_return': total_return,
                    'max_drawdown': self.max_drawdown
                },
                'performance_metrics': performance_metrics,
                'trade_statistics': trade_stats,
                'equity_curve': self.equity_curve,
                'trades': self.trades,
                'daily_stats': self.daily_stats
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to calculate backtest results: {e}")
            return {'error': str(e)}


# Global backtest engine instance
def get_backtest_engine() -> BacktestEngine:
    """Create and return a new backtest engine instance with dependencies."""
    from .di import create_components
    components = create_components()
    return BacktestEngine(components['config'], components['data_manager'], components['signal_calculator'], components['risk_manager'])
