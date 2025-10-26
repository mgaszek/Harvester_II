"""
Data management module for Harvester II trading system.
Handles fetching and caching of price data, Google Trends, and macro indicators.
"""

import asyncio
import logging

from alpha_vantage.timeseries import TimeSeries
from cachetools import TTLCache
import numpy as np
import pandas as pd
from pytrends.request import TrendReq
import requests
import yfinance as yf

# Config is now injected via constructor
from models import MacroCache, PriceCache, TrendsCache, get_data_db_manager


class DataManager:
    """Manages all data fetching and caching for the trading system."""

    def _validate_symbol(self, symbol: str) -> None:
        """
        Validate symbol format.

        Args:
            symbol: Asset symbol to validate

        Raises:
            ValueError: If symbol is invalid
        """
        if not isinstance(symbol, str):
            raise ValueError(f"Symbol must be a string, got {type(symbol)}")
        if not symbol:
            raise ValueError("Symbol cannot be empty")
        if symbol != symbol.upper():
            raise ValueError(f"Symbol must be uppercase, got {symbol}")
        if len(symbol) > 10:
            raise ValueError(f"Symbol too long, got {len(symbol)} characters")
        # Basic validation for common characters
        import re

        if not re.match(r"^[A-Z0-9.^]+$", symbol):
            raise ValueError(f"Symbol contains invalid characters, got {symbol}")

    def __init__(self, config):
        """Initialize data manager with injected configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize data sources
        self._init_yfinance()
        self._init_google_trends()
        self._init_alpha_vantage()
        self._init_database()

        # TTL Cache for frequently accessed data (max 100 items, 15min TTL)
        self._price_cache = TTLCache(maxsize=100, ttl=900)  # 15 minutes
        self._trends_cache = TTLCache(maxsize=50, ttl=3600)  # 1 hour
        self._macro_cache = TTLCache(maxsize=20, ttl=1800)  # 30 minutes

        # TTL Cache for Bayesian posterior probabilities (5min TTL for performance)
        self._posterior_cache = TTLCache(maxsize=50, ttl=300)  # 5 minutes

        # Timestamp tracking for data latency monitoring
        self._trends_timestamps = {}

    def _init_yfinance(self) -> None:
        """Initialize Yahoo Finance client."""
        try:
            # yfinance doesn't need explicit initialization
            self.yf_available = True
            self.logger.info("Yahoo Finance client initialized")
        except (ImportError, ModuleNotFoundError) as e:
            self.yf_available = False
            self.logger.error(f"Failed to import Yahoo Finance module: {e}")
        except Exception as e:
            self.yf_available = False
            self.logger.error(f"Unexpected error initializing Yahoo Finance: {e}")

    def _init_google_trends(self) -> None:
        """Initialize Google Trends client."""
        try:
            self.pytrends = TrendReq(hl="en-US", tz=360)
            self.trends_available = True
            self.logger.info("Google Trends client initialized")
        except (requests.RequestException, ConnectionError) as e:
            self.trends_available = False
            self.logger.error(f"Failed to connect to Google Trends API: {e}")
        except Exception as e:
            self.trends_available = False
            self.logger.error(f"Unexpected error initializing Google Trends: {e}")

    def _init_alpha_vantage(self) -> None:
        """Initialize Alpha Vantage client."""
        try:
            api_key = self.config.get_env("ALPHA_VANTAGE_API_KEY")
            if api_key:
                self.av_client = TimeSeries(key=api_key, output_format="pandas")
                self.av_available = True
                self.logger.info("Alpha Vantage client initialized")
            else:
                self.av_available = False
                self.logger.warning("Alpha Vantage API key not provided")
        except (ValueError, KeyError) as e:
            self.av_available = False
            self.logger.error(f"Invalid Alpha Vantage configuration: {e}")
        except Exception as e:
            self.av_available = False
            self.logger.error(f"Unexpected error initializing Alpha Vantage: {e}")

    def _init_database(self) -> None:
        """Initialize SQLAlchemy database for caching."""
        try:
            self.db_manager = get_data_db_manager()
            self.logger.info("Database initialized with SQLAlchemy")
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            self.db_manager = None

    def get_price_data(
        self, symbol: str, period: str = "1y", use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Get price data for a symbol.

        Args:
            symbol: Stock/crypto symbol
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            use_cache: Whether to use cached data

        Returns:
            DataFrame with OHLCV data
        """
        self._validate_symbol(symbol)

        cache_key = f"{symbol}_{period}"

        # Check cache first (TTLCache handles expiration automatically)
        if use_cache and cache_key in self._price_cache:
            return self._price_cache[cache_key]

        try:
            if self.yf_available:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period)

                if data.empty:
                    self.logger.warning("No data returned for %s", symbol)
                    return pd.DataFrame()

                # Cache the data
                self._price_cache[cache_key] = data
                self._save_to_db("price_cache", symbol, data)

                self.logger.info(
                    "Fetched price data for %s: %d records", symbol, len(data)
                )
                return data

            self.logger.error("Yahoo Finance not available")
            return pd.DataFrame()

        except (requests.RequestException, ConnectionError):
            self.logger.exception("Network error fetching price data for %s", symbol)
            return pd.DataFrame()
        except ValueError:
            self.logger.exception("Invalid data received for %s", symbol)
            return pd.DataFrame()
        except Exception:
            self.logger.exception("Unexpected error fetching price data for %s", symbol)
            return pd.DataFrame()

    async def get_price_data_async(
        self, symbol: str, period: str = "1y", use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Async version: Get price data for a symbol.

        Args:
            symbol: Stock/crypto symbol
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            use_cache: Whether to use cached data

        Returns:
            DataFrame with OHLCV data
        """
        self._validate_symbol(symbol)

        cache_key = f"{symbol}_{period}"

        # Check cache first (TTLCache handles expiration automatically)
        if use_cache and cache_key in self._price_cache:
            return self._price_cache[cache_key]

        try:
            if self.yf_available:
                # Use asyncio.to_thread to run the synchronous yfinance call in a thread
                ticker = yf.Ticker(symbol)
                data = await asyncio.to_thread(ticker.history, period=period)

                if data.empty:
                    self.logger.warning("No data returned for %s", symbol)
                    return pd.DataFrame()

                # Cache the data
                self._price_cache[cache_key] = data
                self._save_to_db("price_cache", symbol, data)

                self.logger.info(
                    "Async fetched price data for %s: %d records", symbol, len(data)
                )
                return data

            self.logger.error("Yahoo Finance not available")
            return pd.DataFrame()

        except (requests.RequestException, ConnectionError):
            self.logger.exception("Network error fetching price data for %s", symbol)
            return pd.DataFrame()
        except ValueError:
            self.logger.exception("Invalid data received for %s", symbol)
            return pd.DataFrame()
        except Exception:
            self.logger.exception("Unexpected error fetching price data for %s", symbol)
            return pd.DataFrame()

    def get_google_trends(
        self, keyword: str, timeframe: str = "today 12-m", use_cache: bool = True
    ) -> tuple[pd.DataFrame, float]:
        """
        Get Google Trends data for a keyword with latency tracking.

        Args:
            keyword: Search keyword
            timeframe: Time period for trends
            use_cache: Whether to use cached data

        Returns:
            Tuple of (DataFrame with trends data, latency_hours)
            latency_hours = 0 if fresh data, >0 if cached/stale, -1 if error
        """
        import time
        cache_key = f"{keyword}_{timeframe}"

        # Check cache first (TTLCache handles expiration automatically)
        if use_cache and cache_key in self._trends_cache:
            # Check if we have a timestamp for this data
            timestamp_key = f"{cache_key}_timestamp"
            if timestamp_key in self._trends_timestamps:
                fetch_time = self._trends_timestamps[timestamp_key]
                latency_hours = (time.time() - fetch_time) / 3600  # Convert to hours

                # Log latency for monitoring
                if latency_hours > 1:  # Only log if > 1 hour old
                    self.logger.info(
                        f"Using cached trends data for {keyword}: {latency_hours:.1f}h old"
                    )

                return self._trends_cache[cache_key], latency_hours
            else:
                # No timestamp available, assume fresh
                return self._trends_cache[cache_key], 0.0

        try:
            if self.trends_available:
                fetch_start = time.time()
                self.pytrends.build_payload([keyword], timeframe=timeframe)
                trends_data = self.pytrends.interest_over_time()
                fetch_time = time.time() - fetch_start

                if trends_data.empty:
                    self.logger.warning(f"No trends data for {keyword}")
                    return pd.DataFrame(), -1

                # Clean up the data
                trends_data = trends_data.drop(columns=["isPartial"])
                trends_data.columns = ["value"]

                # Cache the data with timestamp
                self._trends_cache[cache_key] = trends_data
                self._trends_timestamps[f"{cache_key}_timestamp"] = time.time()

                self._save_to_db("trends_cache", keyword, trends_data)

                self.logger.info(
                    f"Fetched trends data for {keyword}: {len(trends_data)} records "
                    f"(fetch time: {fetch_time:.2f}s)"
                )
                return trends_data, 0.0  # Fresh data

            self.logger.error("Google Trends not available")
            return pd.DataFrame(), -1

        except (requests.RequestException, ConnectionError) as e:
            self.logger.error(f"Network error fetching trends data for {keyword}: {e}")
            return pd.DataFrame(), -1
        except ValueError as e:
            self.logger.error(f"Invalid trends data received for {keyword}: {e}")
            return pd.DataFrame(), -1
        except Exception as e:
            self.logger.error(
                f"Unexpected error fetching trends data for {keyword}: {e}"
            )
            return pd.DataFrame(), -1

    async def get_google_trends_async(
        self, keyword: str, timeframe: str = "today 12-m", use_cache: bool = True
    ) -> tuple[pd.DataFrame, float]:
        """
        Async version: Get Google Trends data for a keyword with latency tracking.

        Args:
            keyword: Search keyword
            timeframe: Time period for trends
            use_cache: Whether to use cached data

        Returns:
            Tuple of (DataFrame with trends data, latency_hours)
            latency_hours = 0 if fresh data, >0 if cached/stale, -1 if error
        """
        import time
        cache_key = f"{keyword}_{timeframe}"

        # Check cache first (TTLCache handles expiration automatically)
        if use_cache and cache_key in self._trends_cache:
            # Check if we have a timestamp for this data
            timestamp_key = f"{cache_key}_timestamp"
            if timestamp_key in self._trends_timestamps:
                fetch_time = self._trends_timestamps[timestamp_key]
                latency_hours = (time.time() - fetch_time) / 3600  # Convert to hours

                # Log latency for monitoring
                if latency_hours > 1:  # Only log if > 1 hour old
                    self.logger.info(
                        f"Using cached trends data for {keyword}: {latency_hours:.1f}h old"
                    )

                return self._trends_cache[cache_key], latency_hours
            else:
                # No timestamp available, assume fresh
                return self._trends_cache[cache_key], 0.0

        try:
            if self.trends_available:
                # Use asyncio.to_thread to run the synchronous pytrends calls in a thread
                fetch_start = time.time()
                pytrends_instance = TrendReq(hl="en-US", tz=360)
                await asyncio.to_thread(
                    pytrends_instance.build_payload, [keyword], timeframe=timeframe
                )
                trends_data = await asyncio.to_thread(
                    pytrends_instance.interest_over_time
                )
                fetch_time = time.time() - fetch_start

                if trends_data.empty:
                    self.logger.warning(f"No trends data for {keyword}")
                    return pd.DataFrame(), -1

                # Clean up the data
                trends_data = trends_data.drop(columns=["isPartial"])
                trends_data.columns = ["value"]

                # Cache the data with timestamp
                self._trends_cache[cache_key] = trends_data
                self._trends_timestamps[f"{cache_key}_timestamp"] = time.time()

                self._save_to_db("trends_cache", keyword, trends_data)

                self.logger.info(
                    f"Async fetched trends data for {keyword}: {len(trends_data)} records "
                    f"(fetch time: {fetch_time:.2f}s)"
                )
                return trends_data, 0.0  # Fresh data

            self.logger.error("Google Trends not available")
            return pd.DataFrame(), -1

        except (requests.RequestException, ConnectionError) as e:
            self.logger.error(f"Network error fetching trends data for {keyword}: {e}")
            return pd.DataFrame(), -1
        except ValueError as e:
            self.logger.error(f"Invalid trends data received for {keyword}: {e}")
            return pd.DataFrame(), -1
        except Exception as e:
            self.logger.error(
                f"Unexpected error fetching trends data for {keyword}: {e}"
            )
            return pd.DataFrame(), -1

    def get_macro_indicator(
        self, indicator: str, symbol: str = None, use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Get macro economic indicators.

        Args:
            indicator: Indicator name (VIX, SPY, USO, etc.)
            symbol: Symbol for the indicator (if different from indicator)
            use_cache: Whether to use cached data

        Returns:
            DataFrame with indicator data
        """
        symbol = symbol or indicator
        self._validate_symbol(symbol)

        cache_key = f"{indicator}_{symbol}"

        # Check cache first (TTLCache handles expiration automatically)
        if use_cache and cache_key in self._macro_cache:
            return self._macro_cache[cache_key]

        try:
            # Try Yahoo Finance first (for VIX, SPY, etc.)
            if self.yf_available:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="1y")

                if not data.empty:
                    # Use close price as the indicator value
                    indicator_data = data[["Close"]].copy()
                    indicator_data.columns = ["value"]

                    # Cache the data
                    self._macro_cache[cache_key] = indicator_data
                    self._save_to_db("macro_cache", indicator, indicator_data)

                    self.logger.info(
                        f"Fetched macro data for {indicator}: {len(indicator_data)} records"
                    )
                    return indicator_data

            # Fallback to Alpha Vantage if available
            if self.av_available and indicator in ["VIX"]:
                try:
                    data, _ = self.av_client.get_daily_adjusted(
                        symbol="VIX", outputsize="compact"
                    )
                    if not data.empty:
                        indicator_data = data[["5. adjusted close"]].copy()
                        indicator_data.columns = ["value"]

                        self._macro_cache[cache_key] = indicator_data
                        self._save_to_db("macro_cache", indicator, indicator_data)

                        self.logger.info(
                            f"Fetched macro data for {indicator} via Alpha Vantage"
                        )
                        return indicator_data
                except (requests.RequestException, ConnectionError) as e:
                    self.logger.warning(
                        f"Alpha Vantage network error for {indicator}: {e}"
                    )
                except ValueError as e:
                    self.logger.warning(
                        f"Alpha Vantage data error for {indicator}: {e}"
                    )
                except Exception as e:
                    self.logger.warning(
                        f"Unexpected Alpha Vantage error for {indicator}: {e}"
                    )

            self.logger.warning(f"No macro data available for {indicator}")
            return pd.DataFrame()

        except (requests.RequestException, ConnectionError) as e:
            self.logger.error(f"Network error fetching macro data for {indicator}: {e}")
            return pd.DataFrame()
        except ValueError as e:
            self.logger.error(f"Invalid macro data received for {indicator}: {e}")
            return pd.DataFrame()
        except Exception as e:
            self.logger.error(
                f"Unexpected error fetching macro data for {indicator}: {e}"
            )
            return pd.DataFrame()

    def get_universe_data(
        self, symbols: list[str], period: str = "3mo"
    ) -> dict[str, pd.DataFrame]:
        """
        Get price data for multiple symbols in the universe.

        Args:
            symbols: List of symbols to fetch
            period: Data period

        Returns:
            Dictionary mapping symbols to their price data
        """
        if not isinstance(symbols, list):
            raise ValueError(f"Symbols must be a list, got {type(symbols)}")

        for symbol in symbols:
            self._validate_symbol(symbol)

        universe_data = {}

        for symbol in symbols:
            try:
                data = self.get_price_data(symbol, period)
                if not data.empty:
                    universe_data[symbol] = data
                else:
                    self.logger.warning(f"No data for {symbol}")
            except (requests.RequestException, ConnectionError) as e:
                self.logger.error(f"Network error fetching data for {symbol}: {e}")
            except ValueError as e:
                self.logger.error(f"Invalid data received for {symbol}: {e}")
            except Exception as e:
                self.logger.error(f"Unexpected error fetching data for {symbol}: {e}")

        self.logger.info(
            f"Fetched data for {len(universe_data)}/{len(symbols)} symbols"
        )
        return universe_data

    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for price data.

        Args:
            data: OHLCV price data

        Returns:
            DataFrame with additional technical indicators
        """
        if data.empty:
            return data

        result = data.copy()

        try:
            # ATR (Average True Range)
            high_low = data["High"] - data["Low"]
            high_close = np.abs(data["High"] - data["Close"].shift())
            low_close = np.abs(data["Low"] - data["Close"].shift())
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            result["ATR"] = true_range.rolling(window=14).mean()

            # Volume moving average
            result["Volume_MA"] = data["Volume"].rolling(window=14).mean()

            # Price returns
            result["Returns"] = data["Close"].pct_change()
            result["Returns_5d"] = data["Close"].pct_change(5)

            # Volatility (rolling standard deviation of returns)
            result["Volatility"] = result["Returns"].rolling(window=14).std()

        except (ValueError, TypeError) as e:
            self.logger.error(f"Data error calculating technical indicators: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error calculating technical indicators: {e}")

        return result

    def _save_to_db(self, table: str, symbol: str, data: pd.DataFrame) -> None:
        """Save data to database cache using SQLAlchemy."""
        if not self.db_manager or data.empty:
            return

        session = self.db_manager.get_session()
        try:
            for date, row in data.iterrows():
                if table == "price_cache":
                    # Check if record exists
                    existing = (
                        session.query(PriceCache)
                        .filter_by(symbol=symbol, date=str(date.date()))
                        .first()
                    )

                    if existing:
                        # Update existing record
                        existing.open = row["Open"]
                        existing.high = row["High"]
                        existing.low = row["Low"]
                        existing.close = row["Close"]
                        existing.volume = row["Volume"]
                    else:
                        # Create new record
                        record = PriceCache(
                            symbol=symbol,
                            date=str(date.date()),
                            open=row["Open"],
                            high=row["High"],
                            low=row["Low"],
                            close=row["Close"],
                            volume=row["Volume"],
                        )
                        session.add(record)

                elif table == "trends_cache":
                    existing = (
                        session.query(TrendsCache)
                        .filter_by(keyword=symbol, date=str(date.date()))
                        .first()
                    )

                    if existing:
                        existing.value = row["value"]
                    else:
                        record = TrendsCache(
                            keyword=symbol, date=str(date.date()), value=row["value"]
                        )
                        session.add(record)

                elif table == "macro_cache":
                    existing = (
                        session.query(MacroCache)
                        .filter_by(indicator=symbol, date=str(date.date()))
                        .first()
                    )

                    if existing:
                        existing.value = row["value"]
                    else:
                        record = MacroCache(
                            indicator=symbol, date=str(date.date()), value=row["value"]
                        )
                        session.add(record)

            session.commit()

        except Exception as e:
            session.rollback()
            self.logger.error(f"Error saving data to database: {e}")
        finally:
            session.close()

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._price_cache.clear()
        self._trends_cache.clear()
        self._macro_cache.clear()
        self.logger.info("Cache cleared")

    def close(self) -> None:
        """Close database connection."""
        if self.db_manager:
            self.db_manager.close()
            self.logger.info("Database connection closed")

    def get_cached_posterior(self, features_key: str) -> dict | None:
        """
        Get cached Bayesian posterior probabilities.

        Args:
            features_key: String key representing the feature vector

        Returns:
            Cached posterior data or None if not cached
        """
        return self._posterior_cache.get(features_key)

    def cache_posterior(self, features_key: str, posterior_data: dict) -> None:
        """
        Cache Bayesian posterior probabilities.

        Args:
            features_key: String key representing the feature vector
            posterior_data: Posterior probability data to cache
        """
        self._posterior_cache[features_key] = posterior_data
        self.logger.debug(f"Cached posterior for key: {features_key}")

    def clear_posterior_cache(self) -> None:
        """Clear the posterior probability cache."""
        self._posterior_cache.clear()
        self.logger.info("Posterior cache cleared")


# DataManager is now created via dependency injection in di.py
