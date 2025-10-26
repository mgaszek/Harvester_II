"""
Main trading engine for Harvester II trading system.
Orchestrates the daily trading loop and coordinates all components.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from loguru import logger
import requests
from datetime import datetime, timedelta
import time
import schedule
import sqlite3
import logging
from pathlib import Path

from config import SensitiveDataFilter

# Import Prometheus metrics
try:
    from prometheus_client import Gauge, start_http_server, CollectorRegistry, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    Gauge = None
    start_http_server = None
    CollectorRegistry = None
    generate_latest = None


class TradingEngine:
    """Main trading engine that orchestrates the Harvester II system."""

    def __init__(self, config, data_manager, signal_calculator, risk_manager, portfolio_manager):
        """Initialize trading engine with injected dependencies."""
        self.config = config
        self.data_manager = data_manager
        self.signal_calculator = signal_calculator
        self.risk_manager = risk_manager
        self.portfolio_manager = portfolio_manager
        self.logger = logging.getLogger(__name__)
        
        # Setup logging
        self._setup_logging()

        # Setup Prometheus metrics
        self._setup_metrics()

        # Trading state
        self.is_running = False
        self.tradable_universe: List[str] = []
        self.last_universe_update = None
        
        # Trading schedule
        self.run_time = self.config.get('trading.schedule.run_time', '16:00')
        self.timezone = self.config.get('trading.schedule.timezone', 'US/Eastern')
        
        logger.info("Trading Engine initialized")
    
    def _setup_logging(self) -> None:
        """Setup Loguru structured logging configuration."""
        log_config = self.config.logging

        # Create logs directory
        log_path = Path(log_config.get('file_path', 'logs/harvester_ii.log'))
        log_path.parent.mkdir(exist_ok=True)

        # Configure Loguru with sensitive data filtering
        logger.remove()  # Remove default handler

        # Add file handler with rotation and sensitive data filtering
        logger.add(
            log_path,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}",
            level=log_config.get('level', 'INFO'),
            rotation="10 MB",
            retention="1 week",
            filter=lambda record: self._filter_sensitive_data(record)
        )

        # Add console handler
        logger.add(
            lambda msg: print(msg, end=""),
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
            level=log_config.get('level', 'INFO'),
            filter=lambda record: self._filter_sensitive_data(record)
        )

    def _filter_sensitive_data(self, record) -> bool:
        """Filter sensitive data from log records."""
        message = str(record["message"])
        sensitive_filter = SensitiveDataFilter()

        # Apply sensitive data filtering to the message
        # Loguru doesn't have direct filter support like standard logging,
        # so we'll check for sensitive patterns and filter them
        sensitive_patterns = ['api_key', 'password', 'secret', 'token']
        if any(pattern in message.lower() for pattern in sensitive_patterns):
            return False
        return True

    def _setup_metrics(self) -> None:
        """Setup Prometheus metrics for monitoring."""
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus client not available - metrics disabled")
            self.metrics_enabled = False
            return

        try:
            # Create metrics registry
            self.metrics_registry = CollectorRegistry()

            # Define metrics
            self.equity_gauge = Gauge(
                'harvester_equity_total',
                'Total portfolio equity in USD',
                registry=self.metrics_registry
            )

            self.drawdown_gauge = Gauge(
                'harvester_drawdown_percentage',
                'Current portfolio drawdown as percentage',
                registry=self.metrics_registry
            )

            self.positions_gauge = Gauge(
                'harvester_positions_open',
                'Number of open positions',
                registry=self.metrics_registry
            )

            self.daily_pnl_gauge = Gauge(
                'harvester_daily_pnl',
                'Daily profit and loss in USD',
                registry=self.metrics_registry
            )

            self.g_score_gauge = Gauge(
                'harvester_g_score',
                'Current G-Score for macro risk assessment',
                registry=self.metrics_registry
            )

            self.conviction_gauge = Gauge(
                'harvester_signal_conviction',
                'Current signal conviction level (0.0-1.0)',
                registry=self.metrics_registry
            )

            # Start metrics server
            metrics_port = self.config.get('monitoring.prometheus_port', 8000)
            start_http_server(metrics_port, registry=self.metrics_registry)
            logger.info(f"Prometheus metrics server started on port {metrics_port}")

            self.metrics_enabled = True

        except Exception as e:
            logger.error(f"Failed to setup Prometheus metrics: {e}")
            self.metrics_enabled = False

    def update_metrics(self) -> None:
        """Update Prometheus metrics with current system state."""
        if not self.metrics_enabled:
            return

        try:
            # Get current system status
            status = self.get_system_status()

            # Update metrics
            portfolio = status.get('portfolio', {})
            risk = status.get('risk_management', {})
            macro = status.get('macro_risk', {})

            self.equity_gauge.set(portfolio.get('current_equity', 0))
            self.drawdown_gauge.set(risk.get('current_drawdown', 0))
            self.positions_gauge.set(portfolio.get('open_positions', 0))
            self.daily_pnl_gauge.set(portfolio.get('daily_pnl', 0))
            self.g_score_gauge.set(macro.get('g_score', 0))
            # Update conviction gauge with most recent signal conviction (default to 0.5 if no recent signals)
            self.conviction_gauge.set(getattr(self, '_last_conviction', 0.5))

        except Exception as e:
            logger.error(f"Failed to update metrics: {e}")

    def get_metrics(self) -> str:
        """Get current metrics in Prometheus format."""
        if not self.metrics_enabled:
            return "# Metrics not available - Prometheus client not installed"

        try:
            return generate_latest(self.metrics_registry).decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to generate metrics: {e}")
            return f"# Error generating metrics: {e}"

    def initialize_system(self) -> bool:
        """
        Initialize the trading system.
        
        Returns:
            True if initialization successful
        """
        try:
            self.logger.info("Initializing Harvester II Trading System...")
            
            # Validate configuration
            if not self.config.validate():
                self.logger.error("Configuration validation failed")
                return False
            
            # Update tradable universe
            self.update_tradable_universe()
            
            # Reset daily statistics
            self.risk_manager.reset_daily_stats()
            
            # Check system health
            if not self.check_system_health():
                self.logger.error("System health check failed")
                return False
            
            logger.info("System initialization completed successfully")
            return True
            
        except (ValueError, KeyError) as e:
            self.logger.error(f"Configuration error during initialization: {e}")
            return False
        except (requests.RequestException, ConnectionError) as e:
            self.logger.error(f"Network error during initialization: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error during system initialization: {e}")
            return False
    
    def update_tradable_universe(self) -> None:
        """Update the tradable universe based on CRI filtering."""
        try:
            self.logger.info("Updating tradable universe...")
            
            universe = self.config.get('universe.assets', [])
            if not universe:
                self.logger.error("No assets defined in universe")
                return
            
            # Calculate CRI for all assets and filter
            self.tradable_universe = self.signal_calculator.get_tradable_universe(universe)
            
            self.last_universe_update = datetime.now()
            
            self.logger.info(f"Tradable universe updated: {len(self.tradable_universe)} assets")
            self.logger.info(f"Tradable assets: {', '.join(self.tradable_universe)}")
            
        except (ValueError, KeyError) as e:
            self.logger.error(f"Configuration error updating tradable universe: {e}")
        except (requests.RequestException, ConnectionError) as e:
            self.logger.error(f"Network error updating tradable universe: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error updating tradable universe: {e}")
    
    def check_system_health(self) -> bool:
        """
        Check system health and connectivity.
        
        Returns:
            True if system is healthy
        """
        try:
            self.logger.info("Performing system health check...")
            
            # Check data sources
            test_symbol = "SPY"
            price_data = self.data_manager.get_price_data(test_symbol, period="5d")
            if price_data.empty:
                self.logger.error("Price data source not available")
                return False
            
            # Check Google Trends
            trends_data = self.data_manager.get_google_trends(test_symbol, timeframe="today 5-d")
            if trends_data.empty:
                self.logger.warning("Google Trends not available - continuing without trends data")
            
            # Check macro indicators
            vix_data = self.data_manager.get_macro_indicator("VIX")
            if vix_data.empty:
                self.logger.warning("VIX data not available - G-Score calculation may be limited")
            
            # Check risk manager
            if not self.risk_manager.can_open_new_position():
                self.logger.warning("Risk manager indicates no new positions allowed")
            
            self.logger.info("System health check completed")
            return True
            
        except (requests.RequestException, ConnectionError) as e:
            self.logger.error(f"Network error during health check: {e}")
            return False
        except ValueError as e:
            self.logger.error(f"Data validation error during health check: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error during system health check: {e}")
            return False
    
    def run_daily_trading_loop(self) -> None:
        """Execute the main daily trading loop."""
        try:
            self.logger.info("Starting daily trading loop...")
            
            # Check drawdown limit first
            if self.risk_manager.check_drawdown_limit():
                self.logger.critical("DAILY DRAWDOWN LIMIT EXCEEDED - STOPPING TRADING")
                self.stop_trading()
                return
            
            # Calculate G-Score for macro risk assessment
            g_score = self.signal_calculator.calculate_g_score()
            self.logger.info(f"Current G-Score: {g_score:.1f}")
            
            # Process exit signals first
            closed_positions = self.portfolio_manager.process_exit_signals()
            if closed_positions > 0:
                self.logger.info(f"Closed {closed_positions} positions")
            
            # Check if we can open new positions
            if not self.risk_manager.can_open_new_position():
                self.logger.info("Cannot open new positions - limits reached")
                return
            
            # Update tradable universe if needed (weekly)
            if (self.last_universe_update is None or 
                datetime.now() - self.last_universe_update > timedelta(days=7)):
                self.update_tradable_universe()
            
            # Get entry signals
            entry_signals = self.signal_calculator.get_entry_signals(self.tradable_universe)
            
            if not entry_signals:
                self.logger.info("No entry signals generated")
                return
            
            self.logger.info(f"Generated {len(entry_signals)} entry signals")
            
            # Process entry signals
            executed_orders = 0
            for signal in entry_signals:
                symbol = signal['symbol']

                # Log conviction level for monitoring
                conviction = signal.get('confidence', 0.0)
                conviction_level = signal.get('conviction_level', 'unknown')
                market_state = signal.get('market_state', 'unknown')
                assessment_method = signal.get('assessment_method', 'unknown')

                # Store last conviction for metrics
                self._last_conviction = conviction

                self.logger.info(f"Signal conviction {conviction:.2f} ({conviction_level}) for {symbol} "
                               f"in {market_state} state via {assessment_method}")
                
                # Skip if position already exists
                if symbol in self.portfolio_manager.positions:
                    self.logger.debug(f"Skipping {symbol} - position already exists")
                    continue
                
                # Get current price
                price_data = self.data_manager.get_price_data(symbol, period="5d")
                if price_data.empty:
                    self.logger.warning(f"No price data for {symbol}")
                    continue
                
                current_price = price_data['Close'].iloc[-1]
                
                # Execute entry order
                if self.portfolio_manager.execute_entry_order(signal, current_price):
                    executed_orders += 1
                    self.logger.info(f"Executed entry order for {symbol}")
                else:
                    self.logger.warning(f"Failed to execute entry order for {symbol}")
            
            # Update metrics
            self.update_metrics()

            logger.info(f"Daily trading loop completed - {executed_orders} orders executed")
            
        except (ValueError, KeyError) as e:
            self.logger.error(f"Configuration or data error in trading loop: {e}")
        except (requests.RequestException, ConnectionError) as e:
            self.logger.error(f"Network error in trading loop: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error in daily trading loop: {e}")
    
    def start_trading(self) -> None:
        """Start the trading system."""
        try:
            if not self.initialize_system():
                self.logger.error("Failed to initialize system")
                return
            
            self.is_running = True
            
            # Schedule daily trading
            schedule.every().day.at(self.run_time).do(self.run_daily_trading_loop)
            
            self.logger.info(f"Trading system started - scheduled to run daily at {self.run_time}")
            
            # Run initial trading loop
            self.run_daily_trading_loop()
            
            # Keep running and process scheduled tasks
            while self.is_running:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            self.logger.info("Trading system stopped by user")
            self.stop_trading()
        except (ValueError, KeyError) as e:
            self.logger.error(f"Configuration error in trading system: {e}")
            self.stop_trading()
        except (requests.RequestException, ConnectionError) as e:
            self.logger.error(f"Network error in trading system: {e}")
            self.stop_trading()
        except Exception as e:
            self.logger.error(f"Unexpected error in trading system: {e}")
            self.stop_trading()
    
    def stop_trading(self) -> None:
        """Stop the trading system."""
        try:
            self.is_running = False
            schedule.clear()
            
            # Close all connections
            self.data_manager.close()
            self.portfolio_manager.close()
            
            self.logger.info("Trading system stopped")
            
        except (sqlite3.Error, OSError) as e:
            self.logger.error(f"Database/filesystem error stopping trading system: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error stopping trading system: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status.
        
        Returns:
            Dictionary with system status information
        """
        try:
            # Get portfolio summary
            portfolio_summary = self.portfolio_manager.get_portfolio_summary()
            
            # Get risk summary
            risk_summary = self.risk_manager.get_portfolio_summary()
            
            # Calculate G-Score
            g_score = self.signal_calculator.calculate_g_score()
            
            status = {
                'system_status': 'running' if self.is_running else 'stopped',
                'last_update': datetime.now(),
                'tradable_universe': {
                    'total_assets': len(self.tradable_universe),
                    'assets': self.tradable_universe,
                    'last_update': self.last_universe_update
                },
                'macro_risk': {
                    'g_score': g_score,
                    'risk_level': 'high' if g_score >= self.config.get('macro_risk.g_score_threshold', 2) else 'normal'
                },
                'portfolio': portfolio_summary,
                'risk_management': risk_summary,
                'data_sources': {
                    'yfinance': self.data_manager.yf_available,
                    'google_trends': self.data_manager.trends_available,
                    'alpha_vantage': self.data_manager.av_available
                }
            }
            
            return status
            
        except (ValueError, KeyError) as e:
            self.logger.error(f"Data error getting system status: {e}")
            return {}
        except Exception as e:
            self.logger.error(f"Unexpected error getting system status: {e}")
            return {}
    
    def run_backtest(self, start_date: str = None, end_date: str = None, use_vectorbt: bool = None) -> Dict[str, Any]:
        """
        Run backtest of the trading system using the enhanced backtest engine.

        Args:
            start_date: Start date for backtest (YYYY-MM-DD)
            end_date: End date for backtest (YYYY-MM-DD)
            use_vectorbt: Override config to use vectorbt (None = use config setting)

        Returns:
            Dictionary with backtest results
        """
        try:
            # Import backtest engine
            from backtest import BacktestEngine

            # Get backtest parameters from config
            backtest_config = self.config.backtesting
            start_date = start_date or backtest_config.get('start_date', '2020-01-01')
            end_date = end_date or backtest_config.get('end_date', '2024-01-01')
            initial_capital = backtest_config.get('initial_capital', 100000)

            self.logger.info(f"Starting backtest: {start_date} to {end_date}")

            # Use enhanced backtest engine with injected dependencies
            backtest_engine = BacktestEngine(self.config, self.data_manager, self.signal_calculator, self.risk_manager)
            results = backtest_engine.run_backtest(start_date, end_date, initial_capital, use_vectorbt=use_vectorbt)
            
            if 'error' not in results:
                self.logger.info("Backtest completed successfully")
                self.logger.info(f"Total Return: {results.get('capital', {}).get('total_return', 0):.2%}")
                self.logger.info(f"Max Drawdown: {results.get('capital', {}).get('max_drawdown', 0):.2%}")
                self.logger.info(f"Total Trades: {results.get('trade_statistics', {}).get('total_trades', 0)}")
            
            return results
            
        except (ValueError, TypeError) as e:
            self.logger.error(f"Data/configuration error in backtest: {e}")
            return {'error': str(e)}
        except (ImportError, ModuleNotFoundError) as e:
            self.logger.error(f"Module import error in backtest: {e}")
            return {'error': str(e)}
        except Exception as e:
            self.logger.error(f"Unexpected error in backtest: {e}")
            return {'error': str(e)}

    def run_ab_test(self, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """
        Run A/B test comparing Bayesian State Machine enabled vs disabled.

        Args:
            start_date: Start date for backtest (YYYY-MM-DD)
            end_date: End date for backtest (YYYY-MM-DD)

        Returns:
            Dictionary with A/B test results comparison
        """
        try:
            # Import backtest engine
            from backtest import BacktestEngine

            # Get backtest parameters from config
            backtest_config = self.config.backtesting
            start_date = start_date or backtest_config.get('start_date', '2020-01-01')
            end_date = end_date or backtest_config.get('end_date', '2024-01-01')
            initial_capital = backtest_config.get('initial_capital', 100000)

            self.logger.info(f"Starting A/B test: {start_date} to {end_date}")

            # Use enhanced backtest engine with injected dependencies
            backtest_engine = BacktestEngine(self.config, self.data_manager, self.signal_calculator, self.risk_manager)
            results = backtest_engine.run_ab_test(start_date, end_date, initial_capital)

            if 'error' not in results:
                self.logger.info("A/B test completed successfully")
                comparison = results.get('comparison', {})
                if comparison:
                    self.logger.info(f"Sharpe ratio improvement: {comparison.get('sharpe_ratio_improvement', 0):.3f}")
                    self.logger.info(f"Total return improvement: {comparison.get('total_return_improvement', 0):.3f}")
                    self.logger.info(f"Max drawdown improvement: {comparison.get('max_drawdown_improvement', 0):.3f}")
            else:
                self.logger.error(f"A/B test failed: {results['error']}")

            return results

        except (ValueError, TypeError) as e:
            self.logger.error(f"Data/configuration error in A/B test: {e}")
            return {'error': str(e)}
        except (ImportError, ModuleNotFoundError) as e:
            self.logger.error(f"Module import error in A/B test: {e}")
            return {'error': str(e)}
        except Exception as e:
            self.logger.error(f"Unexpected error in A/B test: {e}")
            return {'error': str(e)}

    def run_walk_forward_validation(self, start_date: str = None, end_date: str = None,
                                   train_window_months: int = 12, test_window_months: int = 3,
                                   step_months: int = 1) -> Dict[str, Any]:
        """
        Run walk-forward validation to detect overfitting and look-ahead bias.

        Args:
            start_date: Start date for validation (YYYY-MM-DD)
            end_date: End date for validation (YYYY-MM-DD)
            train_window_months: Training window in months
            test_window_months: Testing window in months
            step_months: Step size in months

        Returns:
            Dictionary with walk-forward validation results
        """
        try:
            # Import backtest engine
            from backtest import BacktestEngine

            # Get validation parameters from config or use defaults
            backtest_config = self.config.backtesting
            start_date = start_date or backtest_config.get('start_date', '2020-01-01')
            end_date = end_date or backtest_config.get('end_date', '2024-01-01')
            initial_capital = backtest_config.get('initial_capital', 100000)

            self.logger.info(f"Starting walk-forward validation: {start_date} to {end_date}")

            # Use enhanced backtest engine with injected dependencies
            backtest_engine = BacktestEngine(self.config, self.data_manager, self.signal_calculator, self.risk_manager)
            results = backtest_engine.run_walk_forward_validation(
                start_date, end_date, initial_capital,
                train_window_months, test_window_months, step_months
            )

            if 'error' not in results:
                self.logger.info("Walk-forward validation completed successfully")
                summary = results.get('summary', {})
                if summary:
                    overfitting_rate = summary.get('overfitting_rate', 0)
                    recommendation = summary.get('recommendation', '')
                    self.logger.info(f"Overfitting rate: {overfitting_rate:.2f}, Recommendation: {recommendation}")
            else:
                self.logger.error(f"Walk-forward validation failed: {results['error']}")

            return results

        except (ValueError, TypeError) as e:
            self.logger.error(f"Data/configuration error in walk-forward validation: {e}")
            return {'error': str(e)}
        except (ImportError, ModuleNotFoundError) as e:
            self.logger.error(f"Module import error in walk-forward validation: {e}")
            return {'error': str(e)}
        except Exception as e:
            self.logger.error(f"Unexpected error in walk-forward validation: {e}")
            return {'error': str(e)}

    def run_survivor_free_backtest(self, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """
        Run survivor-free backtest to mitigate survivorship bias.

        Args:
            start_date: Start date for backtest (YYYY-MM-DD)
            end_date: End date for backtest (YYYY-MM-DD)

        Returns:
            Dictionary with survivor-free backtest results
        """
        try:
            # Import backtest engine
            from backtest import BacktestEngine

            # Get backtest parameters from config
            backtest_config = self.config.backtesting
            start_date = start_date or backtest_config.get('start_date', '2020-01-01')
            end_date = end_date or backtest_config.get('end_date', '2024-01-01')
            initial_capital = backtest_config.get('initial_capital', 100000)

            self.logger.info(f"Starting survivor-free backtest: {start_date} to {end_date}")

            # Use enhanced backtest engine with injected dependencies
            backtest_engine = BacktestEngine(self.config, self.data_manager, self.signal_calculator, self.risk_manager)
            results = backtest_engine.run_survivor_free_backtest(start_date, end_date, initial_capital)

            if 'error' not in results:
                self.logger.info("Survivor-free backtest completed successfully")
                survivor_analysis = results.get('survivor_analysis', {})
                if survivor_analysis:
                    survival_rate = survivor_analysis.get('survival_rate', 0)
                    self.logger.info(f"Survival rate: {survival_rate:.2f} "
                                   f"({survivor_analysis.get('survivor_universe_size', 0)}/"
                                   f"{survivor_analysis.get('original_universe_size', 0)} assets)")
            else:
                self.logger.error(f"Survivor-free backtest failed: {results['error']}")

            return results

        except (ValueError, TypeError) as e:
            self.logger.error(f"Data/configuration error in survivor-free backtest: {e}")
            return {'error': str(e)}
        except (ImportError, ModuleNotFoundError) as e:
            self.logger.error(f"Module import error in survivor-free backtest: {e}")
            return {'error': str(e)}
        except Exception as e:
            self.logger.error(f"Unexpected error in survivor-free backtest: {e}")
            return {'error': str(e)}

    def detect_backtest_biases(self, backtest_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze backtest results for common biases and issues.

        Args:
            backtest_result: Result from run_backtest (if None, runs a default backtest)

        Returns:
            Dictionary with bias analysis
        """
        try:
            # Import backtest engine
            from backtest import BacktestEngine

            if backtest_result is None:
                self.logger.info("Running default backtest for bias analysis")
                backtest_result = self.run_backtest()

            if 'error' in backtest_result:
                return {'error': f"Cannot analyze biases - backtest failed: {backtest_result['error']}"}

            # Use enhanced backtest engine for bias detection
            backtest_engine = BacktestEngine(self.config, self.data_manager, self.signal_calculator, self.risk_manager)
            bias_analysis = backtest_engine.detect_backtest_biases(backtest_result)

            if 'error' not in bias_analysis:
                self.logger.info("Bias analysis completed")
                recommendations = bias_analysis.get('recommendations', [])
                if recommendations:
                    self.logger.warning(f"Bias analysis recommendations: {recommendations}")
                else:
                    self.logger.info("No significant biases detected")
            else:
                self.logger.error(f"Bias analysis failed: {bias_analysis['error']}")

            return bias_analysis

        except (ValueError, TypeError) as e:
            self.logger.error(f"Data/configuration error in bias detection: {e}")
            return {'error': str(e)}
        except (ImportError, ModuleNotFoundError) as e:
            self.logger.error(f"Module import error in bias detection: {e}")
            return {'error': str(e)}
        except Exception as e:
            self.logger.error(f"Unexpected error in bias detection: {e}")
            return {'error': str(e)}

    def _calculate_max_drawdown(self, equity_series: pd.Series) -> float:
        """Calculate maximum drawdown from equity series."""
        try:
            peak = equity_series.expanding().max()
            drawdown = (equity_series - peak) / peak
            return drawdown.min()
        except (ValueError, TypeError) as e:
            self.logger.error(f"Data error calculating max drawdown: {e}")
            return 0.0
        except Exception as e:
            self.logger.error(f"Unexpected error calculating max drawdown: {e}")
            return 0.0


# TradingEngine is now created via dependency injection in di.py


def main():
    """Main entry point for the trading system."""
    from di import create_trading_engine

    engine = create_trading_engine()

    try:
        engine.start_trading()
    except KeyboardInterrupt:
        print("\\nTrading system stopped by user")
    except (ValueError, KeyError) as e:
        print(f"Configuration error: {e}")
    except (requests.RequestException, ConnectionError) as e:
        print(f"Network error: {e}")
    except Exception as e:
        print(f"Unexpected trading system error: {e}")


if __name__ == "__main__":
    main()