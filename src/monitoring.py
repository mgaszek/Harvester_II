"""
Monitoring and Metrics for Harvester II.
Provides Prometheus gauges for comprehensive system monitoring.
"""

import time
from typing import Any, Optional

try:
    from prometheus_client import (
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
        start_http_server,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    Gauge = None
    start_http_server = None
    CollectorRegistry = None
    generate_latest = None
    Histogram = None
    Counter = None

from logging_config import harvester_logger


class HarvesterMetrics:
    """
    Comprehensive metrics collection for Harvester II trading system.
    Extends Prometheus monitoring with bias analysis and conviction tracking.
    """

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """
        Initialize metrics collection.

        Args:
            registry: Optional custom Prometheus registry
        """
        if not PROMETHEUS_AVAILABLE:
            harvester_logger.warning(
                "Prometheus client not available - metrics disabled"
            )
            self.enabled = False
            return

        self.enabled = True
        self.registry = registry or CollectorRegistry()

        # Core trading metrics
        self.equity_gauge = Gauge(
            "harvester_equity_total",
            "Total portfolio equity in USD",
            registry=self.registry,
        )

        self.drawdown_gauge = Gauge(
            "harvester_drawdown_percentage",
            "Current portfolio drawdown as percentage",
            registry=self.registry,
        )

        self.positions_gauge = Gauge(
            "harvester_positions_open",
            "Number of open positions",
            registry=self.registry,
        )

        self.daily_pnl_gauge = Gauge(
            "harvester_daily_pnl",
            "Daily profit and loss in USD",
            registry=self.registry,
        )

        self.g_score_gauge = Gauge(
            "harvester_g_score",
            "Current G-Score for macro risk assessment",
            registry=self.registry,
        )

        # Signal and conviction metrics
        self.conviction_gauge = Gauge(
            "harvester_signal_conviction",
            "Current signal conviction level (0.0-1.0)",
            registry=self.registry,
        )

        self.last_signal_time = Gauge(
            "harvester_last_signal_timestamp",
            "Timestamp of last trading signal",
            registry=self.registry,
        )

        # Bias analysis metrics
        self.look_ahead_bias_gauge = Gauge(
            "harvester_bias_look_ahead",
            "Look-ahead bias detection (1=detection, 0=no bias)",
            registry=self.registry,
        )

        self.survivorship_bias_gauge = Gauge(
            "harvester_bias_survivorship",
            "Survivorship bias detection (1=detection, 0=no bias)",
            registry=self.registry,
        )

        self.overfitting_gauge = Gauge(
            "harvester_bias_overfitting",
            "Overfitting detection (1=detection, 0=no overfitting)",
            registry=self.registry,
        )

        self.bias_selection_rate = Gauge(
            "harvester_bias_selection_rate",
            "Asset selection rate for bias analysis (0.0-1.0)",
            registry=self.registry,
        )

        # Performance metrics
        self.sharpe_ratio_gauge = Gauge(
            "harvester_performance_sharpe_ratio",
            "Current Sharpe ratio",
            registry=self.registry,
        )

        self.win_rate_gauge = Gauge(
            "harvester_performance_win_rate",
            "Current win rate (0.0-1.0)",
            registry=self.registry,
        )

        self.max_drawdown_gauge = Gauge(
            "harvester_performance_max_drawdown",
            "Maximum drawdown percentage",
            registry=self.registry,
        )

        # System health metrics
        self.system_health_gauge = Gauge(
            "harvester_system_health",
            "System health status (1=healthy, 0=unhealthy)",
            registry=self.registry,
        )

        self.data_sources_available = Gauge(
            "harvester_data_sources_available",
            "Number of available data sources",
            registry=self.registry,
        )

        # Bayesian State Machine metrics
        self.bayesian_states = Gauge(
            "harvester_bayesian_states",
            "Number of Bayesian states detected",
            ["state"],
            registry=self.registry,
        )

        self.bayesian_training_samples = Gauge(
            "harvester_bayesian_training_samples",
            "Number of training samples for Bayesian model",
            registry=self.registry,
        )

        # Counter metrics
        self.signals_generated = Counter(
            "harvester_signals_total",
            "Total number of trading signals generated",
            ["signal_type", "outcome"],
            registry=self.registry,
        )

        self.trades_executed = Counter(
            "harvester_trades_total",
            "Total number of trades executed",
            ["trade_type", "symbol"],
            registry=self.registry,
        )

        # Histogram metrics
        self.signal_processing_time = Histogram(
            "harvester_signal_processing_seconds",
            "Time spent processing trading signals",
            registry=self.registry,
        )

        self.backtest_execution_time = Histogram(
            "harvester_backtest_execution_seconds",
            "Time spent executing backtests",
            registry=self.registry,
        )

        harvester_logger.info("Harvester metrics initialized")

    def update_portfolio_metrics(self, portfolio_summary: dict[str, Any]) -> None:
        """Update portfolio-related metrics."""
        if not self.enabled:
            return

        self.equity_gauge.set(portfolio_summary.get("current_equity", 0))
        self.positions_gauge.set(portfolio_summary.get("open_positions", 0))
        self.daily_pnl_gauge.set(portfolio_summary.get("daily_pnl", 0))

    def update_risk_metrics(self, risk_summary: dict[str, Any]) -> None:
        """Update risk-related metrics."""
        if not self.enabled:
            return

        self.drawdown_gauge.set(risk_summary.get("current_drawdown", 0))

    def update_macro_metrics(self, macro_data: dict[str, Any]) -> None:
        """Update macroeconomic metrics."""
        if not self.enabled:
            return

        self.g_score_gauge.set(macro_data.get("g_score", 0))

    def update_signal_metrics(
        self, conviction: float = 0.5, signal_type: str = "unknown"
    ) -> None:
        """Update signal-related metrics."""
        if not self.enabled:
            return

        self.conviction_gauge.set(conviction)
        self.last_signal_time.set(time.time())

        # Count signal generation
        self.signals_generated.labels(
            signal_type=signal_type, outcome="generated"
        ).inc()

    def update_bias_metrics(self, bias_analysis: dict[str, Any]) -> None:
        """Update bias analysis metrics."""
        if not self.enabled:
            return

        # Update bias detection flags
        look_ahead = bias_analysis.get("look_ahead_bias", {})
        survivorship = bias_analysis.get("survivorship_bias", {})
        overfitting = bias_analysis.get("overfitting", {})

        self.look_ahead_bias_gauge.set(1 if look_ahead.get("detected", False) else 0)
        self.survivorship_bias_gauge.set(
            1 if survivorship.get("detected", False) else 0
        )
        self.overfitting_gauge.set(1 if overfitting.get("detected", False) else 0)

        # Update selection rate
        selection_rate = survivorship.get("selection_rate", 0)
        self.bias_selection_rate.set(selection_rate)

    def update_performance_metrics(self, performance_data: dict[str, Any]) -> None:
        """Update performance metrics."""
        if not self.enabled:
            return

        capital = performance_data.get("capital", {})
        self.sharpe_ratio_gauge.set(capital.get("sharpe_ratio", 0))
        self.win_rate_gauge.set(capital.get("win_rate", 0))
        self.max_drawdown_gauge.set(capital.get("max_drawdown", 0))

    def update_equity_curve_metrics(self, equity_curve: list[dict[str, Any]]) -> None:
        """Update equity curve metrics for monitoring."""
        if not self.enabled or not equity_curve:
            return

        try:
            # Get the latest equity value
            latest_equity = equity_curve[-1].get("equity", 0)
            self.equity_gauge.set(latest_equity)

            # Calculate current drawdown if we have enough data
            if len(equity_curve) > 1:
                peak_equity = max(entry.get("equity", 0) for entry in equity_curve)
                current_drawdown = (peak_equity - latest_equity) / peak_equity * 100
                self.drawdown_gauge.set(current_drawdown)

            # Calculate daily P&L if we have at least 2 data points
            if len(equity_curve) >= 2:
                previous_equity = equity_curve[-2].get("equity", 0)
                daily_pnl = latest_equity - previous_equity
                self.daily_pnl_gauge.set(daily_pnl)

            harvester_logger.debug(f"Updated equity curve metrics: equity={latest_equity}")

        except Exception as e:
            harvester_logger.error(f"Failed to update equity curve metrics: {e}")

    def update_system_health(self, health_status: dict[str, Any]) -> None:
        """Update system health metrics."""
        if not self.enabled:
            return

        # Overall health status
        is_healthy = health_status.get("system_status") == "running"
        self.system_health_gauge.set(1 if is_healthy else 0)

        # Data sources availability
        data_sources = health_status.get("data_sources", {})
        available_count = sum(
            1 for source, available in data_sources.items() if available
        )
        self.data_sources_available.set(available_count)

    def update_bayesian_metrics(self, bsm_data: dict[str, Any]) -> None:
        """Update Bayesian State Machine metrics."""
        if not self.enabled:
            return

        # Update state counts
        states = bsm_data.get("states", {})
        for state_name, count in states.items():
            self.bayesian_states.labels(state=state_name).set(count)

        # Update training samples
        training_samples = bsm_data.get("training_samples", 0)
        self.bayesian_training_samples.set(training_samples)

    def record_trade_execution(self, trade_type: str, symbol: str) -> None:
        """Record trade execution."""
        if not self.enabled:
            return

        self.trades_executed.labels(trade_type=trade_type, symbol=symbol).inc()

    def record_signal_processing_time(self, duration: float) -> None:
        """Record signal processing time."""
        if not self.enabled:
            return

        self.signal_processing_time.observe(duration)

    def record_backtest_execution_time(self, duration: float) -> None:
        """Record backtest execution time."""
        if not self.enabled:
            return

        self.backtest_execution_time.observe(duration)

    def get_metrics_text(self) -> str:
        """Get current metrics in Prometheus format."""
        if not self.enabled or not generate_latest:
            return "# Metrics not available - Prometheus client not installed"

        try:
            return generate_latest(self.registry).decode("utf-8")
        except Exception as e:
            harvester_logger.error(f"Failed to generate metrics: {e}")
            return f"# Error generating metrics: {e}"

    def start_server(self, port: int = 8000) -> None:
        """Start Prometheus metrics server."""
        if not self.enabled or not start_http_server:
            harvester_logger.warning(
                "Cannot start metrics server - Prometheus not available"
            )
            return

        try:
            start_http_server(port, registry=self.registry)
            harvester_logger.info(f"Prometheus metrics server started on port {port}")
        except Exception as e:
            harvester_logger.error(f"Failed to start metrics server: {e}")


# Global metrics instance
_metrics: HarvesterMetrics | None = None


def get_metrics() -> HarvesterMetrics:
    """Get the global metrics instance."""
    global _metrics
    if _metrics is None:
        _metrics = HarvesterMetrics()
    return _metrics


def init_metrics(port: int = 8000) -> HarvesterMetrics:
    """Initialize global metrics and start server."""
    metrics = get_metrics()
    if metrics.enabled:
        metrics.start_server(port)
    return metrics
