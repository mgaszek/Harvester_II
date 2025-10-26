"""
Standardized Logging Configuration for Harvester II.
Uses Loguru for structured logging with JSON toggle capability.
"""

import os
from pathlib import Path
import sys
from typing import Any

try:
    from loguru import logger

    LOGURU_AVAILABLE = True
except ImportError:
    LOGURU_AVAILABLE = False
    import logging

    logger = logging.getLogger(__name__)

# Global configuration
LOG_LEVEL = os.getenv("HARVESTER_LOG_LEVEL", "INFO")
LOG_FORMAT_JSON = os.getenv("HARVESTER_LOG_JSON", "false").lower() == "true"
LOG_FILE_PATH = os.getenv("HARVESTER_LOG_FILE", "logs/harvester_ii.log")
LOG_MAX_SIZE = os.getenv("HARVESTER_LOG_MAX_SIZE", "10 MB")
LOG_RETENTION = os.getenv("HARVESTER_LOG_RETENTION", "1 week")


def setup_logging(
    level: str = LOG_LEVEL,
    json_format: bool = LOG_FORMAT_JSON,
    log_file: str = LOG_FILE_PATH,
    max_size: str = LOG_MAX_SIZE,
    retention: str = LOG_RETENTION,
    enable_console: bool = True,
    enable_file: bool = True,
) -> None:
    """
    Setup standardized logging configuration for Harvester II.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_format: Whether to use JSON format for logs
        log_file: Path to log file
        max_size: Maximum log file size before rotation
        retention: How long to keep log files
        enable_console: Whether to enable console logging
        enable_file: Whether to enable file logging
    """
    if not LOGURU_AVAILABLE:
        # Fallback to standard logging if loguru not available
        import logging

        logging.basicConfig(
            level=getattr(logging, level.upper(), logging.INFO),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        print("Warning: Loguru not available, using standard logging")
        return

    # Remove default logger
    logger.remove()

    # Create logs directory
    if enable_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(exist_ok=True)

    # Configure console handler
    if enable_console:
        if json_format:
            # JSON format for structured logging
            console_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>"
            serialize = False  # Set to True for pure JSON
        else:
            # Human-readable format
            console_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
            serialize = False

        logger.add(
            sys.stdout,
            format=console_format,
            level=level.upper(),
            serialize=serialize,
            colorize=not json_format,  # Disable colors in JSON mode
        )

    # Configure file handler
    if enable_file:
        if json_format:
            file_format = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
            serialize = False
        else:
            file_format = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}"
            serialize = False

        logger.add(
            log_file,
            format=file_format,
            level=level.upper(),
            rotation=max_size,
            retention=retention,
            serialize=serialize,
            encoding="utf-8",
        )

    # Log the configuration
    logger.info(
        "Logging system initialized",
        level=level,
        json_format=json_format,
        log_file=log_file,
        max_size=max_size,
        retention=retention,
    )


def get_logger(name: str = None) -> Any:
    """
    Get a standardized logger instance.

    Args:
        name: Logger name (optional, will use calling module name)

    Returns:
        Logger instance
    """
    if LOGURU_AVAILABLE:
        return logger
    import logging

    return logging.getLogger(name or __name__)


# Global logger instance
harvester_logger = get_logger("harvester_ii")


def log_system_status(status: dict[str, Any], level: str = "info") -> None:
    """
    Log comprehensive system status information.

    Args:
        status: System status dictionary
        level: Log level (debug, info, warning, error, critical)
    """
    log_func = getattr(harvester_logger, level.lower(), harvester_logger.info)

    # Extract key metrics for structured logging
    portfolio = status.get("portfolio", {})
    risk = status.get("risk_management", {})
    macro = status.get("macro_risk", {})

    log_func(
        "System status update",
        equity=round(portfolio.get("current_equity", 0), 2),
        drawdown=round(risk.get("current_drawdown", 0), 4),
        positions=portfolio.get("open_positions", 0),
        g_score=round(macro.get("g_score", 0), 1),
        conviction=getattr(status, "_last_conviction", 0.5),
    )


def log_trading_signal(
    symbol: str,
    signal_type: str,
    conviction: float,
    market_state: str = None,
    assessment_method: str = None,
    level: str = "info",
) -> None:
    """
    Log trading signal information.

    Args:
        symbol: Asset symbol
        signal_type: Type of signal (entry/exit)
        conviction: Signal conviction level (0.0-1.0)
        market_state: Current market state (optional)
        assessment_method: Assessment method used (optional)
        level: Log level
    """
    log_func = getattr(harvester_logger, level.lower(), harvester_logger.info)

    log_func(
        "Trading signal generated",
        symbol=symbol,
        signal_type=signal_type,
        conviction=round(conviction, 3),
        market_state=market_state,
        assessment_method=assessment_method,
    )


def log_performance_metrics(
    metrics: dict[str, Any], period: str = "backtest", level: str = "info"
) -> None:
    """
    Log performance metrics in structured format.

    Args:
        metrics: Performance metrics dictionary
        period: Period type (backtest, live, etc.)
        level: Log level
    """
    log_func = getattr(harvester_logger, level.lower(), harvester_logger.info)

    capital = metrics.get("capital", {})
    trade_stats = metrics.get("trade_statistics", {})

    log_func(
        "Performance metrics",
        period=period,
        total_return=round(capital.get("total_return", 0), 4),
        sharpe_ratio=round(capital.get("sharpe_ratio", 0), 3),
        max_drawdown=round(capital.get("max_drawdown", 0), 4),
        win_rate=round(capital.get("win_rate", 0), 3),
        total_trades=trade_stats.get("total_trades", 0),
    )


def log_bias_analysis(results: dict[str, Any], level: str = "warning") -> None:
    """
    Log bias analysis results.

    Args:
        results: Bias analysis results
        level: Log level
    """
    log_func = getattr(harvester_logger, level.lower(), harvester_logger.warning)

    recommendations = results.get("recommendations", [])
    if recommendations:
        for rec in recommendations:
            log_func("Bias analysis recommendation", recommendation=rec)

    # Log specific bias metrics
    look_ahead = results.get("look_ahead_bias", {})
    survivorship = results.get("survivorship_bias", {})
    overfitting = results.get("overfitting", {})

    log_func(
        "Bias analysis summary",
        look_ahead_detected=look_ahead.get("detected", False),
        survivorship_detected=survivorship.get("detected", False),
        overfitting_detected=overfitting.get("detected", False),
    )


# Initialize logging on import
if LOGURU_AVAILABLE:
    setup_logging()
