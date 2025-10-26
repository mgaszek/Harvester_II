#!/usr/bin/env python3
"""
Harvester II Trading System - Main Entry Point
Volatility and attention-driven trading system
"""

import sys
import argparse
from pathlib import Path
from loguru import logger

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.di import create_trading_engine


def setup_logging():
    """Setup Loguru structured logging configuration."""
    # Remove default logger
    logger.remove()

    # Add console handler with JSON format for structured logging
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
        serialize=False  # Set to True for JSON output
    )

    # Add file handler
    log_path = Path("logs/harvester_ii.log")
    log_path.parent.mkdir(exist_ok=True)
    logger.add(
        log_path,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="10 MB",
        retention="1 week",
        serialize=False
    )


def main():
    """Main entry point for Harvester II trading system."""
    parser = argparse.ArgumentParser(description='Harvester II Trading System')
    parser.add_argument('--mode', choices=['live', 'backtest', 'status', 'metrics'],
                       default='live', help='Trading mode')
    parser.add_argument('--start-date', help='Backtest start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='Backtest end date (YYYY-MM-DD)')
    parser.add_argument('--config', default='config.json',
                       help='Configuration file path')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()

    try:
        logger.info("Starting Harvester II Trading System...", mode=args.mode, config=args.config)

        # Get trading engine with dependency injection
        engine = create_trading_engine(args.config)

        if args.mode == 'live':
            logger.info("Starting live trading mode")
            engine.start_trading()

        elif args.mode == 'backtest':
            logger.info("Starting backtest mode", start_date=args.start_date, end_date=args.end_date)
            results = engine.run_backtest(args.start_date, args.end_date)

            if 'error' in results:
                logger.error("Backtest failed", error=results['error'])
            else:
                logger.info("Backtest completed successfully",
                          total_return=results.get('total_return', 0),
                          max_drawdown=results.get('max_drawdown', 0),
                          total_trades=results.get('total_trades', 0))

        elif args.mode == 'status':
            logger.info("Getting system status")
            status = engine.get_system_status()

            print("\\n=== Harvester II System Status ===")
            print(f"Status: {status.get('system_status', 'unknown')}")
            print(f"Last Update: {status.get('last_update', 'unknown')}")

            # Portfolio info
            portfolio = status.get('portfolio', {})
            print(f"\\nPortfolio:")
            print(f"  Current Equity: ${portfolio.get('current_equity', 0):,.2f}")
            print(f"  Open Positions: {portfolio.get('open_positions', 0)}")
            print(f"  Daily P&L: ${portfolio.get('daily_pnl', 0):,.2f}")

            # Risk info
            risk = status.get('risk_management', {})
            print(f"\\nRisk Management:")
            print(f"  Current Drawdown: {risk.get('current_drawdown', 0):.2%}")
            print(f"  Max Drawdown: {risk.get('max_drawdown', 0):.2%}")
            print(f"  Can Open New: {risk.get('can_open_new', False)}")

            # Macro risk
            macro = status.get('macro_risk', {})
            print(f"\\nMacro Risk:")
            print(f"  G-Score: {macro.get('g_score', 0):.1f}")
            print(f"  Risk Level: {macro.get('risk_level', 'unknown')}")

            # Tradable universe
            universe = status.get('tradable_universe', {})
            print(f"\\nTradable Universe:")
            print(f"  Total Assets: {universe.get('total_assets', 0)}")
            print(f"  Assets: {', '.join(universe.get('assets', []))}")

        elif args.mode == 'metrics':
            logger.info("Getting Prometheus metrics")
            metrics = engine.get_metrics()
            print(metrics)

    except KeyboardInterrupt:
        logger.info("System stopped by user")
    except Exception as e:
        logger.error("System error", error=str(e), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
