#!/usr/bin/env python3
"""
Harvester II Trading System - Main Entry Point
Volatility and attention-driven trading system
"""

import argparse
from pathlib import Path
import sys

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from di import create_trading_engine
from logging_config import harvester_logger as logger
from logging_config import setup_logging


def main():
    """Main entry point for Harvester II trading system."""
    parser = argparse.ArgumentParser(description="Harvester II Trading System")
    parser.add_argument(
        "--mode",
        choices=[
            "live",
            "backtest",
            "ab-test",
            "walk-forward",
            "survivor-free",
            "bias-check",
            "status",
            "metrics",
            "optimize",
        ],
        default="live",
        help="Trading mode",
    )
    parser.add_argument("--start-date", help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="Backtest end date (YYYY-MM-DD)")
    parser.add_argument(
        "--config", default="config.json", help="Configuration file path"
    )
    parser.add_argument(
        "--vectorbt",
        action="store_true",
        help="Use Vectorbt backtesting engine (when available)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging()

    try:
        logger.info(
            "Starting Harvester II Trading System...",
            mode=args.mode,
            config=args.config,
        )

        # Get trading engine with dependency injection
        engine = create_trading_engine(args.config)

        if args.mode == "live":
            logger.info("Starting live trading mode")
            engine.start_trading()

        elif args.mode == "backtest":
            logger.info(
                "Starting backtest mode",
                start_date=args.start_date,
                end_date=args.end_date,
                vectorbt=args.vectorbt,
            )
            results = engine.run_backtest(
                args.start_date, args.end_date, use_vectorbt=args.vectorbt
            )

            if "error" in results:
                logger.error("Backtest failed", error=results["error"])
            else:
                # Extract metrics from custom backtest results
                capital = results.get("capital", {})
                total_return = capital.get("total_return", 0)
                max_drawdown = capital.get("max_drawdown", 0)
                total_trades = results.get("trade_statistics", {}).get(
                    "total_trades", 0
                )

                logger.info(
                    "Backtest completed successfully",
                    total_return=total_return,
                    max_drawdown=max_drawdown,
                    total_trades=total_trades,
                )

                # Print summary for CLI users
                print("\n=== Backtest Results ===")
                print(f"Total Return: {total_return:.2%}")
                print(f"Max Drawdown: {max_drawdown:.2%}")
                print(f"Total Trades: {total_trades}")
                return  # Exit successfully after backtest

        elif args.mode == "ab-test":
            logger.info(
                "Starting A/B test mode",
                start_date=args.start_date,
                end_date=args.end_date,
            )
            results = engine.run_ab_test(args.start_date, args.end_date)

            if "error" in results:
                logger.error("A/B test failed", error=results["error"])
            else:
                comparison = results.get("comparison", {})
                if comparison:
                    print("\\n=== A/B Test Results ===")
                    print(f"Test Period: {results.get('test_period', 'Unknown')}")
                    print(f"Initial Capital: ${results.get('initial_capital', 0):,.2f}")

                    print("\\n--- Bayesian State Machine ENABLED ---")
                    enabled_capital = results["bayesian_enabled"].get("capital", {})
                    print(
                        f"  Total Return: {enabled_capital.get('total_return', 0):.2%}"
                    )
                    print(
                        f"  Sharpe Ratio: {enabled_capital.get('sharpe_ratio', 0):.3f}"
                    )
                    print(
                        f"  Max Drawdown: {enabled_capital.get('max_drawdown', 0):.2%}"
                    )
                    print(f"  Win Rate: {enabled_capital.get('win_rate', 0):.2%}")

                    print("\\n--- Bayesian State Machine DISABLED ---")
                    disabled_capital = results["bayesian_disabled"].get("capital", {})
                    print(
                        f"  Total Return: {disabled_capital.get('total_return', 0):.2%}"
                    )
                    print(
                        f"  Sharpe Ratio: {disabled_capital.get('sharpe_ratio', 0):.3f}"
                    )
                    print(
                        f"  Max Drawdown: {disabled_capital.get('max_drawdown', 0):.2%}"
                    )
                    print(f"  Win Rate: {disabled_capital.get('win_rate', 0):.2%}")

                    print("\\n--- IMPROVEMENT (Enabled - Disabled) ---")
                    print(
                        f"  Sharpe Ratio: {comparison.get('sharpe_ratio_improvement', 0):.3f}"
                    )
                    print(
                        f"  Total Return: {comparison.get('total_return_improvement', 0):.2%}"
                    )
                    print(
                        f"  Max Drawdown: {comparison.get('max_drawdown_improvement', 0):.2%}"
                    )
                    print(
                        f"  Win Rate: {comparison.get('win_rate_improvement', 0):.2%}"
                    )
                    print(
                        f"  Conviction Correlation: {comparison.get('conviction_correlation', 0):.3f}"
                    )

                    logger.info(
                        "A/B test completed successfully",
                        sharpe_improvement=comparison.get(
                            "sharpe_ratio_improvement", 0
                        ),
                        return_improvement=comparison.get(
                            "total_return_improvement", 0
                        ),
                    )
                else:
                    logger.warning(
                        "A/B test completed but no comparison data available"
                    )

        elif args.mode == "walk-forward":
            logger.info(
                "Starting walk-forward validation mode",
                start_date=args.start_date,
                end_date=args.end_date,
            )
            results = engine.run_walk_forward_validation(args.start_date, args.end_date)

            if "error" in results:
                logger.error("Walk-forward validation failed", error=results["error"])
            else:
                print("\\n=== Walk-Forward Validation Results ===")
                print(f"Period: {results.get('overall_period', 'Unknown')}")
                print(
                    f"Training Window: {results.get('train_window_months', 0)} months"
                )
                print(f"Testing Window: {results.get('test_window_months', 0)} months")

                summary = results.get("summary", {})
                if summary:
                    print("\\n--- Summary Statistics ---")
                    print(f"Total Folds: {summary.get('total_folds', 0)}")
                    print(f"Overfitting Rate: {summary.get('overfitting_rate', 0):.2%}")
                    print(f"Avg Sharpe Gap: {summary.get('avg_sharpe_gap', 0):.3f}")
                    print(f"Recommendation: {summary.get('recommendation', 'Unknown')}")

                folds = results.get("folds", [])
                if folds:
                    print("\\n--- Fold Details ---")
                    for fold in folds[:5]:  # Show first 5 folds
                        perf_gap = fold.get("performance_gap", {})
                        print(
                            f"Fold {fold.get('fold', 0)}: Sharpe Gap {perf_gap.get('sharpe_gap', 0):.3f}, "
                            f"Overfitting: {perf_gap.get('overfitting_detected', False)}"
                        )

                logger.info(
                    "Walk-forward validation completed",
                    total_folds=summary.get("total_folds", 0),
                    overfitting_rate=summary.get("overfitting_rate", 0),
                )

        elif args.mode == "survivor-free":
            logger.info(
                "Starting survivor-free backtest mode",
                start_date=args.start_date,
                end_date=args.end_date,
            )
            results = engine.run_survivor_free_backtest(args.start_date, args.end_date)

            if "error" in results:
                logger.error("Survivor-free backtest failed", error=results["error"])
            else:
                survivor_analysis = results.get("survivor_analysis", {})
                capital = results.get("capital", {})

                print("\\n=== Survivor-Free Backtest Results ===")
                print(
                    f"Period: {args.start_date or 'Default'} to {args.end_date or 'Default'}"
                )

                if survivor_analysis:
                    print("\\n--- Survivor Analysis ---")
                    print(
                        f"Original Universe: {survivor_analysis.get('original_universe_size', 0)} assets"
                    )
                    print(
                        f"Survivor Universe: {survivor_analysis.get('survivor_universe_size', 0)} assets"
                    )
                    print(
                        f"Survival Rate: {survivor_analysis.get('survival_rate', 0):.2%}"
                    )
                    excluded = survivor_analysis.get("excluded_assets", [])
                    if excluded:
                        print(
                            f"Excluded Assets: {', '.join(excluded[:5])}{'...' if len(excluded) > 5 else ''}"
                        )

                print("\\n--- Performance ---")
                print(f"Total Return: {capital.get('total_return', 0):.2%}")
                print(f"Sharpe Ratio: {capital.get('sharpe_ratio', 0):.3f}")
                print(f"Max Drawdown: {capital.get('max_drawdown', 0):.2%}")

                logger.info(
                    "Survivor-free backtest completed",
                    survival_rate=survivor_analysis.get("survival_rate", 0),
                    total_return=capital.get("total_return", 0),
                )

        elif args.mode == "bias-check":
            logger.info("Starting bias detection mode")
            results = engine.detect_backtest_biases()

            if "error" in results:
                logger.error("Bias detection failed", error=results["error"])
            else:
                print("\\n=== Backtest Bias Analysis ===")

                # Look-ahead bias
                look_ahead = results.get("look_ahead_bias", {})
                if look_ahead.get("micro_trading_detected"):
                    print(
                        "WARNING: LOOK-AHEAD BIAS: Micro-trading detected (very short trade durations)"
                    )
                else:
                    print("OK: LOOK-AHEAD BIAS: No micro-trading detected")

                # Survivorship bias
                survivorship = results.get("survivorship_bias", {})
                if survivorship.get("potential_bias"):
                    selection_rate = survivorship.get("selection_rate", 0)
                    print(
                        f"WARNING: SURVIVORSHIP BIAS: Low selection rate ({selection_rate:.1%})"
                    )
                else:
                    print("OK: SURVIVORSHIP BIAS: Acceptable selection rate")

                # Overfitting indicators
                overfitting = results.get("overfitting_indicators", {})
                if overfitting.get("unrealistic_sharpe"):
                    sharpe = overfitting.get("sharpe_ratio", 0)
                    print(
                        f"WARNING: OVERFITTING: Unrealistically high Sharpe ratio ({sharpe:.2f})"
                    )
                else:
                    print("OK: OVERFITTING: Sharpe ratio within reasonable bounds")

                # Recommendations
                recommendations = results.get("recommendations", [])
                if recommendations:
                    print("\\n--- Recommendations ---")
                    for rec in recommendations:
                        print(f"• {rec}")
                else:
                    print("\\nOK: No significant biases detected")

        elif args.mode == "optimize":
            logger.info(
                "Starting hyperparameter optimization mode",
                start_date=args.start_date,
                end_date=args.end_date,
            )
            results = engine.run_hyperparameter_optimization(
                args.start_date, args.end_date
            )

            if "error" in results:
                logger.error("Optimization failed", error=results["error"])
            else:
                print("\\n=== Hyperparameter Optimization Results ===")
                print(f"Best Sharpe Ratio: {results.get('best_sharpe_ratio', 0):.3f}")
                print(f"Optimization Trials: {results.get('optimization_trials', 0)}")

                best_params = results.get("best_parameters", {})
                if best_params:
                    print("\\n--- Optimized Parameters ---")
                    for param, value in best_params.items():
                        print(f"  {param}: {value}")

                final_results = results.get("final_backtest_results", {})
                if "capital" in final_results:
                    capital = final_results["capital"]
                    print("\\n--- Final Backtest Results ---")
                    print(f"  Total Return: {capital.get('total_return', 0):.2%}")
                    print(f"  Sharpe Ratio: {capital.get('sharpe_ratio', 0):.3f}")
                    print(f"  Max Drawdown: {capital.get('max_drawdown', 0):.2%}")

                logger.info(
                    "Optimization completed",
                    best_sharpe=results.get("best_sharpe_ratio", 0),
                )

        elif args.mode == "status":
            logger.info("Getting system status")
            status = engine.get_system_status()

            print("\\n=== Harvester II System Status ===")
            print(f"Status: {status.get('system_status', 'unknown')}")
            print(f"Last Update: {status.get('last_update', 'unknown')}")

            # Portfolio info
            portfolio = status.get("portfolio", {})
            print("\\nPortfolio:")
            print(f"  Current Equity: ${portfolio.get('current_equity', 0):,.2f}")
            print(f"  Open Positions: {portfolio.get('open_positions', 0)}")
            print(f"  Daily P&L: ${portfolio.get('daily_pnl', 0):,.2f}")

            # Risk info
            risk = status.get("risk_management", {})
            print("\\nRisk Management:")
            print(f"  Current Drawdown: {risk.get('current_drawdown', 0):.2%}")
            print(f"  Max Drawdown: {risk.get('max_drawdown', 0):.2%}")
            print(f"  Can Open New: {risk.get('can_open_new', False)}")

            # Macro risk
            macro = status.get("macro_risk", {})
            print("\\nMacro Risk:")
            print(f"  G-Score: {macro.get('g_score', 0):.1f}")
            print(f"  Risk Level: {macro.get('risk_level', 'unknown')}")

            # Tradable universe
            universe = status.get("tradable_universe", {})
            print("\\nTradable Universe:")
            print(f"  Total Assets: {universe.get('total_assets', 0)}")
            print(f"  Assets: {', '.join(universe.get('assets', []))}")

        elif args.mode == "metrics":
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
