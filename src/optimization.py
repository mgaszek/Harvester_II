"""
Hyperparameter optimization module for Harvester II.
Uses Optuna to optimize trading system parameters for maximum Sharpe ratio.
"""

import logging
from typing import Any

import optuna

logger = logging.getLogger(__name__)


class TradingSystemOptimizer:
    """
    Optuna-based hyperparameter optimization for Harvester II trading system.

    Optimizes key parameters to maximize Sharpe ratio while maintaining
    reasonable drawdown constraints.
    """

    def __init__(
        self, config, data_manager, signal_calculator, risk_manager, backtest_engine
    ):
        """
        Initialize optimizer with system components.

        Args:
            config: Configuration object
            data_manager: DataManager instance
            signal_calculator: SignalCalculator instance
            risk_manager: RiskManager instance
            backtest_engine: BacktestEngine instance
        """
        self.config = config
        self.data_manager = data_manager
        self.signal_calculator = signal_calculator
        self.risk_manager = risk_manager
        self.backtest_engine = backtest_engine

        # Optimization settings
        self.n_trials = config.get("optimization.n_trials", 100)
        self.timeout = config.get("optimization.timeout_seconds", 3600)  # 1 hour
        self.study_name = config.get(
            "optimization.study_name", "harvester_ii_optimization"
        )

        # Parameter bounds
        self.param_bounds = {
            "panic_threshold": (1.0, 5.0),
            "cri_threshold": (0.1, 0.8),
            "max_position_size": (0.01, 0.10),  # 1% to 10% of capital
            "stop_loss_pct": (0.05, 0.25),  # 5% to 25%
            "take_profit_pct": (0.05, 0.50),  # 5% to 50%
            "min_holding_period": (1, 30),  # 1 to 30 days
            "max_open_positions": (1, 10),
            "g_score_threshold": (1.0, 3.0),
        }

        logger.info("Trading system optimizer initialized")

    def optimize_parameters(
        self, start_date: str, end_date: str, initial_capital: float = 100000
    ) -> dict[str, Any]:
        """
        Run hyperparameter optimization using Optuna.

        Args:
            start_date: Optimization start date
            end_date: Optimization end date
            initial_capital: Starting capital

        Returns:
            Dictionary with optimization results
        """
        try:
            logger.info(
                f"Starting hyperparameter optimization: {start_date} to {end_date}"
            )

            # Create Optuna study
            study = optuna.create_study(
                study_name=self.study_name,
                direction="maximize",
                sampler=optuna.samplers.TPESampler(),
                pruner=optuna.pruners.MedianPruner(),
            )

            # Run optimization
            study.optimize(
                lambda trial: self._objective_function(
                    trial, start_date, end_date, initial_capital
                ),
                n_trials=self.n_trials,
                timeout=self.timeout,
            )

            # Get best parameters and results
            best_params = study.best_params
            best_value = study.best_value

            # Run final backtest with best parameters
            final_results = self._evaluate_parameters(
                best_params, start_date, end_date, initial_capital
            )

            results = {
                "best_parameters": best_params,
                "best_sharpe_ratio": best_value,
                "optimization_trials": len(study.trials),
                "final_backtest_results": final_results,
                "study_statistics": {
                    "n_completed": len(
                        [
                            t
                            for t in study.trials
                            if t.state == optuna.TrialState.COMPLETE
                        ]
                    ),
                    "n_pruned": len(
                        [t for t in study.trials if t.state == optuna.TrialState.PRUNED]
                    ),
                    "best_trial_number": study.best_trial.number,
                },
            }

            logger.info(f"Optimization completed. Best Sharpe: {best_value:.3f}")
            logger.info(f"Best parameters: {best_params}")

            return results

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return {"error": str(e)}

    def _objective_function(
        self,
        trial: optuna.Trial,
        start_date: str,
        end_date: str,
        initial_capital: float,
    ) -> float:
        """
        Objective function for Optuna optimization.

        Args:
            trial: Optuna trial object
            start_date: Backtest start date
            end_date: Backtest end date
            initial_capital: Starting capital

        Returns:
            Sharpe ratio (to maximize)
        """
        try:
            # Sample parameters
            params = {
                "panic_threshold": trial.suggest_float(
                    "panic_threshold",
                    self.param_bounds["panic_threshold"][0],
                    self.param_bounds["panic_threshold"][1],
                ),
                "cri_threshold": trial.suggest_float(
                    "cri_threshold",
                    self.param_bounds["cri_threshold"][0],
                    self.param_bounds["cri_threshold"][1],
                ),
                "max_position_size": trial.suggest_float(
                    "max_position_size",
                    self.param_bounds["max_position_size"][0],
                    self.param_bounds["max_position_size"][1],
                ),
                "stop_loss_pct": trial.suggest_float(
                    "stop_loss_pct",
                    self.param_bounds["stop_loss_pct"][0],
                    self.param_bounds["stop_loss_pct"][1],
                ),
                "take_profit_pct": trial.suggest_float(
                    "take_profit_pct",
                    self.param_bounds["take_profit_pct"][0],
                    self.param_bounds["take_profit_pct"][1],
                ),
                "min_holding_period": trial.suggest_int(
                    "min_holding_period",
                    self.param_bounds["min_holding_period"][0],
                    self.param_bounds["min_holding_period"][1],
                ),
                "max_open_positions": trial.suggest_int(
                    "max_open_positions",
                    self.param_bounds["max_open_positions"][0],
                    self.param_bounds["max_open_positions"][1],
                ),
                "g_score_threshold": trial.suggest_float(
                    "g_score_threshold",
                    self.param_bounds["g_score_threshold"][0],
                    self.param_bounds["g_score_threshold"][1],
                ),
            }

            # Evaluate parameters
            results = self._evaluate_parameters(
                params, start_date, end_date, initial_capital
            )

            if "error" in results:
                # Return very poor score for failed backtests
                return -10.0

            # Extract Sharpe ratio
            capital_metrics = results.get("capital", {})
            sharpe_ratio = capital_metrics.get("sharpe_ratio", -10.0)

            # Penalize excessive drawdown
            max_drawdown = capital_metrics.get("max_drawdown", 0)
            if max_drawdown > 0.30:  # More than 30% drawdown
                sharpe_ratio -= 2.0

            # Penalize too few trades (overfitting concern)
            trade_stats = results.get("trade_statistics", {})
            total_trades = trade_stats.get("total_trades", 0)
            if total_trades < 5:
                sharpe_ratio -= 1.0

            return sharpe_ratio

        except Exception as e:
            logger.debug(f"Trial failed: {e}")
            return -10.0

    def _evaluate_parameters(
        self,
        params: dict[str, Any],
        start_date: str,
        end_date: str,
        initial_capital: float,
    ) -> dict[str, Any]:
        """
        Evaluate a set of parameters by running a backtest.

        Args:
            params: Parameter dictionary
            start_date: Backtest start date
            end_date: Backtest end date
            initial_capital: Starting capital

        Returns:
            Backtest results dictionary
        """
        try:
            # Create modified config with trial parameters
            trial_config = self._create_trial_config(params)

            # Create trial components with modified config
            from di import (
                create_data_manager,
                create_risk_manager,
                create_signal_calculator,
            )

            trial_data_manager = create_data_manager(trial_config)
            trial_signal_calc = create_signal_calculator(
                trial_config, trial_data_manager
            )
            trial_risk_manager = create_risk_manager(trial_config)

            # Create backtest engine with trial components
            from backtest import BacktestEngine

            trial_backtest_engine = BacktestEngine(
                trial_config, trial_data_manager, trial_signal_calc, trial_risk_manager
            )

            # Run backtest
            results = trial_backtest_engine.run_backtest(
                start_date, end_date, initial_capital
            )

            return results

        except Exception as e:
            logger.debug(f"Parameter evaluation failed: {e}")
            return {"error": str(e)}

    def _create_trial_config(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        Create a trial configuration with optimized parameters.

        Args:
            params: Parameter dictionary from Optuna

        Returns:
            Modified configuration dictionary
        """
        # Start with base config
        trial_config = dict(self.config._config_data)  # Copy base config

        # Override with trial parameters
        overrides = {
            "signals.panic_threshold": params["panic_threshold"],
            "universe.cri_threshold": params["cri_threshold"],
            "risk_management.position_sizing.max_position_size": params[
                "max_position_size"
            ],
            "risk_management.stop_loss_pct": params["stop_loss_pct"],
            "risk_management.take_profit_pct": params["take_profit_pct"],
            "risk_management.min_holding_period": params["min_holding_period"],
            "risk_management.max_open_positions": params["max_open_positions"],
            "macro_risk.g_score_threshold": params["g_score_threshold"],
        }

        # Apply overrides
        for key, value in overrides.items():
            self._set_nested_config(trial_config, key, value)

        return trial_config

    def _set_nested_config(self, config: dict[str, Any], key: str, value: Any) -> None:
        """
        Set a nested configuration value using dot notation.

        Args:
            config: Configuration dictionary
            key: Dot-separated key (e.g., 'signals.panic_threshold')
            value: Value to set
        """
        keys = key.split(".")
        current = config

        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        # Set the value
        current[keys[-1]] = value

    def get_parameter_importance(self, study: optuna.Study) -> dict[str, float]:
        """
        Calculate parameter importance using Optuna's built-in methods.

        Args:
            study: Completed Optuna study

        Returns:
            Dictionary mapping parameter names to importance scores
        """
        try:
            importance = optuna.importance.get_param_importances(study)
            return dict(importance)
        except Exception as e:
            logger.warning(f"Could not calculate parameter importance: {e}")
            return {}

    def save_optimization_results(
        self, results: dict[str, Any], output_path: str = None
    ) -> None:
        """
        Save optimization results to file.

        Args:
            results: Optimization results dictionary
            output_path: Output file path
        """
        if output_path is None:
            output_path = f"optimization_results_{self.study_name}.json"

        try:
            import json

            with open(output_path, "w") as f:
                # Convert numpy types to Python types for JSON serialization
                json_results = self._make_json_serializable(results)
                json.dump(json_results, f, indent=2)

            logger.info(f"Optimization results saved to {output_path}")

        except Exception as e:
            logger.error(f"Failed to save optimization results: {e}")

    def _make_json_serializable(self, obj: Any) -> Any:
        """
        Convert numpy types to Python types for JSON serialization.

        Args:
            obj: Object to convert

        Returns:
            JSON-serializable object
        """
        import numpy as np

        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {
                key: self._make_json_serializable(value) for key, value in obj.items()
            }
        if isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        return obj


# Global optimizer instance
_optimizer: TradingSystemOptimizer | None = None


def get_optimizer(
    config, data_manager, signal_calculator, risk_manager, backtest_engine
) -> TradingSystemOptimizer:
    """
    Get the global optimizer instance.

    Args:
        config: Configuration object
        data_manager: DataManager instance
        signal_calculator: SignalCalculator instance
        risk_manager: RiskManager instance
        backtest_engine: BacktestEngine instance

    Returns:
        TradingSystemOptimizer instance
    """
    global _optimizer
    if _optimizer is None:
        _optimizer = TradingSystemOptimizer(
            config, data_manager, signal_calculator, risk_manager, backtest_engine
        )
    return _optimizer
