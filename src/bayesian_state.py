"""
Bayesian State Machine for Harvester II trading system.
Implements Hidden Markov Model (HMM) for market regime detection and signal conviction assessment.
"""

from datetime import datetime
import logging
from typing import Any

import numpy as np

# Machine learning imports
try:
    from hmmlearn import hmm
    import optuna
    from sklearn.preprocessing import StandardScaler

    HMM_AVAILABLE = True
    OPTUNA_AVAILABLE = True
except ImportError:
    hmm = None
    StandardScaler = None
    optuna = None
    HMM_AVAILABLE = False
    OPTUNA_AVAILABLE = False


class BayesianStateMachine:
    """
    Bayesian State Machine for market regime detection using Hidden Markov Models.

    Models market states (calm, volatile, panic) and provides probabilistic
    conviction levels for trading signals instead of hard-coded thresholds.

    Features:
    - HMM-based regime detection
    - Probabilistic signal conviction
    - Fallback to rule-based logic
    - Configurable state priors
    """

    def __init__(self, config: dict[str, Any], data_manager=None):
        """
        Initialize the Bayesian State Machine.

        Args:
            config: Configuration dictionary with Bayesian settings
            data_manager: Optional DataManager for caching posterior probabilities
        """
        self.config = config
        self.data_manager = data_manager
        self.logger = logging.getLogger(__name__)

        # Configuration parameters
        self.enabled = config.get("bayesian.enabled", True)
        self.n_states = config.get("bayesian.n_states", 3)
        self.conviction_threshold = config.get("bayesian.conviction_threshold", 0.7)
        self.priors = config.get("bayesian.priors", [0.3, 0.4, 0.3])
        self.training_samples = config.get("bayesian.training_samples", 1000)
        self.inference_timeout = config.get("bayesian.inference_timeout", 2.0)

        # Internal state
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.last_inference_time: float | None = None

        # Online learning buffer for recent observations
        self.observation_buffer: list[np.ndarray] = []
        self.max_buffer_size = 100

        if not self.enabled:
            self.logger.info("Bayesian State Machine disabled by configuration")
            return

        if not HMM_AVAILABLE:
            self.logger.warning(
                "HMM dependencies not available - Bayesian State Machine disabled"
            )
            return

        # Initialize HMM model with enhanced covariance
        covariance_type = self.config.get("bayesian.covariance_type", "full")
        self.model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type=covariance_type,
            n_iter=100,
            random_state=42,
        )
        self.scaler = StandardScaler()

        self.logger.info(
            f"Bayesian State Machine initialized with {self.n_states} states"
        )

    def prepare_features(
        self,
        volatility_z: float,
        volume_z: float,
        trends_z: float,
        g_score: float,
        price_change_5d: float,
    ) -> np.ndarray:
        """
        Prepare feature vector for state machine input.

        Args:
            volatility_z: Volatility z-score
            volume_z: Volume z-score
            trends_z: Trends z-score
            g_score: Geopolitical score
            price_change_5d: 5-day price change

        Returns:
            Feature vector as numpy array
        """
        return np.array([[volatility_z, volume_z, trends_z, g_score, price_change_5d]])

    def train(self, historical_features: np.ndarray | None = None) -> bool:
        """
        Train the HMM on historical market data.

        Args:
            historical_features: Historical feature matrix (n_samples, n_features)
                                 If None, uses synthetic data

        Returns:
            True if training successful
        """
        if not self.enabled or not HMM_AVAILABLE or self.model is None:
            return False

        try:
            # Use provided data or generate synthetic data
            if historical_features is None:
                historical_features = self.generate_synthetic_training_data(
                    self.training_samples
                )

            # Scale features
            scaled_features = self.scaler.fit_transform(historical_features)

            # Train model
            self.model.fit(scaled_features)
            self.is_trained = True

            self.logger.info(
                f"Bayesian State Machine trained on {len(historical_features)} samples"
            )
            return True

        except Exception as e:
            self.logger.error(f"Failed to train Bayesian State Machine: {e}")
            return False

    def assess_conviction(self, features: np.ndarray) -> dict[str, Any]:
        """
        Assess signal conviction using the trained HMM.

        Args:
            features: Current feature vector

        Returns:
            Dictionary with conviction assessment
        """
        if not self.enabled:
            return self._fallback_assessment(features)

        if not self.is_trained:
            # Try to train on first use
            if not self.train():
                return self._fallback_assessment(features)

        try:
            start_time = datetime.now()

            # Check cache for recent posterior calculations
            features_key = None
            if self.data_manager:
                # Create a cache key from the features (rounded to reduce cache misses)
                features_key = ",".join(f"{x:.2f}" for x in features.flatten())
                cached_result = self.data_manager.get_cached_posterior(features_key)
                if cached_result:
                    self.logger.debug("Using cached posterior probabilities")
                    return cached_result

            # Scale features
            scaled_features = self.scaler.transform(features)

            # Get state probabilities
            state_probs = self.model.predict_proba(scaled_features)[0]

            # Add to observation buffer for online learning
            self.observation_buffer.append(scaled_features[0])
            if len(self.observation_buffer) > self.max_buffer_size:
                self.observation_buffer = self.observation_buffer[
                    -self.max_buffer_size :
                ]

            # Check KL-divergence for evidence quality (abstain if low evidence)
            if (
                len(self.observation_buffer) >= 10
            ):  # Need minimum samples for KL-divergence
                kl_divergence = self._calculate_kl_divergence()
                evidence_threshold = 0.1  # KL-divergence threshold for low evidence

                if kl_divergence < evidence_threshold:
                    self.logger.debug(
                        f"Low evidence signal (KL-divergence: {kl_divergence:.3f} < {evidence_threshold})"
                    )
                    return self._fallback_assessment(features)

            # Determine most likely state and conviction
            most_likely_state = np.argmax(state_probs)
            max_probability = np.max(state_probs)

            # State interpretation (assuming states are ordered: 0=calm, 1=volatile, 2=panic)
            state_names = ["calm", "volatile", "panic"]
            state_name = state_names[min(most_likely_state, len(state_names) - 1)]

            # Conviction levels based on probability confidence
            if max_probability >= self.conviction_threshold:
                conviction_level = "high"
                should_trade = state_name in [
                    "volatile",
                    "panic",
                ]  # Only trade in stressed markets
            elif max_probability >= 0.5:
                conviction_level = "medium"
                should_trade = False  # Too uncertain
            else:
                conviction_level = "low"
                should_trade = False  # Not confident enough

            # Check inference timeout
            inference_time = (datetime.now() - start_time).total_seconds()
            if inference_time > self.inference_timeout:
                self.logger.warning(
                    f"HMM inference took {inference_time:.2f}s, exceeding timeout {self.inference_timeout}s"
                )
                return self._fallback_assessment(features)

            self.last_inference_time = inference_time

            result = {
                "conviction_level": conviction_level,
                "should_trade": should_trade,
                "state": state_name,
                "confidence": float(max_probability),
                "state_probabilities": state_probs.tolist(),
                "inference_time": inference_time,
                "method": "hmm",
            }

            # Cache the result for future use
            if self.data_manager and features_key:
                self.data_manager.cache_posterior(features_key, result)

            return result

        except Exception as e:
            self.logger.warning(
                f"HMM conviction assessment failed: {e} - falling back to rule-based logic"
            )
            return self._fallback_assessment(features)

    def _fallback_assessment(self, features: np.ndarray) -> dict[str, Any]:
        """
        Enhanced fallback rule-based conviction assessment with default conviction.

        Args:
            features: Feature vector

        Returns:
            Dictionary with rule-based assessment
        """
        try:
            # Check if we have enough data for proper assessment
            if len(features) == 0 or len(features[0]) < 5:
                self.logger.warning(
                    "Insufficient feature data - using default conviction"
                )
                return self._default_conviction_assessment()

            volatility_z, volume_z, trends_z, g_score, price_change_5d = features[0]

            # Validate feature values
            if any(
                np.isnan([volatility_z, volume_z, trends_z, g_score, price_change_5d])
            ):
                self.logger.warning("NaN values in features - using default conviction")
                return self._default_conviction_assessment()

            # Calculate composite panic score
            panic_score = (abs(volatility_z) + abs(volume_z) + abs(trends_z)) / 3

            # Enhanced rule-based logic with better thresholds
            if panic_score > 3.0 and g_score >= 1:
                conviction_level = "high"
                should_trade = True
            elif panic_score > 2.0:
                conviction_level = "medium"
                should_trade = False
            else:
                conviction_level = "low"
                should_trade = False

            return {
                "conviction_level": conviction_level,
                "should_trade": should_trade,
                "panic_score": float(panic_score),
                "confidence": min(panic_score / 4.0, 1.0),  # Normalized confidence
                "method": "rules",
            }

        except Exception as e:
            self.logger.warning(
                f"Rule-based conviction assessment failed: {e} - using default conviction"
            )
            return self._default_conviction_assessment()

    def _default_conviction_assessment(self) -> dict[str, Any]:
        """
        Default conviction assessment when other methods fail.

        Returns:
            Dictionary with default assessment (medium conviction)
        """
        return {
            "conviction_level": "medium",
            "should_trade": False,
            "confidence": 0.5,  # Default medium conviction
            "method": "default",
        }

    def generate_synthetic_training_data(self, n_samples: int = 1000) -> np.ndarray:
        """
        Generate synthetic training data for HMM training.

        Args:
            n_samples: Number of samples to generate

        Returns:
            Synthetic feature matrix
        """
        np.random.seed(42)  # For reproducibility

        features = []

        for _ in range(n_samples):
            # Generate market scenarios based on priors
            scenario = np.random.choice(["calm", "volatile", "panic"], p=self.priors)

            if scenario == "calm":
                volatility_z = np.random.normal(0, 0.5)
                volume_z = np.random.normal(0, 0.3)
                trends_z = np.random.normal(0, 0.4)
                g_score = np.random.choice([0, 1], p=[0.9, 0.1])
                price_change_5d = np.random.normal(0, 0.02)

            elif scenario == "volatile":
                volatility_z = np.random.normal(1.5, 0.8)
                volume_z = np.random.normal(1.2, 0.6)
                trends_z = np.random.normal(0.8, 0.7)
                g_score = np.random.choice([0, 1, 2], p=[0.4, 0.4, 0.2])
                price_change_5d = np.random.normal(0, 0.05)

            else:  # panic
                volatility_z = np.random.normal(3.0, 1.0)
                volume_z = np.random.normal(2.5, 0.8)
                trends_z = np.random.normal(2.0, 0.9)
                g_score = np.random.choice([1, 2, 3], p=[0.3, 0.4, 0.3])
                price_change_5d = np.random.normal(0, 0.08)

            features.append(
                [volatility_z, volume_z, trends_z, g_score, price_change_5d]
            )

        return np.array(features)

    def optimize_priors(
        self, historical_data: np.ndarray | None = None, n_trials: int = 20
    ) -> dict[str, Any]:
        """
        Optimize market state priors using Optuna Bayesian optimization.

        Args:
            historical_data: Historical feature data for optimization
            n_trials: Number of optimization trials

        Returns:
            Dictionary with optimized priors and performance metrics
        """
        if not OPTUNA_AVAILABLE:
            self.logger.warning("Optuna not available - cannot optimize priors")
            return {"error": "Optuna not available"}

        try:
            # Use provided data or generate synthetic data
            if historical_data is None:
                historical_data = self.generate_synthetic_training_data(2000)

            def objective(trial):
                # Suggest priors that sum to 1.0
                calm_prior = trial.suggest_float("calm_prior", 0.1, 0.5)
                volatile_prior = trial.suggest_float("volatile_prior", 0.2, 0.6)
                # Panic prior is the remainder
                panic_prior = 1.0 - calm_prior - volatile_prior

                if panic_prior < 0.1 or panic_prior > 0.5:
                    return -float("inf")  # Invalid priors

                test_priors = [calm_prior, volatile_prior, panic_prior]

                # Create temporary model with test priors
                temp_model = hmm.GaussianHMM(
                    n_components=self.n_states,
                    covariance_type="full",
                    n_iter=50,
                    random_state=42,
                )

                # Generate training data with test priors
                test_data = self._generate_data_with_priors(
                    historical_data.shape[0], test_priors
                )

                try:
                    # Scale and fit
                    scaled_data = self.scaler.fit_transform(test_data)
                    temp_model.fit(scaled_data)

                    # Score based on log-likelihood
                    score = temp_model.score(scaled_data)

                    # Add penalty for extreme priors
                    prior_penalty = (
                        abs(calm_prior - 0.3)
                        + abs(volatile_prior - 0.4)
                        + abs(panic_prior - 0.3)
                    )

                    return (
                        score - prior_penalty * 10
                    )  # Penalize deviation from defaults

                except Exception:
                    return -float("inf")

            # Run optimization
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=n_trials)

            best_params = study.best_params
            best_calm = best_params["calm_prior"]
            best_volatile = best_params["volatile_prior"]
            best_panic = 1.0 - best_calm - best_volatile

            optimized_priors = [best_calm, best_volatile, best_panic]

            result = {
                "optimized_priors": optimized_priors,
                "original_priors": self.priors.copy(),
                "improvement_score": study.best_value,
                "n_trials": n_trials,
                "best_trial": study.best_trial.number,
            }

            # Update the instance priors
            self.priors = optimized_priors
            self.logger.info(
                f"Optimized priors: calm={best_calm:.3f}, volatile={best_volatile:.3f}, panic={best_panic:.3f}"
            )

            return result

        except Exception as e:
            self.logger.error(f"Prior optimization failed: {e}")
            return {"error": str(e)}

    def _generate_data_with_priors(
        self, n_samples: int, priors: list[float]
    ) -> np.ndarray:
        """
        Generate synthetic training data with specific priors.

        Args:
            n_samples: Number of samples to generate
            priors: Prior probabilities for each state

        Returns:
            Generated feature matrix
        """
        np.random.seed(42)  # For reproducibility

        features = []
        priors_array = np.array(priors)

        for _ in range(n_samples):
            # Sample state based on priors
            state = np.random.choice(3, p=priors_array)

            if state == 0:  # calm
                volatility_z = np.random.normal(0, 0.5)
                volume_z = np.random.normal(0, 0.3)
                trends_z = np.random.normal(0, 0.4)
                g_score = np.random.choice([0, 1], p=[0.9, 0.1])
                price_change_5d = np.random.normal(0, 0.02)

            elif state == 1:  # volatile
                volatility_z = np.random.normal(1.5, 0.8)
                volume_z = np.random.normal(1.2, 0.6)
                trends_z = np.random.normal(0.8, 0.7)
                g_score = np.random.choice([0, 1, 2], p=[0.4, 0.4, 0.2])
                price_change_5d = np.random.normal(0, 0.05)

            else:  # panic (state == 2)
                volatility_z = np.random.normal(3.0, 1.0)
                volume_z = np.random.normal(2.5, 0.8)
                trends_z = np.random.normal(2.0, 0.9)
                g_score = np.random.choice([1, 2, 3], p=[0.3, 0.4, 0.3])
                price_change_5d = np.random.normal(0, 0.08)

            features.append(
                [volatility_z, volume_z, trends_z, g_score, price_change_5d]
            )

        return np.array(features)

    def _calculate_kl_divergence(self) -> float:
        """
        Calculate KL-divergence between recent observations and model predictions.

        Returns:
            KL-divergence value (lower = more similar distributions)
        """
        if len(self.observation_buffer) < 10:
            return 0.0

        try:
            # Convert buffer to numpy array
            recent_obs = np.array(
                self.observation_buffer[-20:]
            )  # Use last 20 observations

            # Get empirical distribution (what we've observed)
            obs_mean = np.mean(recent_obs, axis=0)
            obs_cov = np.cov(recent_obs.T) + 1e-6 * np.eye(
                recent_obs.shape[1]
            )  # Add regularization

            # Get model-predicted distribution for each state
            state_means = []
            state_covs = []

            for state_idx in range(self.n_states):
                # Extract state parameters from HMM
                state_mean = self.model.means_[state_idx]
                state_cov = self.model.covars_[state_idx]

                # Convert to numpy arrays if needed
                state_means.append(np.array(state_mean))
                state_covs.append(np.array(state_cov))

            # Calculate KL-divergence to each state and take minimum
            # (how well do observations match any state?)
            kl_divs = []

            for state_mean, state_cov in zip(state_means, state_covs, strict=False):
                try:
                    # KL(P||Q) where P is observations, Q is state distribution
                    # KL(P||Q) = 0.5 * [trace(Q^-1 * P) + (μ_Q - μ_P)^T Q^-1 (μ_Q - μ_P) - k + ln(det(Q)/det(P))]
                    cov_inv = np.linalg.inv(state_cov)
                    mean_diff = state_mean - obs_mean

                    trace_term = np.trace(cov_inv @ obs_cov)
                    mean_term = mean_diff.T @ cov_inv @ mean_diff
                    det_term = np.log(np.linalg.det(state_cov) / np.linalg.det(obs_cov))

                    kl_div = 0.5 * (trace_term + mean_term - len(obs_mean) + det_term)
                    kl_divs.append(max(0, kl_div))  # Ensure non-negative

                except np.linalg.LinAlgError:
                    # If matrix inversion fails, use large KL-divergence
                    kl_divs.append(10.0)

            # Return minimum KL-divergence (best match to any state)
            return min(kl_divs) if kl_divs else 10.0

        except Exception as e:
            self.logger.debug(f"KL-divergence calculation failed: {e}")
            return 10.0  # High divergence indicates low evidence

    def get_performance_stats(self) -> dict[str, Any]:
        """
        Get performance statistics for the Bayesian State Machine.

        Returns:
            Dictionary with performance metrics
        """
        return {
            "enabled": self.enabled,
            "is_trained": self.is_trained,
            "n_states": self.n_states,
            "conviction_threshold": self.conviction_threshold,
            "last_inference_time": self.last_inference_time,
            "available": HMM_AVAILABLE,
        }


# Global Bayesian State Machine instance
_bayesian_state_machine: BayesianStateMachine | None = None


def get_bayesian_state_machine(
    config, data_manager=None
) -> BayesianStateMachine | None:
    """
    Get the global Bayesian State Machine instance.

    Args:
        config: Config object or configuration dictionary
        data_manager: Optional DataManager for caching

    Returns:
        BayesianStateMachine instance or None if not available
    """
    global _bayesian_state_machine
    if _bayesian_state_machine is None:
        # Handle both Config object and raw dict
        if hasattr(config, "_config_data"):
            config_data = config._config_data
        else:
            config_data = config

        _bayesian_state_machine = BayesianStateMachine(config_data, data_manager)
        # Train with synthetic data if no real training data available
        if _bayesian_state_machine.enabled and HMM_AVAILABLE:
            _bayesian_state_machine.train()
    return _bayesian_state_machine
