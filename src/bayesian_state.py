"""
Bayesian State Machine for Harvester II trading system.
Implements Hidden Markov Model (HMM) for market regime detection and signal conviction assessment.
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

# Machine learning imports
try:
    from hmmlearn import hmm
    from sklearn.preprocessing import StandardScaler
    HMM_AVAILABLE = True
except ImportError:
    hmm = None
    StandardScaler = None
    HMM_AVAILABLE = False


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

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Bayesian State Machine.

        Args:
            config: Configuration dictionary with Bayesian settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Configuration parameters
        self.enabled = config.get('bayesian.enabled', True)
        self.n_states = config.get('bayesian.n_states', 3)
        self.conviction_threshold = config.get('bayesian.conviction_threshold', 0.7)
        self.priors = config.get('bayesian.priors', [0.3, 0.4, 0.3])
        self.training_samples = config.get('bayesian.training_samples', 1000)
        self.inference_timeout = config.get('bayesian.inference_timeout', 2.0)

        # Internal state
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.last_inference_time = None

        if not self.enabled:
            self.logger.info("Bayesian State Machine disabled by configuration")
            return

        if not HMM_AVAILABLE:
            self.logger.warning("HMM dependencies not available - Bayesian State Machine disabled")
            return

        # Initialize HMM model
        self.model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=100,
            random_state=42
        )
        self.scaler = StandardScaler()

        self.logger.info(f"Bayesian State Machine initialized with {self.n_states} states")

    def prepare_features(self, volatility_z: float, volume_z: float, trends_z: float,
                        g_score: float, price_change_5d: float) -> np.ndarray:
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

    def train(self, historical_features: Optional[np.ndarray] = None) -> bool:
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
                historical_features = self.generate_synthetic_training_data(self.training_samples)

            # Scale features
            scaled_features = self.scaler.fit_transform(historical_features)

            # Train model
            self.model.fit(scaled_features)
            self.is_trained = True

            self.logger.info(f"Bayesian State Machine trained on {len(historical_features)} samples")
            return True

        except Exception as e:
            self.logger.error(f"Failed to train Bayesian State Machine: {e}")
            return False

    def assess_conviction(self, features: np.ndarray) -> Dict[str, Any]:
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

            # Scale features
            scaled_features = self.scaler.transform(features)

            # Get state probabilities
            state_probs = self.model.predict_proba(scaled_features)[0]

            # Determine most likely state and conviction
            most_likely_state = np.argmax(state_probs)
            max_probability = np.max(state_probs)

            # State interpretation (assuming states are ordered: 0=calm, 1=volatile, 2=panic)
            state_names = ['calm', 'volatile', 'panic']
            state_name = state_names[min(most_likely_state, len(state_names) - 1)]

            # Conviction levels based on probability confidence
            if max_probability >= self.conviction_threshold:
                conviction_level = 'high'
                should_trade = (state_name in ['volatile', 'panic'])  # Only trade in stressed markets
            elif max_probability >= 0.5:
                conviction_level = 'medium'
                should_trade = False  # Too uncertain
            else:
                conviction_level = 'low'
                should_trade = False  # Not confident enough

            # Check inference timeout
            inference_time = (datetime.now() - start_time).total_seconds()
            if inference_time > self.inference_timeout:
                self.logger.warning(f"HMM inference took {inference_time:.2f}s, exceeding timeout {self.inference_timeout}s")
                return self._fallback_assessment(features)

            self.last_inference_time = inference_time

            return {
                'conviction_level': conviction_level,
                'should_trade': should_trade,
                'state': state_name,
                'confidence': float(max_probability),
                'state_probabilities': state_probs.tolist(),
                'inference_time': inference_time,
                'method': 'hmm'
            }

        except Exception as e:
            self.logger.error(f"HMM conviction assessment failed: {e}")
            return self._fallback_assessment(features)

    def _fallback_assessment(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Fallback rule-based conviction assessment.

        Args:
            features: Feature vector

        Returns:
            Dictionary with rule-based assessment
        """
        try:
            volatility_z, volume_z, trends_z, g_score, price_change_5d = features[0]

            # Calculate composite panic score
            panic_score = (abs(volatility_z) + abs(volume_z) + abs(trends_z)) / 3

            # Rule-based logic (original system thresholds)
            if panic_score > 3.0 and g_score >= 1:
                conviction_level = 'high'
                should_trade = True
            elif panic_score > 2.0:
                conviction_level = 'medium'
                should_trade = False
            else:
                conviction_level = 'low'
                should_trade = False

            return {
                'conviction_level': conviction_level,
                'should_trade': should_trade,
                'panic_score': float(panic_score),
                'confidence': min(panic_score / 4.0, 1.0),  # Normalized confidence
                'method': 'rules'
            }

        except Exception as e:
            self.logger.error(f"Rule-based conviction assessment failed: {e}")
            return {
                'conviction_level': 'low',
                'should_trade': False,
                'confidence': 0.0,
                'method': 'fallback'
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
            scenario = np.random.choice(['calm', 'volatile', 'panic'], p=self.priors)

            if scenario == 'calm':
                volatility_z = np.random.normal(0, 0.5)
                volume_z = np.random.normal(0, 0.3)
                trends_z = np.random.normal(0, 0.4)
                g_score = np.random.choice([0, 1], p=[0.9, 0.1])
                price_change_5d = np.random.normal(0, 0.02)

            elif scenario == 'volatile':
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

            features.append([volatility_z, volume_z, trends_z, g_score, price_change_5d])

        return np.array(features)

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for the Bayesian State Machine.

        Returns:
            Dictionary with performance metrics
        """
        return {
            'enabled': self.enabled,
            'is_trained': self.is_trained,
            'n_states': self.n_states,
            'conviction_threshold': self.conviction_threshold,
            'last_inference_time': self.last_inference_time,
            'available': HMM_AVAILABLE
        }


# Global Bayesian State Machine instance
_bayesian_state_machine: Optional[BayesianStateMachine] = None


def get_bayesian_state_machine(config) -> Optional[BayesianStateMachine]:
    """
    Get the global Bayesian State Machine instance.

    Args:
        config: Config object or configuration dictionary

    Returns:
        BayesianStateMachine instance or None if not available
    """
    global _bayesian_state_machine
    if _bayesian_state_machine is None:
        # Handle both Config object and raw dict
        if hasattr(config, '_config_data'):
            config_data = config._config_data
        else:
            config_data = config

        _bayesian_state_machine = BayesianStateMachine(config_data)
        # Train with synthetic data if no real training data available
        if _bayesian_state_machine.enabled and HMM_AVAILABLE:
            _bayesian_state_machine.train()
    return _bayesian_state_machine
