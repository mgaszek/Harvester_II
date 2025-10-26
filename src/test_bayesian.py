#!/usr/bin/env python3
"""
Quick test of the Bayesian State Machine functionality.
"""

from utils import get_bayesian_state_machine
import numpy as np

def main():
    print("Testing Bayesian State Machine...")

    # Test Bayesian State Machine
    bsm = get_bayesian_state_machine()
    if bsm:
        print("OK Bayesian State Machine available")

        # Test with sample features (high panic scenario)
        features = bsm.prepare_features(
            volatility_z=2.0,    # High volatility
            volume_z=1.5,        # Elevated volume
            trends_z=2.5,        # High trends interest
            g_score=2,           # Moderate geo-political risk
            price_change_5d=-0.02  # Slight recent decline
        )

        conviction = bsm.assess_conviction(features)
        print(f"Conviction assessment: {conviction['conviction_level']} ({conviction['confidence']:.2f})")
        print(f"Should trade: {conviction['should_trade']}")
        print(f"Market state: {conviction.get('state', 'unknown')}")
        print(f"Assessment method: {conviction['method']}")

        # Test with calm market scenario
        print("\nTesting calm market scenario...")
        calm_features = bsm.prepare_features(0.1, 0.0, 0.2, 0, 0.01)
        calm_conviction = bsm.assess_conviction(calm_features)
        print(f"Calm market conviction: {calm_conviction['conviction_level']} (should_trade: {calm_conviction['should_trade']})")

    else:
        print("X Bayesian State Machine not available")

if __name__ == "__main__":
    main()
