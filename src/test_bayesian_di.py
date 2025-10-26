#!/usr/bin/env python3
"""
Test the modular Bayesian State Machine via dependency injection.
"""

from src.di import create_components


def main():
    print("Testing Bayesian State Machine via DI...")

    try:
        components = create_components()
        signal_calc = components["signal_calculator"]
        bsm = signal_calc.bayesian_state_machine

        if bsm:
            print("OK Bayesian State Machine loaded via DI")
            print(f"  Enabled: {bsm.enabled}")
            print(f"  Trained: {bsm.is_trained}")
            print(f"  States: {bsm.n_states}")
            print(f"  Conviction threshold: {bsm.conviction_threshold}")

            # Test conviction assessment with panic scenario
            features = bsm.prepare_features(
                volatility_z=2.0,  # High volatility
                volume_z=1.5,  # Elevated volume
                trends_z=2.5,  # High trends interest
                g_score=2,  # Moderate geo-political risk
                price_change_5d=-0.02,  # Slight recent decline
            )

            result = bsm.assess_conviction(features)
            print(
                f"  Panic scenario: {result['conviction_level']} ({result['confidence']:.2f})"
            )
            print(f"  Should trade: {result['should_trade']}")
            print(f"  Market state: {result.get('state', 'unknown')}")
            print(f"  Method: {result['method']}")

            # Test with calm scenario
            calm_features = bsm.prepare_features(0.1, 0.0, 0.2, 0, 0.01)
            calm_result = bsm.assess_conviction(calm_features)
            print(
                f"  Calm scenario: {calm_result['conviction_level']} (should_trade: {calm_result['should_trade']})"
            )

        else:
            print("X Bayesian State Machine not available")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
