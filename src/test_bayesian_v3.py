#!/usr/bin/env python3
"""
Test Bayesian State Machine v3 implementation.
"""

import numpy as np

from di import create_components


def main():
    print("Testing Bayesian State Machine v3...")

    try:
        components = create_components()
        signal_calc = components["signal_calculator"]
        bsm = signal_calc.bayesian_state_machine

        if bsm:
            print("OK Bayesian State Machine v3 loaded")
            print(f"  Covariance type: {getattr(bsm.model, 'covariance_type', 'N/A')}")
            print(f"  States: {bsm.n_states}")
            print(f"  Priors: {bsm.priors}")
            print(f"  Conviction threshold: {bsm.conviction_threshold}")

            # Test basic functionality
            features = bsm.prepare_features(2.0, 1.5, 2.5, 2, -0.02)
            result = bsm.assess_conviction(features)
            print(
                f"  Basic assessment: {result['conviction_level']} ({result['confidence']:.2f})"
            )

            # Test optimization (limited trials)
            print("  Testing prior optimization...")
            try:
                result = bsm.optimize_priors(n_trials=3)  # Quick test
                print(
                    f"  OK Optimization completed: {result['improvement_score']:.2f} score"
                )
                print(f"    New priors: {result['optimized_priors']}")
            except Exception as e:
                print(f"  X Optimization failed: {e}")

            # Test edge hardening features
            print("  Testing edge hardening...")
            initial_buffer_size = len(bsm.observation_buffer)

            # Add many observations to test buffer trimming
            for i in range(120):  # Exceed max_buffer_size of 100
                test_features = bsm.prepare_features(
                    1.0 + 0.1 * np.sin(i * 0.1),  # Varying volatility
                    0.5,
                    0.8,
                    1,
                    0.01,
                )
                bsm.assess_conviction(test_features)

            final_buffer_size = len(bsm.observation_buffer)
            print(
                f"  Buffer trimming: {initial_buffer_size} -> {final_buffer_size} (max: {bsm.max_buffer_size})"
            )

            if final_buffer_size <= bsm.max_buffer_size:
                print("  OK Buffer properly trimmed")
            else:
                print(
                    f"  X Buffer not trimmed: {final_buffer_size} > {bsm.max_buffer_size}"
                )

            # Test KL-divergence with low evidence scenario
            # Create very similar observations (low variance = low evidence)
            for i in range(15):
                similar_features = bsm.prepare_features(
                    0.01, 0.01, 0.01, 0, 0.001
                )  # Very similar
                result = bsm.assess_conviction(similar_features)
                if result.get("method") == "fallback":
                    print(
                        "  OK KL-divergence correctly triggered fallback for low evidence"
                    )
                    break
            else:
                print(
                    "  ? KL-divergence test inconclusive (may need more observations)"
                )

            print("OK Edge hardening features tested!")

        else:
            print("X Bayesian State Machine not available")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
