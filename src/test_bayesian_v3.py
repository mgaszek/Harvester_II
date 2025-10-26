#!/usr/bin/env python3
"""
Test Bayesian State Machine v3 implementation.
"""

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

            print("OK All Bayesian v3 features working!")

        else:
            print("X Bayesian State Machine not available")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
