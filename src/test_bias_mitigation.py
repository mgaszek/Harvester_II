#!/usr/bin/env python3
"""
Test bias mitigation functionality.
"""

from di import create_components


def test_bias_detection():
    """Test bias detection functionality."""
    print("Testing bias detection...")

    components = create_components()
    engine = components["trading_engine"]

    # Create mock backtest result
    mock_result = {
        "trades": [
            {
                "entry_time": "2020-01-01",
                "exit_time": "2020-01-02",
                "pnl_percentage": 0.02,
            },
            {
                "entry_time": "2020-01-03",
                "exit_time": "2020-01-04",
                "pnl_percentage": -0.01,
            },
        ],
        "capital": {"total_return": 0.15, "sharpe_ratio": 1.8, "max_drawdown": -0.12},
    }

    # Test bias detection
    bias_analysis = engine.detect_backtest_biases(mock_result)

    print("OK Bias detection completed")
    print(f"  Look-ahead bias check: {bias_analysis.get('look_ahead_bias', {})}")
    print(f"  Recommendations: {len(bias_analysis.get('recommendations', []))}")

    return True


def test_walk_forward_structure():
    """Test walk-forward validation structure."""
    print("Testing walk-forward validation structure...")

    components = create_components()
    engine = components["trading_engine"]

    # Test that the method exists and can be called (will fail gracefully without full data)
    try:
        result = engine.run_walk_forward_validation(
            "2020-01-01", "2021-01-01", train_window_months=6, test_window_months=1
        )
        print("OK Walk-forward method callable")
        return True
    except Exception as e:
        print(f"X Walk-forward test failed: {e}")
        return False


def main():
    print("Testing Harvester II Bias Mitigation Framework")
    print("=" * 50)

    success = True

    # Test bias detection
    try:
        success &= test_bias_detection()
    except Exception as e:
        print(f"Bias detection test failed: {e}")
        success = False

    # Test walk-forward structure
    try:
        success &= test_walk_forward_structure()
    except Exception as e:
        print(f"Walk-forward test failed: {e}")
        success = False

    print("=" * 50)
    if success:
        print("OK All bias mitigation tests passed!")
    else:
        print("X Some tests failed")

    return success


if __name__ == "__main__":
    main()
