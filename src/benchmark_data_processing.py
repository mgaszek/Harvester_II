#!/usr/bin/env python3
"""
Benchmark script demonstrating Harvester II data processing performance.
Shows the unified API working with the current backend.
"""

import time

import numpy as np
import pandas as pd

from data_processing import DataProcessor, create_dataframe, create_series


def benchmark_current_backend(n_runs: int = 5):
    """Benchmark the current data processing backend."""
    current_backend = DataProcessor.get_backend()
    print(f"\n[*] Benchmarking {current_backend.upper()} Backend")
    print("=" * 50)
    print(f"Data Processing Backend: {current_backend}")
    print(f"Polars Available: {DataProcessor.is_polars_available()}")
    print()

    # Create test data
    n_rows = 10000
    dates = pd.date_range("2020-01-01", periods=n_rows, freq="D")

    # Test data
    test_data = {
        "close": np.random.randn(n_rows).cumsum() + 100,
        "high": np.random.randn(n_rows).cumsum() + 105,
        "low": np.random.randn(n_rows).cumsum() + 95,
        "volume": np.random.randint(100000, 1000000, n_rows),
    }

    results = {}

    # Benchmark DataFrame creation
    print("Testing DataFrame creation...")
    times = []
    for _ in range(n_runs):
        start_time = time.time()
        df = create_dataframe(test_data, index=dates)
        end_time = time.time()
        times.append(end_time - start_time)
    results["dataframe_creation"] = np.mean(times)
    print(".4f")

    # Benchmark Series operations
    print("Testing Series operations...")
    close_series = (
        pd.Series(test_data["close"], index=dates)
        if current_backend == "pandas"
        else create_series(test_data["close"], index=dates)
    )
    times = []
    for _ in range(n_runs):
        start_time = time.time()
        returns = DataProcessor.pct_change(close_series, periods=1)
        vol = DataProcessor.rolling_std(returns, window=20)
        end_time = time.time()
        times.append(end_time - start_time)
    results["series_operations"] = np.mean(times)
    print(".4f")

    # Benchmark correlation calculation
    print("Testing correlation calculation...")
    series1 = (
        pd.Series(test_data["close"][:5000])
        if current_backend == "pandas"
        else create_series(test_data["close"][:5000])
    )
    series2 = (
        pd.Series(test_data["volume"][:5000])
        if current_backend == "pandas"
        else create_series(test_data["volume"][:5000])
    )
    times = []
    for _ in range(n_runs):
        start_time = time.time()
        corr = DataProcessor.correlation(series1, series2)
        end_time = time.time()
        times.append(end_time - start_time)
    results["correlation"] = np.mean(times)
    print(".4f")

    # Benchmark rolling operations
    print("Testing rolling operations...")
    times = []
    for _ in range(n_runs):
        start_time = time.time()
        sma = DataProcessor.rolling_mean(close_series, window=50)
        end_time = time.time()
        times.append(end_time - start_time)
    results["rolling_mean"] = np.mean(times)
    print(".4f")

    return results


def run_benchmarks():
    """Run comprehensive benchmarks for the current data processing backend."""
    print("[+] Harvester II Data Processing Benchmarks")
    print("=" * 60)
    print("Testing with 10,000 data points across multiple operations")
    print()

    # Run benchmark with current backend
    results = benchmark_current_backend()

    # Show results
    print("\n[*] Benchmark Results (milliseconds)")
    print("=" * 60)
    print("Operation              Time (ms)")
    print("-" * 60)

    operations = [
        "dataframe_creation",
        "series_operations",
        "correlation",
        "rolling_mean",
    ]

    for op in operations:
        time_ms = results[op] * 1000
        op_name = op.replace("_", " ").title()
        print("<25")

    current_backend = DataProcessor.get_backend()
    print(f"\n[*] Current Backend: {current_backend.upper()}")
    print(f"[*] Polars Available: {DataProcessor.is_polars_available()}")

    print("\n[*] Performance Notes:")
    print("- For maximum performance, install polars: pip install polars")
    print("- Switch backends with: export HARVESTER_DATA_BACKEND=polars")
    print("- Default backend is pandas for maximum compatibility")

    if not DataProcessor.is_polars_available():
        print("\n[!] To enable Polars performance boost:")
        print("   1. pip install polars")
        print("   2. export HARVESTER_DATA_BACKEND=polars")
        print("   3. Restart the application")


if __name__ == "__main__":
    run_benchmarks()
