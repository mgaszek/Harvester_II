1. Historical Performance Baseline & Bias Checks (2 hours) [✅]
Run walk-forward backtest (2020-2025, 6-month train/test splits); capture Sharpe, max drawdown, win rate.

Approach: backtest.py add run_walk_forward(start='2020-01-01', end='2025-10-26', step='6M'); assert look-ahead free (no future peeks). Use ccxt for survivor-free universe (dynamic asset list). Log to logs/baseline_2025.log.


2. Simplify & Baseline Comparison (1.5 hours) [✅]
Test CRI/Panic without BSM (config: 'bayesian.enabled': false); compare Sharpe vs. v3.

Approach: test_backtest.py parametrize @pytest.mark.parametrize('bsm_enabled', [True, False]); assert v3 > baseline +0.1. Log equity curves to Prometheus.


3. Stress-Test Risk Management (1.5 hours) [✅]
Simulate 2008/2020/2022 drawdowns; test multi-signal spikes (10 assets firing).

Approach: test_risk_manager.py mock 10 simultaneous Panic>3.0; assert drawdown <2%. Add risk_manager.py correlation check (np.corrcoef on signals).


4. Execution Realism (1 hour) [✅]
Enhance backtest.py: Bid/ask spread (0.05%), volume-based slippage (0.1% * volume/avg_volume), fill delay (1min).

Approach: execute_entry_backtest apply spread/slippage; test_backtest.py mock low-liquidity assets (e.g., small-cap).


5. Data Latency & Fallbacks (1 hour) [✅]
Log Trends latency in data_manager.py; abstain if >24h stale (conviction=0).

Approach: signals.py check timestamp; test_signals.py mock stale data (conviction<0.1). Swap yfinance for ccxt in requirements.txt.


6. README & Live Prep (1 hour) [✅]
Add “Quickening v3” section: Decay math, priors example. Stub Alpaca API in portfolio.py.

Approach: README snippet for predict_conviction; portfolio.py add execute_live_alpaca. Badge cov/CI.