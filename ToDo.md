Refined To-Do: v3 Tempering (3-5 Hours) - COMPLETED ✅

Prior Opt & A/B Backtest (1.5 hours) [✅]
test_backtest.py: Optuna priors (5 trials for testing); assert v3 Sharpe improvement.

Approach: study.optimize on bayesian_state.py priors - IMPLEMENTED


Edge Hardening (1 hour) [✅]
Buffer trim on >100; KL-div abstain if low evidence.

Approach: bayesian_state.py buffer trimming + KL-divergence checks - IMPLEMENTED


Perf Cache & CI (1 hour) [✅]
TTLCache posteriors (5min); .github/workflows pytest-cov.

Approach: data_manager.py extend; Actions lint/tests - IMPLEMENTED


README Rite (30 min) [✅]
"v3 Quickening" section: Priors example, decay math.

Approach: Include snippet; badge cov - IMPLEMENTED