# Harvester II To-Do List

Updated October 26, 2025: This is a flattened, numbered list version for easier tracking. Priorities preserved (Critical first, then High, Medium, Low). Items consolidated to avoid duplicates—e.g., testing expansions merged, no repeated Polars mandates. Total estimated effort: ~60-90 hours. Mark off as [x] when done.

1. **Mitigate Backtesting Pitfalls and Biases** (Critical, 8-12 hours)  
   Address overfitting, look-ahead, survivorship biases.  
   - Approach: In backtest.py, add walk-forward optimization; use survivor-free data (ccxt); assert in test_backtest.py.  
   Enhance realism with slippage, commissions, liquidity.  
   - Approach: Volume-based impact in _execute_entry_backtest; config.json toggles.

2. **Add License and Repo Polish** (Critical, 2-4 hours)  
   Include open-source license (e.g., MIT).  
   - Approach: Add LICENSE file; update README badges (coverage, build).  
   Secure .env/.gitignore.  
   - Approach: Confirm gitignore excludes .env/data/*.db; add prod secrets guide in README.

3. **Integrate Mature Backtesting Library** (High, 12-18 hours)  
   Replace custom BacktestEngine with vectorbt or Backtesting.py.  
   - Approach: Refactor backtest.py to vectorbt for speed; plugin custom signals; update test_backtest.py.

4. **Expand and Automate Testing** (High, 10-15 hours)  
   Achieve 90% coverage with pytest-cov; add bias assertions and e2e trade cycles.  
   - Approach: Integrate cov in pytest.ini; add full cycles in test_integration.py; mock positions/asynctest for risk_manager.py/data_manager.py.

5. **Optimize Data Processing Fully** (High, 6-8 hours)  
   Mandate Polars for all Pandas ops.  
   - Approach: Migrate utils.py/signals.py; benchmark in README; pin polars in requirements.txt.

6. **Eliminate Remaining Duplication and Refine Logic** (Medium, 4-6 hours)  
   Centralize validations/calcs in utils.py.  
   - Approach: Merge _validate_symbol; remove dupes from signals.py/data_manager.py.  
   Add ML for signal weighting (scikit-learn).  
   - Approach: Optional regressor in signals.py for Panic; add to requirements.txt.

7. **Standardize Logging/Monitoring** (Medium, 3-5 hours)  
   Use Loguru everywhere; enable JSON prod mode.  
   - Approach: Replace logging.getLogger in all src/; config.json toggle.  
   Enhance Prometheus with bias/drawdown metrics.  
   - Approach: Gauges in engine.py/backtest.py; Grafana setup in README.

8. **Documentation and Style Polish** (Medium, 3-5 hours)  
   Enforce Ruff/black/mypy via pre-commit.  
   - Approach: Add .pre-commit-config.yaml; run on repo; update README setup.  
   Expand README: Signal tuning examples, API key guide, config.json schema.  
   - Approach: Add sections for customization.

9. **Add Advanced Features** (Low, 10-20 hours)  
   Multi-exchange via ccxt.  
   - Approach: Integrate in data_manager.py/utils.py for is_market_open().  
   Webhook alerts + Plotly CLI reports.  
   - Approach: requests.post in risk_manager.py; --report flag in main.py.  
   CI/CD with GitHub Actions.  
   - Approach: .github/workflows for tests/lint on push.