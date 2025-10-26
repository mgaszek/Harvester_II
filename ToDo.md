# Harvester II To-Do List

Updated October 26, 2025: Infused with the Inquisition's rites. Numbered for tracking; [ ] unchecked. Effort: ~60-90 hours. Focus: Anointment (CRI enhance), Quickening (Bayesian), Adaptation (feedback).

1. **Mitigate Backtesting Pitfalls and Biases** (Critical, 8-12 hours) [ ]  
   Address overfitting/look-ahead/survivorship.  
   - Approach: backtest.py walk-forward; ccxt survivor-free data; test_backtest.py asserts.

2. **Add License and Repo Polish** (Critical, 2-4 hours) [ ]  
   MIT LICENSE; README badges/secrets guide.  
   - Approach: .gitignore confirm; prod env vars section.

3. **Integrate Mature Backtesting Library** (High, 12-18 hours) [ ]  
   Vectorbt/Backtesting.py refactor.  
   - Approach: backtest.py vectorize; plugin signals; test_backtest.py update.

4. **Expand Testing with Adaptation Loop** (High, 10-15 hours) [ ]  
   90% cov (pytest-cov); bias/e2e assertions; bridge backtest to live params.  
   - Approach: pytest.ini cov; test_integration.py cycles; optuna feedback in engine.py (Rite of Adaptation).

5. **Optimize Data Processing Fully** (High, 6-8 hours) [ ]  
   Mandate Polars.  
   - Approach: utils.py/signals.py migrate; requirements.txt pin; README benchmark.

6. **Eliminate Duplication and Refine Logic (Quickening Rite)** (Medium, 4-6 hours) [ ]  
   Centralize utils.py; add Bayesian State Machine for conviction.  
   - Approach: Merge validations; hmmlearn/scikit-learn probs in signals.py (IF/THEN → predict_proba >0.7); requirements.txt add.

7. **Standardize Logging/Monitoring** (Medium, 3-5 hours) [ ]  
   Loguru global; JSON toggle.  
   - Approach: Replace getLogger; Prometheus gauges (bias/conviction).

8. **Documentation and Style Polish (Anointment Rite)** (Medium, 3-5 hours) [ ]  
   Pre-commit Ruff/etc.; README tuning/API guide.  
   - Approach: .pre-commit-config.yaml; CRI examples/schema.

9. **Add Advanced Features** (Low, 10-20 hours) [ ]  
   Ccxt multi-exchange; webhooks/Plotly reports; GitHub Actions CI.  
   - Approach: data_manager.py integrate; risk_manager.py alerts; .github/workflows.

The golem stirs—light the fire with Rite #6 (Bayesian soul) for quick wins. What's your first incantation: Code for the State Machine, or a PR scaffold? Let's ascend.