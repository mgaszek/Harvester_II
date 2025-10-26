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

6. ✅ **Eliminate Duplication and Refine Logic (Quickening Rite)** (Medium, 4-6 hours) - COMPLETED
   Centralize utils.py; add Bayesian State Machine for conviction.
   - Approach: Merge validations; hmmlearn/scikit-learn probs in signals.py (IF/THEN → predict_proba >0.7); requirements.txt add.
   - **COMPLETED**: Added BayesianStateMachine class with HMM for market state modeling (calm/volatile/panic). Centralized validation functions in utils.py. Replaced hard-coded IF/THEN logic with probabilistic conviction assessment using predict_proba > 0.7 threshold. Added fallback to rule-based logic when HMM unavailable.

7. **Standardize Logging/Monitoring** (Medium, 3-5 hours) [ ]  
   Loguru global; JSON toggle.  
   - Approach: Replace getLogger; Prometheus gauges (bias/conviction).

8. **Documentation and Style Polish (Anointment Rite)** (Medium, 3-5 hours) [ ]  
   Pre-commit Ruff/etc.; README tuning/API guide.  
   - Approach: .pre-commit-config.yaml; CRI examples/schema.

9. **Add Advanced Features** (Low, 10-20 hours) [ ]  
   Ccxt multi-exchange; webhooks/Plotly reports; GitHub Actions CI.  
   - Approach: data_manager.py integrate; risk_manager.py alerts; .github/workflows.

10. **Implementing Bayesian State Machine v2 in Harvester II**
This list keeps scope tight: Focus on a lightweight HMM-based BSM (using hmmlearn) for regime detection in signals.py, integrated via DI. No full Bayesian nets or MCMC—aim for <10% code bloat, 5-8 hours total. Builds on existing z-scores/CRI/Panic without rewriting. Test-first to avoid creep. Mark [ ] as done.

Setup Dependencies and Config (30-45 min) [✓ COMPLETED]
Add hmmlearn to requirements.txt (lightweight HMM lib).

Approach: pip install hmmlearn==0.3.2; add to config.json: 'bayesian': {'n_states': 3, 'conviction_thresh': 0.7, 'priors': [0.3, 0.4, 0.3]} (bull/bear/panic priors).
Update di.py: Inject BayesianStateMachine(config) into SignalCalculator.

**COMPLETED**: Added hmmlearn>=0.3.0 and scikit-learn>=1.3.0 to requirements.txt. Added comprehensive Bayesian config section to config.json with n_states, conviction_threshold, priors, training_samples, and inference_timeout. Updated di.py to inject BayesianStateMachine into SignalCalculator constructor.


Prototype Core BSM Class (1-1.5 hours) [✓ COMPLETED]
Create src/bayesian_state.py with minimal HMM.

Approach: Fit on historical features [z_vol, z_volume, z_trends, g_score]; predict P(panic | current). Output: {'conviction': float, 'state_probs': dict}. Reuse utils.py z-scores. Lazy-fit on first call.

**COMPLETED**: Created dedicated src/bayesian_state.py module with production-ready BayesianStateMachine class. Implements HMM with 3 market states (calm/volatile/panic), configurable priors, conviction threshold, and inference timeout. Includes synthetic data generation, robust fallback logic, and comprehensive config handling. Integrated via DI with lazy training on first use.


Integrate into Signals (Quickening Lite) (1-1.5 hours) [ ]
Enhance signals.py: Wrap calculate_panic_score with enhanced_panic = panic * bsm.conviction.

Approach: If conviction > thresh, scale entry signal; else, dampen (e.g., *0.5). Keep deterministic fallback if lib fails.


Basic Unit Tests (1 hour) [ ]
Add to test_signals.py: Mock features, assert conviction ~0.8 for panic sim, ~0.2 for noise.

Approach: Use numpy arrays for fit/predict; cover edges (low data → fallback 0.5 conviction).


Backtest Integration and Validation (1 hour) [ ]
Hook into backtest.py: Run A/B (with/without BSM) on sample period (2020-2021).

Approach: Assert Sharpe improves >5% in test_backtest.py; log conviction in equity_curve for debug.


Refine and Polish (30-45 min) [ ]
Add README section: "Bayesian Quickening" with usage example.

Approach: Config toggle ('bayesian.enabled': true); monitor perf (if >2s inference, disable). Commit to develop branch, PR to main with "Quickening Rite v1".
