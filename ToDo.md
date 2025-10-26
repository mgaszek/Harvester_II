Step 1: Foundations - Security and Testing (Priority: Critical)
Before adding new features, we must ensure the system is secure and fully testable.
Implement Full Database Encryption (4-6 hours)
Task: Integrate pysqlcipher3 to enable full encryption for the SQLite database files (data.db, portfolio.db).
Approach: In models.py, add conditional logic that creates the SQLAlchemy engine using SQLCipher if "encrypted_db": true is set in config.json. The encryption key must be stored in an environment variable (.env).
Unify and Secure Logging (3-5 hours)
Task: Replace all standard logging instances with Loguru. Implement a global SensitiveDataFilter for all handlers to automatically redact API keys and other sensitive data.
Approach: Configure Loguru in main.py and engine.py. Apply the SensitiveDataFilter class to every logger.add() call to ensure consistent data masking across the application. Enable JSON serialization for production logs.
Expand and Automate Testing (15-25 hours)
Task: Achieve 80-90% code coverage. Create unit tests for key calculations (CRI, Panic Score, position sizing, trailing stops). Mock all external APIs (yfinance, pytrends) using responses or aiohttp-mock.
Approach: Expand test_signals.py and test_risk_manager.py. In test_integration.py, add tests for asynchronous data fetching. Use pytest-cov to monitor coverage. Configure CI (e.g., GitHub Actions) with pre-commit and Ruff to automatically run tests and linters.
Step 2: Backtesting Engine - Realism and Reliability (Priority: High)
The current engine is good, but replacing it with a mature library will eliminate potential biases and accelerate development.
Integrate a Professional Backtesting Library (12-18 hours)
Task: Replace the current custom BacktestEngine with the vectorbt library. This will provide vectorization (massive speed boost) and built-in mechanisms to prevent common pitfalls.
Approach: Create a new implementation of backtest.py that uses vectorbt. Adapt your custom signals (Panic Score, CRI) to function as "indicators" that can be plugged into the vectorbt engine.
Enhance Simulation Realism (8-12 hours)
Task: Ensure the new backtesting engine accounts for key market factors:
Costs: Model commissions and volume-based slippage.
Data: Integrate loading of historical Google Trends from CSV files. Ensure the use of data free from "survivorship bias" (e.g., from paid sources or specialized libraries).
Calendar: Expand holiday handling to include global exchange calendars if you plan to trade in other markets.
Approach: Configure slippage and commission parameters in vectorbt. Add a function to data_manager.py to load data from CSVs as an alternative to the API.
Step 3: Optimization and Refactoring (Priority: High)
With the new backtesting engine in place, it's time to optimize data processing and finalize the codebase.
Full Migration to Polars and Data Optimization (8-12 hours)
Task: Replace Pandas with Polars for all key data operations (signals, indicators, data management). Implement a retry mechanism with exponential backoff for all network requests.
Approach: Rewrite the logic in signals.py, data_manager.py, and utils.py using Polars syntax. Use the tenacity library to wrap the asynchronous data-fetching functions in data_manager.py.
Final Refactoring and Duplicate Elimination (4-6 hours)
Task: Move all reusable calculation functions (like Z-score, ATR) into utils.py to create a single source of truth. Make the Panic Score components weighted based on parameters from config.json.
Approach: Audit signals.py and risk_manager.py. Move all generic calculations to utils.py and import them where needed. Ensure that calculate_panic_score multiplies the individual Z-scores by their respective weights from the configuration.
Step 4: Monitoring and Feature Enhancements (Priority: Medium & Low)
The system is now stable, fast, and reliable. It's time to add features that enhance usability and functionality.
Expand Monitoring and Alerts (4-6 hours)
Task: Enhance Prometheus metrics to include backtest results (e.g., Sharpe ratio, max drawdown). Configure webhook alerts (e.g., to Slack/Discord) for critical events like a daily drawdown breach.
Approach: In engine.py and backtest.py, add new Gauge metrics. In risk_manager.py, within the check_drawdown_limit method, add a requests.post() call to a webhook URL if the limit is breached.
Add Advanced Features (10-20 hours)
Task: Introduce new capabilities to extend the system's usefulness:
Broker Integration: Add support for a broker API (e.g., Alpaca) in portfolio.py for live trade execution.
CLI Reports: Create a --report flag in main.py that generates an interactive HTML report of the backtest's equity curve using Plotly.
Multi-Exchange Support: Integrate the ccxt library into data_manager.py and utils.py to fetch data and check market hours for various crypto and traditional exchanges.
ML Signals (Optional): In signals.py, add an experimental feature to use a scikit-learn model (e.g., logistic regression) to dynamically weigh Panic Score components or to evaluate the G-Score.
Documentation and Final Polish (3-5 hours)
Task: Update README.md with information about the new libraries (vectorbt, Polars), new features (trailing stops, reports), and CI/CD status (e.g., code coverage badges). Run Ruff and black on the entire project to ensure consistent style.
Approach: Add new sections to README.md. Ensure the pre-commit configuration is set up and working correctly.