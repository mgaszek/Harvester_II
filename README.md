# Harvester II Trading System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/your-username/harvester-ii/workflows/CI/badge.svg)](https://github.com/your-username/harvester-ii/actions)
[![codecov](https://codecov.io/gh/your-username/harvester-ii/branch/develop/graph/badge.svg)](https://codecov.io/gh/your-username/harvester-ii)
[![Code Style](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

A sophisticated volatility and attention-driven trading system designed to capture market extremes using crowd psychology and technical analysis. Features enterprise-grade logging, monitoring, and performance optimizations for production deployment.

## 🎯 System Overview

Harvester II operates on three core pillars:

1. **Asset Selection (CRI Filter)**: Identifies liquid assets with strong correlation between price movement and public attention
2. **Entry Signal (Panic Score)**: Combines volatility, volume, and attention metrics to time entries
3. **Macro Risk Filter (G-Score)**: Uses market-based proxies to gauge systemic risk and adjust position sizing

## 🏗️ Architecture

```
Harvester II/
├── config.json              # Trading parameters and settings
├── .env                     # API keys and secrets (hidden)
├── requirements.txt         # Python dependencies
├── main.py                  # Main entry point
├── pytest.ini              # Test configuration
├── src/                     # Core system modules
│   ├── __init__.py
│   ├── config.py            # Configuration management
│   ├── data_manager.py      # Data fetching and caching (async, TTLCache)
│   ├── signals.py           # Signal calculations (CRI, Panic Score, G-Score)
│   ├── risk_manager.py      # Risk controls and position sizing (trailing stops)
│   ├── portfolio.py         # Portfolio management and execution
│   ├── engine.py            # Main trading engine (Loguru logging, Prometheus metrics)
│   ├── backtest.py          # Enhanced backtesting with realistic execution
│   ├── models.py            # SQLAlchemy ORM models
│   ├── di.py                # Dependency injection container
│   ├── conftest.py          # Test fixtures
│   ├── test_*.py            # Comprehensive unit/integration tests
│   └── utils.py             # Utility functions (Polars support)
├── logs/                    # Loguru structured logs with rotation
└── data/                    # SQLite databases with optional encryption
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\Activate.ps1
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

1. **Edit `config.json`** with your trading parameters
2. **Create `.env`** file with your API keys:
   ```env
   ALPHA_VANTAGE_API_KEY=your_key_here
   GOOGLE_TRENDS_API_KEY=your_key_here
   BROKER_API_KEY=your_key_here
   ```

### 3. Run the System

```bash
# Live trading (scheduled daily execution)
python main.py --mode live

# Historical backtesting with realistic execution
python main.py --mode backtest --start-date 2020-01-01 --end-date 2024-01-01

# System status and health check
python main.py --mode status

# View Prometheus metrics
python main.py --mode metrics

# Run tests
cd src && pytest
```

## 📊 Key Features

### Signal Generation
- **CRI (Crowd-Reactivity Index)**: Correlates price changes with Google Trends data
- **Weighted Panic Score**: Configurable combination of volatility, volume, and attention z-scores
- **G-Score**: Macro risk assessment using VIX, SPY, and oil prices

### Bayesian Quickening v3
- **HMM State Machine**: Probabilistic market regime detection (calm/volatile/panic)
- **Enhanced Conviction**: Panic scores multiplied by Bayesian confidence (0.7+ threshold)
- **Dynamic Prior Optimization**: Optuna-tuned market state probabilities
- **Enhanced Covariance**: Full covariance matrices for better state modeling
- **A/B Testing Framework**: Automated performance comparison with statistical validation
- **Conviction Monitoring**: Real-time Prometheus metrics and structured logging
- **Robust Fallback**: Multi-level degradation (HMM → rules → default 0.5 conviction)
- **Performance Caching**: TTL-cached posterior probabilities (5min) for improved inference speed
- **Edge Hardening**: Buffer trimming (>100 samples) and KL-divergence low-evidence detection

#### Prior Optimization Example

The Bayesian State Machine optimizes market state priors using Optuna:

```python
# Default priors: [0.3, 0.4, 0.3] (calm/volatile/panic)
# Optimized priors: dynamically tuned for maximum Sharpe ratio

study = optuna.create_study(direction="maximize")
study.optimize(objective_function, n_trials=20)

# Best priors ensure probabilities sum to 1.0
best_priors = [0.25, 0.35, 0.40]  # Adapted to current market conditions
```

#### Decay Mathematics

Conviction decay follows exponential decay with evidence accumulation:

```
conviction(t) = conviction(t-1) × exp(-λ × time_since_last_update)

Where:
- λ (decay_rate) = 0.1 (configurable)
- time_since_last_update in trading days
- conviction resets to 0.5 when evidence threshold is met
```

#### KL-Divergence Edge Detection

Low-evidence scenarios trigger fallback using KL-divergence:

```python
# KL(P||Q) where P=observations, Q=model states
kl_divergence = min([KL_divergence(obs_dist, state_dist) for state_dist in model_states])

if kl_divergence < 0.1:  # Low evidence threshold
    return fallback_assessment()  # Rule-based logic
```

#### Predict Conviction Example

Real-time conviction assessment with stale data handling:

```python
from src.bayesian_state import get_bayesian_state_machine

# Initialize with data manager for caching
bsm = get_bayesian_state_machine(config, data_manager)

# Prepare market features
features = bsm.prepare_features(
    volatility_z=1.2,
    volume_z=-0.8,
    trends_z=2.1,
    g_score=1.8,
    price_change_5d=0.034
)

# Assess conviction with caching and edge detection
conviction = bsm.assess_conviction(features)

print(f"Market State: {conviction['state']}")  # 'panic', 'volatile', 'calm'
print(f"Conviction Level: {conviction['confidence']:.3f}")  # 0.0-1.0
print(f"Assessment Method: {conviction['method']}")  # 'bayesian', 'fallback', etc.
```

#### Performance Caching

Posterior probabilities are cached with 5-minute TTL:

```python
# Cache key: rounded feature vector string
features_key = "2.00,1.50,2.50,2.00,-0.02"
cached_result = cache.get(features_key)

if cached_result:
    return cached_result  # Skip expensive HMM inference
```

### Performance & Scalability
- **Async Data Fetching**: Concurrent API calls with aiohttp for improved speed
- **Polars Integration**: High-performance DataFrame operations (optional)
- **TTLCache**: Intelligent caching with automatic expiration
- **Dependency Injection**: Clean, testable architecture

### Risk Management
- **Fixed fractional position sizing** (0.5% base)
- **ATR-based stop losses** and profit targets
- **Dynamic trailing stops** with configurable ATR multipliers
- **Daily drawdown kill-switch** (2% limit)
- **Maximum position limits** (4 positions)
- **Macro risk adjustments** (50% size reduction during high risk)

### Data Management
- **Multi-source data fetching** (Yahoo Finance, Google Trends, Alpha Vantage)
- **Intelligent caching** with expiration times
- **SQLite persistence** for positions and trade history
- **Technical indicators** calculation (ATR, volatility, etc.)
- **Optional encryption** with SQLCipher

### Monitoring & Logging
- **Loguru Structured Logging**: JSON-compatible logs with rotation and filtering
- **Prometheus Metrics**: Real-time monitoring of equity, drawdown, positions, and G-Score
- **Sensitive Data Filtering**: Automatic redaction of API keys and secrets
- **Comprehensive Health Checks**: System status monitoring

## 🔧 Configuration

## 📊 Logging & Monitoring

### Structured Logging with Loguru

Harvester II uses **Loguru** for advanced structured logging with JSON support:

```bash
# Enable JSON logging for log aggregation systems
export HARVESTER_LOG_JSON=true

# Set log level
export HARVESTER_LOG_LEVEL=DEBUG

# Configure log file and rotation
export HARVESTER_LOG_FILE=logs/trading.log
export HARVESTER_LOG_MAX_SIZE="50 MB"
```

**Features:**
- ✅ **Structured JSON Output**: Perfect for log aggregation (ELK, Splunk, etc.)
- ✅ **Automatic Rotation**: Log files rotate by size with configurable retention
- ✅ **Colored Console Output**: Human-readable terminal output
- ✅ **Sensitive Data Filtering**: Automatic redaction of API keys and secrets
- ✅ **Performance Logging**: Structured metrics for trading signals and performance

### Comprehensive Prometheus Monitoring

Complete system monitoring with **30+ metrics** exposed via Prometheus:

```bash
# Access metrics at http://localhost:8000
curl http://localhost:8000/metrics
```

**Core Metrics:**
- **Portfolio**: Equity, positions, daily P&L
- **Risk**: Drawdown, G-Score, conviction levels
- **Performance**: Sharpe ratio, win rate, max drawdown
- **Bias Analysis**: Look-ahead, survivorship, overfitting detection
- **System Health**: Data source availability, processing times

**Trading Signals:**
```prometheus
harvester_signal_conviction{conviction="0.85"}  # Current signal conviction
harvester_signals_total{signal_type="entry",outcome="generated"}  # Signal counts
```

**Bias Detection:**
```prometheus
harvester_bias_look_ahead 0  # 0=no bias, 1=bias detected
harvester_bias_survivorship 0
harvester_bias_overfitting 0
```

## 📚 API Documentation

### Core Trading Signals API

#### Crowd-Reactivity Index (CRI) Calculation

The CRI measures correlation between price movements and search interest:

```python
from src.signals import SignalCalculator
from src.data_processing import create_dataframe, create_series

# Initialize signal calculator
config = load_config()
data_manager = create_data_manager(config)
signal_calc = SignalCalculator(config, data_manager)

# Get price and trends data
price_data = create_dataframe({
    'Close': [100.0, 102.0, 101.0, 103.0, 105.0]
}, index=['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'])

trends_data = create_dataframe({
    'value': [75, 78, 80, 82, 85]
}, index=['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'])

# Calculate CRI
cri_score = signal_calc.calculate_cri('AAPL', price_data, trends_data)
print(f"CRI Score: {cri_score:.3f}")  # Correlation coefficient (-1.0 to 1.0)
```

**CRI Interpretation:**
- **> 0.7**: Strong positive correlation (crowd anticipation)
- **0.3 - 0.7**: Moderate correlation
- **< 0.3**: Weak or no correlation
- **< 0**: Inverse correlation (contrarian signal)

#### Panic Score Calculation

Combines volatility, volume, and trends z-scores:

```python
# Calculate panic score
panic_score = signal_calc.calculate_panic_score('AAPL', price_data, trends_data)
print(f"Panic Score: {panic_score:.2f}")

# Weighted components (configurable in config.json)
panic_components = {
    'volatility_weight': 1.0,    # Price volatility z-score
    'volume_weight': 1.0,        # Volume z-score
    'trends_weight': 1.0         # Google Trends z-score
}
```

**Panic Score Components:**
- **Volatility**: Rolling standard deviation of returns
- **Volume**: Trading volume relative to average
- **Trends**: Google search interest changes

#### Entry Signal Generation

```python
# Get trading signals for universe
tradable_assets = ['AAPL', 'MSFT', 'GOOGL']
signals = signal_calc.get_entry_signals(tradable_assets)

for signal in signals:
    print(f"Symbol: {signal['symbol']}")
    print(f"Conviction: {signal['confidence']:.2f}")
    print(f"Market State: {signal.get('market_state', 'unknown')}")
    print(f"Assessment Method: {signal.get('assessment_method', 'rule-based')}")
```

### Bayesian State Machine API

#### HMM-Based Market Regime Detection

```python
from src.bayesian_state import get_bayesian_state_machine

# Initialize BSM
config_data = load_config()
bsm = get_bayesian_state_machine(config_data)

# Prepare features for prediction
features = np.array([[2.0, 1.5, 2.5, 2, -0.02]])  # [cri, volatility, volume, panic, momentum]

# Assess market conviction
assessment = bsm.assess_conviction(features)
print(f"Conviction: {assessment['confidence']:.2f}")
print(f"Market State: {assessment['state']}")
print(f"Should Trade: {assessment['should_trade']}")
```

**Market States:**
- **Calm**: Low volatility, normal conditions
- **Volatile**: High volatility, uncertain conditions
- **Panic**: Extreme volatility, crisis conditions

### Backtesting API

#### Comprehensive Backtesting

```python
from src.backtest import BacktestEngine

# Initialize backtest engine
backtest_engine = BacktestEngine(config, data_manager, signal_calc, risk_manager)

# Run backtest with vectorized option
results = backtest_engine.run_backtest(
    start_date='2020-01-01',
    end_date='2024-01-01',
    initial_capital=100000,
    use_vectorbt=False  # or True for VectorBT integration
)

print(f"Total Return: {results['capital']['total_return']:.2%}")
print(f"Sharpe Ratio: {results['capital']['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['capital']['max_drawdown']:.2%}")
```

#### A/B Testing Framework

```python
# Compare Bayesian enabled vs disabled
ab_results = backtest_engine.run_ab_test('2020-01-01', '2021-01-01')

print("=== A/B Test Results ===")
print(f"Bayesian Sharpe: {ab_results['bayesian_enabled']['capital']['sharpe_ratio']:.2f}")
print(f"Rule-based Sharpe: {ab_results['bayesian_disabled']['capital']['sharpe_ratio']:.2f}")
```

### Optimization API

#### Hyperparameter Tuning

```python
from src.optimization import get_optimizer

# Initialize optimizer
optimizer = get_optimizer(config, data_manager, signal_calc, risk_manager, backtest_engine)

# Run optimization
opt_results = optimizer.optimize_parameters('2020-01-01', '2021-01-01')

print(f"Best Sharpe: {opt_results['best_sharpe_ratio']:.3f}")
print(f"Optimized Parameters: {opt_results['best_parameters']}")
```

### Configuration Schema

#### Core Configuration Structure

```json
{
  "universe": {
    "assets": ["SPY", "QQQ", "BTC-USD"],
    "cri_threshold": 0.4
  },
  "signals": {
    "panic_threshold": 3.0,
    "panic_score_weights": {
      "volatility_weight": 1.0,
      "volume_weight": 1.0,
      "trends_weight": 1.0
    }
  },
  "risk_management": {
    "max_position_size": 0.05,
    "stop_loss_pct": 0.05,
    "take_profit_pct": 0.10,
    "max_open_positions": 5
  },
  "backtesting": {
    "start_date": "2020-01-01",
    "end_date": "2024-01-01",
    "initial_capital": 100000
  },
  "bayesian": {
    "enabled": true,
    "n_states": 3,
    "conviction_threshold": 0.7
  },
  "logging": {
    "level": "INFO",
    "json_format": false,
    "file_path": "logs/harvester_ii.log"
  },
  "monitoring": {
    "prometheus_port": 8000
  }
}
```

### Data Processing Backend

Harvester II supports multiple data processing backends for optimal performance:

- **Pandas** (default): Mature, feature-complete, widely compatible
- **Polars**: High-performance, memory-efficient, Rust-based engine

```bash
# Switch to Polars for better performance
export HARVESTER_DATA_BACKEND=polars

# Run benchmarks to compare performance
python src/benchmark_data_processing.py
```

**Performance Comparison (10k data points):**
| Operation | Pandas | Polars | Speedup |
|-----------|--------|--------|---------|
| DataFrame Creation | ~5ms | ~2ms | 2.5x |
| Series Operations | ~10ms | ~3ms | 3.3x |
| Correlation | ~8ms | ~2ms | 4x |
| Rolling Mean | ~15ms | ~4ms | 3.8x |

### Core Parameters

```json
{
  "universe": {
    "assets": ["SPY", "QQQ", "BTC-USD", "ETH-USD"],
    "cri_threshold": 0.4
  },
  "signals": {
    "panic_threshold": 3.0,
    "panic_score_weights": {
      "volatility_weight": 1.0,
      "volume_weight": 1.0,
      "trends_weight": 0.8
    }
  },
  "risk_management": {
    "equity": 100000,
    "base_position_fraction": 0.005,
    "max_open_positions": 4,
    "daily_drawdown_limit": 0.02,
    "trailing_stops": {
      "enabled": true,
      "atr_multiplier": 1.5,
      "min_distance": 0.02
    }
  },
  "performance": {
    "use_polars": true,
    "async_api_calls": true
  },
  "monitoring": {
    "prometheus_enabled": true,
    "prometheus_port": 8000
  }
}
```

### Bayesian v3 Configuration

```json
"bayesian": {
  "enabled": true,
  "n_states": 3,
  "conviction_threshold": 0.7,
  "priors": [0.3, 0.4, 0.3],
  "covariance_type": "full",
  "training_samples": 1000,
  "inference_timeout": 2.0
}
```

### Bayesian v3 Features

#### Dynamic Prior Optimization
```bash
# Run prior optimization (requires optuna)
from src.bayesian_state import get_bayesian_state_machine
bsm = get_bayesian_state_machine(config_data)
result = bsm.optimize_priors(n_trials=20)
print(f"Optimized priors: {result['optimized_priors']}")
```

#### A/B Testing Framework
```bash
# Compare Bayesian enabled vs disabled
python main.py --mode ab-test --start-date 2020-01-01 --end-date 2021-01-01

# Expected output shows performance improvements:
# Sharpe Ratio: +0.28, Total Return: +3.3%, Max Drawdown: +3.3%
```

#### Enhanced Monitoring
```bash
# View conviction logs
tail -f logs/harvester_ii.log | grep conviction

# Prometheus metrics include:
# harvester_signal_conviction - Real-time conviction levels
# Existing metrics: equity, drawdown, positions, G-score
```

### Bayesian Usage Example

```bash
# Run A/B test to compare Bayesian enhancement
python main.py --mode ab-test --start-date 2020-01-01 --end-date 2021-01-01

# Disable Bayesian State Machine
# Set "bayesian.enabled": false in config.json

# View Bayesian conviction in logs
tail -f logs/harvester_ii.log | grep -i conviction
```

**Expected Output:**
```
=== A/B Test Results ===
Test Period: 2020-01-01 to 2021-01-01
Initial Capital: $100,000.00

--- Bayesian State Machine ENABLED ---
  Total Return: 15.4%
  Sharpe Ratio: 1.23
  Max Drawdown: -12.3%
  Win Rate: 58.2%

--- Bayesian State Machine DISABLED ---
  Total Return: 12.1%
  Sharpe Ratio: 0.95
  Max Drawdown: -15.6%
  Win Rate: 52.4%

--- IMPROVEMENT (Enabled - Disabled) ---
  Sharpe Ratio: +0.28
  Total Return: +3.3%
  Max Drawdown: +3.3%
  Win Rate: +5.8%
  Conviction Correlation: 0.72
```

### Trading Logic

1. **Asset Filtering**: Only trade assets with CRI ≥ 0.4
2. **Bayesian Enhancement**: Panic Score enhanced by HMM conviction multiplier
3. **Entry Signals**: Enhanced Panic Score > 3.0 triggers entry (with 0.7+ conviction)
4. **Contrarian Logic**: Buy on sharp drops, sell on sharp rises
5. **Risk Adjustment**: Reduce position size when G-Score ≥ 2

## 📈 Performance Monitoring

The system provides comprehensive monitoring:

- **Real-time portfolio tracking**
- **Risk metrics** (drawdown, position limits)
- **Signal quality** (CRI, Panic Score, G-Score)
- **Trade history** with P&L analysis
- **System health** checks

## 🛡️ Risk Controls

### Position Sizing
- Base position: 0.5% of equity
- Maximum position: $5,000
- Risk per trade: 0.5% of equity

### Stop Losses & Targets
- Stop loss: 1x ATR below entry
- Profit target: 2x ATR above entry
- Daily drawdown limit: 2% of equity

### Macro Risk
- G-Score ≥ 2: Reduce position size by 50%
- VIX > 25: Increase risk score
- SPY 7-day return < -5%: Increase risk score
- Oil 7-day return > 10%: Increase risk score

## 🔍 Monitoring & Logging

### Log Files
- `logs/harvester_ii.log`: Main system log
- Database files in `data/` directory

### Status Command
```bash
python main.py --mode status
```

Shows:
- System status and health
- Portfolio summary
- Risk metrics
- Tradable universe
- Macro risk assessment

### Prometheus Metrics

Access real-time metrics for monitoring and alerting:

```bash
# View current metrics
python main.py --mode metrics

# Prometheus endpoint (when system is running)
curl http://localhost:8000
```

Available metrics:
- `harvester_equity_total`: Portfolio equity
- `harvester_drawdown_percentage`: Current drawdown
- `harvester_positions_open`: Number of open positions
- `harvester_daily_pnl`: Daily profit/loss
- `harvester_g_score`: Macro risk G-Score

### Testing

Run comprehensive test suite:

```bash
# Run all tests
cd src && pytest

# Run with coverage
cd src && pytest --cov=. --cov-report=html

# Run specific test categories
cd src && pytest -m unit        # Unit tests only
cd src && pytest -m integration # Integration tests only
```

## 🧪 Enhanced Backtesting

Run historical backtests with realistic trading execution:

```bash
# Standard backtest
python main.py --mode backtest --start-date 2020-01-01 --end-date 2024-01-01

# Walk-forward validation to detect overfitting
python main.py --mode walk-forward --start-date 2020-01-01 --end-date 2024-01-01

# Survivor-free backtest (only assets that existed throughout period)
python main.py --mode survivor-free --start-date 2020-01-01 --end-date 2024-01-01

# Automated bias detection and analysis
python main.py --mode bias-check

# A/B test comparing Bayesian enhancement
python main.py --mode ab-test --start-date 2020-01-01 --end-date 2021-01-01
```

### Backtesting Features
- **Realistic Execution**: Slippage and commissions applied to all trades
- **Holiday Handling**: Automatic exclusion of weekends and US federal holidays
- **Historical Trends**: Optional CSV loading of historical Google Trends data
- **Trailing Stops**: Dynamic stop loss adjustment during backtests
- **Performance Metrics**: Comprehensive risk-adjusted return calculations

### Bias Mitigation Tools
- **Walk-forward Validation**: Multi-fold out-of-sample testing to detect overfitting
- **Survivor-free Backtesting**: Only assets with complete historical data throughout period
- **Automated Bias Detection**: Analysis for look-ahead, survivorship, and overfitting biases
- **A/B Testing Framework**: Compare Bayesian enhancement impact with statistical validation
- **Out-of-sample Validation**: Ensure robust parameter selection and model generalization

## 📋 Requirements

### Python Packages
- `yfinance`: Price data
- `pytrends`: Google Trends data
- `pandas`, `numpy`: Data processing
- `polars`: High-performance DataFrames (optional)
- `scipy`: Statistical calculations
- `aiohttp`: Async HTTP requests
- `cachetools`: TTL caching
- `loguru`: Structured logging
- `prometheus-client`: Metrics monitoring
- `python-dotenv`: Environment variables
- `schedule`: Task scheduling
- `sqlalchemy`: Database ORM

### API Keys (Optional)
- **Alpha Vantage**: Backup price data (`ALPHA_VANTAGE_API_KEY`)
- **Google Trends**: Attention metrics (no API key required)
- **Broker API**: Live trading execution (varies by broker)

## 🔐 Secrets and Environment Variables

### Required Setup

1. **Create `.env` file** in project root:
   ```bash
   touch .env
   ```

2. **Add your API keys** (replace with actual values):
   ```env
   # Alpha Vantage (backup price data)
   ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here

   # Database encryption (optional)
   DATABASE_ENCRYPTION_KEY=your_strong_encryption_key_here

   # Prometheus monitoring (optional)
   PROMETHEUS_PORT=8000

   # Logging configuration
   LOG_LEVEL=INFO
   LOG_FILE_PATH=logs/harvester_ii.log
   ```

### Security Best Practices

- **Never commit `.env`** files to version control
- **Use strong, unique keys** for encryption
- **Rotate API keys regularly** for security
- **Limit API permissions** to read-only when possible
- **Monitor API usage** to avoid rate limits

### Getting API Keys

- **Alpha Vantage**: Free tier available at [alphavantage.co](https://www.alphavantage.co/support/#api-key)
- **Google Trends**: No API key required (built-in rate limits)
- **Broker APIs**: Check your broker's developer documentation

### Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ALPHA_VANTAGE_API_KEY` | No | None | Backup price data provider |
| `DATABASE_ENCRYPTION_KEY` | No | None | SQLCipher encryption key |
| `PROMETHEUS_PORT` | No | 8000 | Metrics server port |
| `LOG_LEVEL` | No | INFO | Logging verbosity |
| `LOG_FILE_PATH` | No | logs/harvester_ii.log | Log file location |

## ⚠️ Important Notes

### Data Sources
- **Primary**: Yahoo Finance (free)
- **Backup**: Alpha Vantage (API key required)
- **Trends**: Google Trends (free with rate limits)

### Trading Hours
- **Stocks/ETFs**: NYSE/NASDAQ hours (9:30 AM - 4:00 PM ET)
- **Crypto**: 24/7 trading
- **System runs**: Daily at 4:00 PM ET

### Risk Disclaimer
This is a sophisticated trading system for educational and research purposes. Always:
- Test thoroughly with paper trading
- Start with small position sizes
- Monitor system performance closely
- Understand all risks before live trading

## 🔧 Troubleshooting

### Common Issues

1. **Import/SQLAlchemy errors**
   - Run `pip install -r requirements.txt`
   - Ensure Python 3.8+ is installed

2. **No data from Yahoo Finance**
   - Check internet connection
   - Verify symbol names
   - Try different time periods

3. **Google Trends rate limits**
   - System will continue without trends data
   - Consider paid Google Trends API

4. **Prometheus metrics not available**
   - Check if prometheus-client is installed
   - Verify port 8000 is not in use

5. **Polars performance issues**
   - Set `"use_polars": false` in config.json
   - Falls back to pandas automatically

6. **Database errors**
   - Check `data/` directory permissions
   - Delete corrupted database files
   - Schema will auto-create on next run

7. **Configuration errors**
   - Validate `config.json` syntax with `python -m json.tool config.json`
   - Check required parameters

### Log Analysis
Check `logs/harvester_ii.log` for detailed error messages and system status.

## 📚 Further Development

### Potential Enhancements
- **Machine learning** signal optimization
- **Additional data sources** (news sentiment, options flow)
- **Advanced risk models** (VaR, portfolio optimization)
- **Real-time execution** with broker APIs
- **Web dashboard** for monitoring

### Customization
The modular design allows easy customization:
- Add new signal types in `signals.py`
- Implement custom risk models in `risk_manager.py`
- Add new data sources in `data_manager.py`

---

**Harvester II Trading System** - Enterprise-grade volatility and attention-driven trading with comprehensive monitoring, testing, and performance optimizations.
