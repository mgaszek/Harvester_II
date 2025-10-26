# Harvester II Trading System

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

### Trading Logic

1. **Asset Filtering**: Only trade assets with CRI ≥ 0.4
2. **Entry Signals**: Panic Score > 3.0 triggers entry
3. **Contrarian Logic**: Buy on sharp drops, sell on sharp rises
4. **Risk Adjustment**: Reduce position size when G-Score ≥ 2

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
python main.py --mode backtest --start-date 2020-01-01 --end-date 2024-01-01
```

### Backtesting Features
- **Realistic Execution**: Slippage and commissions applied to all trades
- **Holiday Handling**: Automatic exclusion of weekends and US federal holidays
- **Historical Trends**: Optional CSV loading of historical Google Trends data
- **Trailing Stops**: Dynamic stop loss adjustment during backtests
- **Performance Metrics**: Comprehensive risk-adjusted return calculations

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
- **Alpha Vantage**: Backup price data
- **Google Trends**: Attention metrics
- **Broker API**: Live trading execution

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
