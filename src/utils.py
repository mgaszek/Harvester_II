"""
Utility functions for Harvester II trading system.
Helper functions for data processing, calculations, and system utilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
import json
import os
from pathlib import Path


def calculate_returns(prices: pd.Series, periods: int = 1) -> pd.Series:
    """
    Calculate returns for a price series.
    
    Args:
        prices: Price series
        periods: Number of periods for return calculation
        
    Returns:
        Series with returns
    """
    return prices.pct_change(periods)


def calculate_volatility(returns: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculate rolling volatility.
    
    Args:
        returns: Returns series
        window: Rolling window size
        
    Returns:
        Series with volatility
    """
    return returns.rolling(window=window).std()


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, 
                 window: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR).
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        window: Rolling window size
        
    Returns:
        Series with ATR values
    """
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    return true_range.rolling(window=window).mean()


def calculate_z_score(value: float, series: pd.Series) -> float:
    """
    Calculate z-score for a value against a series.
    
    Args:
        value: Current value
        series: Historical series
        
    Returns:
        Z-score
    """
    if len(series) < 2:
        return 0.0
    
    mean_val = series.mean()
    std_val = series.std()
    
    if std_val == 0:
        return 0.0
    
    return (value - mean_val) / std_val


def calculate_correlation(series1: pd.Series, series2: pd.Series) -> float:
    """
    Calculate correlation between two series.
    
    Args:
        series1: First series
        series2: Second series
        
    Returns:
        Correlation coefficient
    """
    try:
        # Align series by index
        aligned = pd.concat([series1, series2], axis=1).dropna()
        
        if len(aligned) < 2:
            return 0.0
        
        return aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
    except Exception:
        return 0.0


def align_data_by_date(df1: pd.DataFrame, df2: pd.DataFrame, 
                      date_col: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align two DataFrames by date.
    
    Args:
        df1: First DataFrame
        df2: Second DataFrame
        date_col: Date column name (if using column instead of index)
        
    Returns:
        Tuple of aligned DataFrames
    """
    try:
        if date_col:
            df1_dates = pd.to_datetime(df1[date_col]).dt.date
            df2_dates = pd.to_datetime(df2[date_col]).dt.date
        else:
            df1_dates = pd.to_datetime(df1.index).date
            df2_dates = pd.to_datetime(df2.index).date
        
        # Find common dates
        common_dates = set(df1_dates) & set(df2_dates)
        
        if not common_dates:
            return pd.DataFrame(), pd.DataFrame()
        
        # Filter DataFrames to common dates
        if date_col:
            df1_aligned = df1[df1_dates.isin(common_dates)]
            df2_aligned = df2[df2_dates.isin(common_dates)]
        else:
            df1_aligned = df1[df1_dates.isin(common_dates)]
            df2_aligned = df2[df2_dates.isin(common_dates)]
        
        return df1_aligned, df2_aligned
        
    except Exception as e:
        logging.error(f"Failed to align data by date: {e}")
        return pd.DataFrame(), pd.DataFrame()


def calculate_performance_metrics(equity_curve: pd.Series) -> Dict[str, float]:
    """
    Calculate performance metrics from equity curve.
    
    Args:
        equity_curve: Series with equity values over time
        
    Returns:
        Dictionary with performance metrics
    """
    try:
        if equity_curve.empty:
            return {}
        
        # Basic metrics
        total_return = (equity_curve.iloc[-1] - equity_curve.iloc[0]) / equity_curve.iloc[0]
        
        # Calculate returns
        returns = equity_curve.pct_change().dropna()
        
        # Volatility
        volatility = returns.std() * np.sqrt(252)  # Annualized
        
        # Sharpe ratio (assuming risk-free rate = 0)
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # Maximum drawdown
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        max_drawdown = drawdown.min()
        
        # Win rate
        positive_returns = returns[returns > 0]
        win_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0
        
        return {
            'total_return': total_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len(returns)
        }
        
    except Exception as e:
        logging.error(f"Failed to calculate performance metrics: {e}")
        return {}


def format_currency(amount: float, currency: str = "USD") -> str:
    """
    Format currency amount for display.
    
    Args:
        amount: Amount to format
        currency: Currency code
        
    Returns:
        Formatted currency string
    """
    if currency == "USD":
        return f"${amount:,.2f}"
    else:
        return f"{amount:,.2f} {currency}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format percentage for display.
    
    Args:
        value: Value to format as percentage
        decimals: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{decimals}f}%"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division by zero
        
    Returns:
        Division result or default
    """
    try:
        if denominator == 0 or pd.isna(denominator):
            return default
        return numerator / denominator
    except Exception:
        return default


def validate_data_quality(data: pd.DataFrame, required_columns: List[str] = None) -> bool:
    """
    Validate data quality for trading.
    
    Args:
        data: DataFrame to validate
        required_columns: List of required columns
        
    Returns:
        True if data quality is acceptable
    """
    try:
        if data.empty:
            return False
        
        # Check for required columns
        if required_columns:
            missing_columns = set(required_columns) - set(data.columns)
            if missing_columns:
                logging.warning(f"Missing required columns: {missing_columns}")
                return False
        
        # Check for excessive missing values
        missing_pct = data.isnull().sum() / len(data)
        high_missing = missing_pct[missing_pct > 0.1]  # More than 10% missing
        
        if not high_missing.empty:
            logging.warning(f"High missing values in columns: {high_missing.to_dict()}")
        
        # Check for negative prices (if Close column exists)
        if 'Close' in data.columns:
            negative_prices = (data['Close'] <= 0).sum()
            if negative_prices > 0:
                logging.warning(f"Found {negative_prices} negative or zero prices")
                return False
        
        return True
        
    except Exception as e:
        logging.error(f"Data validation failed: {e}")
        return False


def create_backup(file_path: str) -> str:
    """
    Create a backup of a file.
    
    Args:
        file_path: Path to file to backup
        
    Returns:
        Path to backup file
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return ""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = path.parent / f"{path.stem}_backup_{timestamp}{path.suffix}"
        
        # Copy file
        import shutil
        shutil.copy2(file_path, backup_path)
        
        return str(backup_path)
        
    except Exception as e:
        logging.error(f"Failed to create backup: {e}")
        return ""


def load_json_safe(file_path: str, default: Dict = None) -> Dict:
    """
    Safely load JSON file with error handling.
    
    Args:
        file_path: Path to JSON file
        default: Default value if loading fails
        
    Returns:
        Loaded data or default
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load JSON from {file_path}: {e}")
        return default or {}


def save_json_safe(data: Dict, file_path: str) -> bool:
    """
    Safely save data to JSON file with error handling.
    
    Args:
        data: Data to save
        file_path: Path to save file
        
    Returns:
        True if successful
    """
    try:
        # Create directory if it doesn't exist
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        return True
        
    except Exception as e:
        logging.error(f"Failed to save JSON to {file_path}: {e}")
        return False


def get_market_hours() -> Dict[str, str]:
    """
    Get market hours for different exchanges.
    
    Returns:
        Dictionary with market hours
    """
    return {
        'NYSE': '09:30-16:00',
        'NASDAQ': '09:30-16:00',
        'Crypto': '24/7',
        'Forex': '24/7'
    }


def is_market_open(exchange: str = 'NYSE') -> bool:
    """
    Check if market is currently open.
    
    Args:
        exchange: Exchange to check
        
    Returns:
        True if market is open
    """
    try:
        now = datetime.now()
        
        if exchange in ['Crypto', 'Forex']:
            return True  # Always open
        
        if exchange in ['NYSE', 'NASDAQ']:
            # Check if it's a weekday
            if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
                return False
            
            # Check time (simplified - doesn't account for holidays)
            market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
            
            return market_open <= now <= market_close
        
        return False
        
    except Exception as e:
        logging.error(f"Failed to check market hours: {e}")
        return False


def calculate_position_size_fixed_fractional(equity: float, risk_percent: float, 
                                           entry_price: float, stop_price: float) -> int:
    """
    Calculate position size using fixed fractional method.
    
    Args:
        equity: Available equity
        risk_percent: Risk percentage (e.g., 0.01 for 1%)
        entry_price: Entry price
        stop_price: Stop loss price
        
    Returns:
        Number of shares/units
    """
    try:
        risk_amount = equity * risk_percent
        risk_per_share = abs(entry_price - stop_price)
        
        if risk_per_share <= 0:
            return 0
        
        shares = int(risk_amount / risk_per_share)
        return max(0, shares)
        
    except Exception as e:
        logging.error(f"Failed to calculate position size: {e}")
        return 0


def calculate_position_size_percentage(equity: float, position_percent: float, 
                                     price: float) -> int:
    """
    Calculate position size using percentage of equity method.
    
    Args:
        equity: Available equity
        position_percent: Position percentage (e.g., 0.01 for 1%)
        price: Asset price
        
    Returns:
        Number of shares/units
    """
    try:
        position_value = equity * position_percent
        shares = int(position_value / price)
        return max(0, shares)
        
    except Exception as e:
        logging.error(f"Failed to calculate position size: {e}")
        return 0