"""
Data Processing Abstraction Layer for Harvester II.
Provides unified interface for pandas and polars operations.
"""

import os
from typing import Any, Union

import numpy as np

# Backend selection - can be controlled via environment variable
DATA_BACKEND = os.getenv("HARVESTER_DATA_BACKEND", "pandas").lower()

# Import backends conditionally
if DATA_BACKEND == "polars":
    try:
        import polars as pl

        POLARS_AVAILABLE = True
    except ImportError:
        POLARS_AVAILABLE = False
        import pandas as pd

        pl = None
        DATA_BACKEND = "pandas"
else:
    import pandas as pd

    POLARS_AVAILABLE = False
    pl = None
    DATA_BACKEND = "pandas"

# Type aliases
DataFrame = (
    Union["pd.DataFrame", "pl.DataFrame"] if POLARS_AVAILABLE else "pd.DataFrame"
)
Series = Union["pd.Series", "pl.Series"] if POLARS_AVAILABLE else "pd.Series"


class DataProcessor:
    """
    Unified data processing interface that works with both pandas and polars.

    Provides a consistent API regardless of the underlying dataframe library,
    allowing seamless switching between pandas and polars for performance optimization.
    """

    @staticmethod
    def create_dataframe(
        data: dict[str, list[Any]], index: list[Any] | None = None
    ) -> DataFrame:
        """Create a DataFrame from dictionary data."""
        if DATA_BACKEND == "polars":
            df = pl.DataFrame(data)
            if index is not None:
                df = df.with_columns(pl.Series("index", index).alias("__index"))
                df = df.select(
                    ["__index"] + [col for col in df.columns if col != "__index"]
                )
            return df
        return pd.DataFrame(data, index=index)

    @staticmethod
    def create_series(
        data: list[Any], index: list[Any] | None = None, name: str | None = None
    ) -> Series:
        """Create a Series from data."""
        if DATA_BACKEND == "polars":
            return pl.Series(name or "series", data, index=index)
        return pd.Series(data, index=index, name=name)

    @staticmethod
    def date_range(start: str, end: str, freq: str = "D") -> Series:
        """Create a date range."""
        if DATA_BACKEND == "polars":
            # Polars date_range is different
            start_dt = pl.datetime(int(start[:4]), int(start[5:7]), int(start[8:10]))
            end_dt = pl.datetime(int(end[:4]), int(end[5:7]), int(end[8:10]))
            # For now, return a simple range - this would need more complex implementation
            return pl.Series("date_range", [])
        return pd.date_range(start=start, end=end, freq=freq)

    @staticmethod
    def concat(dfs: list[DataFrame], axis: int = 0) -> DataFrame:
        """Concatenate DataFrames."""
        if DATA_BACKEND == "polars":
            return pl.concat(dfs, how="vertical" if axis == 0 else "horizontal")
        return pd.concat(dfs, axis=axis)

    @staticmethod
    def is_empty(df: DataFrame) -> bool:
        """Check if DataFrame is empty."""
        if DATA_BACKEND == "polars":
            return len(df) == 0
        return df.empty

    @staticmethod
    def dropna(df: DataFrame, axis: int = 0, how: str = "any") -> DataFrame:
        """Drop NA values."""
        if DATA_BACKEND == "polars":
            return df.drop_nulls()
        return df.dropna(axis=axis, how=how)

    @staticmethod
    def pct_change(series, periods: int = 1):
        """Calculate percentage change."""
        if DATA_BACKEND == "polars":
            # Polars pct_change implementation
            return series.pct_change(periods=periods)
        return series.pct_change(periods=periods)

    @staticmethod
    def rolling_mean(series: Series, window: int) -> Series:
        """Calculate rolling mean."""
        if DATA_BACKEND == "polars":
            return series.rolling_mean(window_size=window)
        return series.rolling(window=window).mean()

    @staticmethod
    def rolling_std(series, window: int):
        """Calculate rolling standard deviation."""
        if DATA_BACKEND == "polars":
            # Polars rolling std implementation
            return series.rolling_std(window_size=window)
        return series.rolling(window=window).std()

    @staticmethod
    def ewma(series: Series, span: int) -> Series:
        """Calculate exponentially weighted moving average."""
        if DATA_BACKEND == "polars":
            # Polars has limited EWMA support
            return series.ewm_mean(span=span)
        return series.ewm(span=span).mean()

    @staticmethod
    def correlation(series1: Series, series2: Series) -> float:
        """Calculate correlation between two series."""
        if DATA_BACKEND == "polars":
            return series1.corr(series2)
        return series1.corr(series2)

    @staticmethod
    def mean(series: Series) -> float:
        """Calculate mean of series."""
        if DATA_BACKEND == "polars":
            return series.mean()
        return series.mean()

    @staticmethod
    def std(series: Series) -> float:
        """Calculate standard deviation of series."""
        if DATA_BACKEND == "polars":
            return series.std()
        return series.std()

    @staticmethod
    def quantile(series: Series, q: float) -> float:
        """Calculate quantile of series."""
        if DATA_BACKEND == "polars":
            return series.quantile(q)
        return series.quantile(q)

    @staticmethod
    def max(series: Series) -> float:
        """Calculate maximum of series."""
        if DATA_BACKEND == "polars":
            return series.max()
        return series.max()

    @staticmethod
    def min(series: Series) -> float:
        """Calculate minimum of series."""
        if DATA_BACKEND == "polars":
            return series.min()
        return series.min()

    @staticmethod
    def to_numpy(series: Series) -> np.ndarray:
        """Convert series to numpy array."""
        if DATA_BACKEND == "polars":
            return series.to_numpy()
        return series.to_numpy()

    @staticmethod
    def to_dict(df: DataFrame) -> dict[str, Any]:
        """Convert DataFrame to dictionary."""
        if DATA_BACKEND == "polars":
            return df.to_dict(as_series=False)
        return df.to_dict()

    @staticmethod
    def set_index(df: DataFrame, column: str) -> DataFrame:
        """Set index of DataFrame."""
        if DATA_BACKEND == "polars":
            # Polars handles indexing differently
            return df
        return df.set_index(column)

    @staticmethod
    def reset_index(df: DataFrame) -> DataFrame:
        """Reset index of DataFrame."""
        if DATA_BACKEND == "polars":
            return df
        return df.reset_index()

    @staticmethod
    def groupby(df: DataFrame, by: str) -> Any:
        """Group DataFrame by column."""
        if DATA_BACKEND == "polars":
            return df.group_by(by)
        return df.groupby(by)

    @staticmethod
    def merge(df1: DataFrame, df2: DataFrame, on: str, how: str = "left") -> DataFrame:
        """Merge DataFrames."""
        if DATA_BACKEND == "polars":
            return df1.join(df2, on=on, how=how)
        return pd.merge(df1, df2, on=on, how=how)

    @staticmethod
    def fillna(series: Series, value: Any) -> Series:
        """Fill NA values."""
        if DATA_BACKEND == "polars":
            return series.fill_null(value)
        return series.fillna(value)

    @staticmethod
    def isna(series: Series) -> Series:
        """Check for NA values."""
        if DATA_BACKEND == "polars":
            return series.is_null()
        return series.isna()

    @staticmethod
    def get_backend() -> str:
        """Get current data processing backend."""
        return DATA_BACKEND

    @staticmethod
    def is_polars_available() -> bool:
        """Check if polars is available."""
        return POLARS_AVAILABLE


# Convenience functions for common operations
def create_dataframe(
    data: dict[str, list[Any]], index: list[Any] | None = None
) -> DataFrame:
    """Create a DataFrame from dictionary data."""
    return DataProcessor.create_dataframe(data, index)


def create_series(
    data: list[Any], index: list[Any] | None = None, name: str | None = None
) -> Series:
    """Create a Series from data."""
    return DataProcessor.create_series(data, index, name)


def calculate_z_score(value: float, series: Series) -> float:
    """Calculate z-score for a value relative to a series."""
    if DataProcessor.is_empty(series) or len(series) < 2:
        return 0.0

    mean_val = DataProcessor.mean(series)
    std_val = DataProcessor.std(series)

    if std_val == 0:
        return 0.0

    return (value - mean_val) / std_val


def calculate_correlation(series1: Series, series2: Series) -> float:
    """Calculate correlation between two series."""
    if DataProcessor.is_empty(series1) or DataProcessor.is_empty(series2):
        return 0.0

    return DataProcessor.correlation(series1, series2)


def calculate_returns(price_series: Series, method: str = "simple") -> Series:
    """Calculate returns from price series."""
    if method == "log":
        # Log returns: ln(P_t / P_{t-1})
        if DATA_BACKEND == "polars":
            return DataProcessor.pct_change(price_series).log()
        return np.log(price_series / price_series.shift(1))
    # Simple returns: (P_t - P_{t-1}) / P_{t-1}
    return DataProcessor.pct_change(price_series)


def calculate_volatility(returns: Series, annualize: bool = True) -> float:
    """Calculate volatility from returns."""
    if DataProcessor.is_empty(returns):
        return 0.0

    vol = DataProcessor.std(returns)

    if annualize:
        # Assuming daily returns, annualize by sqrt(252)
        vol = vol * np.sqrt(252)

    return vol


def calculate_sharpe_ratio(
    returns: Series, risk_free_rate: float = 0.0, annualize: bool = True
) -> float:
    """Calculate Sharpe ratio."""
    if DataProcessor.is_empty(returns):
        return 0.0

    mean_return = DataProcessor.mean(returns)
    volatility = DataProcessor.std(returns)

    if annualize:
        mean_return = mean_return * 252  # Annualize returns
        volatility = volatility * np.sqrt(252)  # Annualize volatility

    if volatility == 0:
        return 0.0

    return (mean_return - risk_free_rate) / volatility


def calculate_max_drawdown(price_series: Series) -> float:
    """Calculate maximum drawdown from price series."""
    if DataProcessor.is_empty(price_series):
        return 0.0

    if DATA_BACKEND == "polars":
        # Polars implementation
        cumulative = price_series.cum_max()
        drawdowns = (price_series - cumulative) / cumulative
        return float(drawdowns.min())
    # Pandas implementation
    cumulative = price_series.expanding().max()
    drawdowns = (price_series - cumulative) / cumulative
    return float(drawdowns.min())
