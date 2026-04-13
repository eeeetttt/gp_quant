"""Shared test fixtures"""
import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Generate sample OHLCV data for testing"""
    np.random.seed(42)
    n = 500
    dates = pd.date_range(start="2023-01-01", periods=n, freq="D")

    base = 50 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame({
        "date": dates,
        "open": base,
        "high": base + np.abs(np.random.randn(n)),
        "low": base - np.abs(np.random.randn(n)),
        "close": base + np.random.randn(n) * 0.3,
        "volume": np.random.randint(1000000, 10000000, n),
    })
    df["high"] = np.maximum(df["open"], df["high"])
    df["low"] = np.minimum(df["open"], df["low"])
    df["close"] = df["close"].clip(lower=df["low"] * 0.9, upper=df["high"] * 1.1)
    return df
