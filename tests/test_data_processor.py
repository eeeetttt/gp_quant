"""Tests for data processor"""
import pytest
import pandas as pd
import numpy as np
from gp_quant.data.processor import StockDataProcessor


class TestStockDataProcessor:
    def test_clean_basic(self, sample_ohlcv):
        proc = StockDataProcessor()
        cleaned = proc.clean(sample_ohlcv)
        assert "change" in cleaned.columns
        assert "change_abs" in cleaned.columns
        assert "range" in cleaned.columns
        assert "range_pct" in cleaned.columns

    def test_clean_handles_na(self):
        proc = StockDataProcessor()
        df = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=5),
            "open": [10, 11, np.nan, 13, 14],
            "high": [11, 12, 13, 14, 15],
            "low": [9, 10, 11, 12, 13],
            "close": [10.5, 11.5, 12.5, 13.5, 14.5],
            "volume": [1000] * 5,
        })
        cleaned = proc.clean(df)
        assert len(cleaned) > 0

    def test_normalize_minmax(self, sample_ohlcv):
        proc = StockDataProcessor()
        normalized = proc.normalize(sample_ohlcv, method="minmax")
        numeric_cols = normalized.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            assert normalized[col].min() >= 0
            assert normalized[col].max() <= 1

    def test_normalize_zscore(self, sample_ohlcv):
        proc = StockDataProcessor()
        normalized = proc.normalize(sample_ohlcv, method="zscore")
        numeric_cols = normalized.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            assert abs(normalized[col].mean()) < 1e-10 or normalized[col].std() > 0

    def test_calculate_returns(self, sample_ohlcv):
        proc = StockDataProcessor()
        result = proc.calculate_returns(sample_ohlcv, periods=[1, 5])
        assert "return_1d" in result.columns
        assert "return_5d" in result.columns
