"""Tests for technical indicators"""
import pytest
import pandas as pd
import numpy as np
from gp_quant.strategy.indicators import TechnicalIndicators


class TestTechnicalIndicators:
    def test_basic_indicators(self, sample_ohlcv):
        ti = TechnicalIndicators(sample_ohlcv)
        assert "change" in ti.data.columns
        assert "range" in ti.data.columns

    def test_ma(self, sample_ohlcv):
        ti = TechnicalIndicators(sample_ohlcv)
        df = ti.add_indicator("ma5")
        assert "ma5" in df.columns
        assert not df["ma5"].iloc[4:].isna().any()

    def test_macd(self, sample_ohlcv):
        ti = TechnicalIndicators(sample_ohlcv)
        df = ti.add_indicator("macd")
        assert "macd" in df.columns
        assert "signal" in df.columns
        assert "histogram" in df.columns

    def test_rsi(self, sample_ohlcv):
        ti = TechnicalIndicators(sample_ohlcv)
        df = ti.add_indicator("rsi")
        assert "rsi" in df.columns
        valid_rsi = df["rsi"].dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()

    def test_bollinger(self, sample_ohlcv):
        ti = TechnicalIndicators(sample_ohlcv)
        df = ti.add_indicator("bollinger")
        assert "bb_upper" in df.columns
        assert "bb_middle" in df.columns
        assert "bb_lower" in df.columns
        assert (df["bb_upper"].dropna() >= df["bb_middle"].dropna()).all()

    def test_atr(self, sample_ohlcv):
        ti = TechnicalIndicators(sample_ohlcv)
        df = ti.add_indicator("atr")
        assert "atr" in df.columns
        assert (df["atr"].dropna() >= 0).all()

    def test_adx(self, sample_ohlcv):
        ti = TechnicalIndicators(sample_ohlcv)
        df = ti.add_indicator("adx")
        assert "adx" in df.columns

    def test_stochastic(self, sample_ohlcv):
        ti = TechnicalIndicators(sample_ohlcv)
        df = ti.add_indicator("stochastic")
        assert "k" in df.columns
        assert "d" in df.columns

    def test_cci(self, sample_ohlcv):
        ti = TechnicalIndicators(sample_ohlcv)
        df = ti.add_indicator("cci")
        assert "cci" in df.columns

    def test_wr(self, sample_ohlcv):
        ti = TechnicalIndicators(sample_ohlcv)
        df = ti.add_indicator("wr")
        assert "wr" in df.columns

    def test_obv(self, sample_ohlcv):
        ti = TechnicalIndicators(sample_ohlcv)
        df = ti.add_indicator("obv")
        assert "obv" in df.columns

    def test_mfi(self, sample_ohlcv):
        ti = TechnicalIndicators(sample_ohlcv)
        df = ti.add_indicator("mfi")
        assert "mfi" in df.columns

    def test_unknown_indicator(self, sample_ohlcv):
        ti = TechnicalIndicators(sample_ohlcv)
        with pytest.raises(ValueError, match="Unknown indicator"):
            ti.add_indicator("nonexistent")

    def test_get_all_indicators(self, sample_ohlcv):
        ti = TechnicalIndicators(sample_ohlcv)
        df = ti.get_all_indicators()
        expected = ["rsi", "macd", "signal", "histogram", "bb_upper", "bb_lower",
                     "atr", "adx", "k", "d", "cci", "wr", "obv", "mfi"]
        for col in expected:
            assert col in df.columns
