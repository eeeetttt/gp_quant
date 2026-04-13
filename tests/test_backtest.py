"""Tests for backtest engine"""
import pytest
import pandas as pd
import numpy as np
from gp_quant.backtest.engine import BacktestEngine, Trade, Portfolio


class TestTrade:
    def test_close_buy_trade(self):
        trade = Trade(
            symbol="000001.SZ",
            entry_date="2023-01-01",
            entry_price=10.0,
            quantity=100,
            side="BUY",
            fees=1.0,
        )
        pnl = trade.close(12.0)
        assert trade.status == "CLOSED"
        assert trade.exit_price == 12.0
        assert pnl == (12.0 - 10.0) * 100 - 1.0

    def test_close_sell_trade(self):
        trade = Trade(
            symbol="000001.SZ",
            entry_date="2023-01-01",
            entry_price=10.0,
            quantity=100,
            side="SELL",
            fees=1.0,
        )
        pnl = trade.close(8.0)
        assert trade.pnl > 0

    def test_to_dict(self):
        trade = Trade(
            symbol="000001.SZ",
            entry_date="2023-01-01",
            entry_price=10.0,
            quantity=100,
            side="BUY",
        )
        trade.close(12.0)
        d = trade.to_dict()
        assert "symbol" in d
        assert d["exit_price"] == 12.0


class TestPortfolio:
    def test_win_rate(self):
        portfolio = Portfolio(initial_capital=100000)
        t1 = Trade(symbol="A", entry_date="2023-01-01", entry_price=10, quantity=100, side="BUY")
        t1.close(12.0)
        portfolio.add_trade(t1)
        t2 = Trade(symbol="B", entry_date="2023-01-02", entry_price=10, quantity=100, side="BUY")
        t2.close(8.0)
        portfolio.add_trade(t2)
        assert portfolio.get_win_rate() == 50.0

    def test_total_return(self):
        portfolio = Portfolio(initial_capital=100000)
        t = Trade(symbol="A", entry_date="2023-01-01", entry_price=10, quantity=100, side="BUY")
        t.close(15.0)
        portfolio.add_trade(t)
        assert portfolio.get_total_return() == pytest.approx(0.5, rel=1e-2)


class TestBacktestEngine:
    def test_run_with_buy_sell_signals(self, sample_ohlcv):
        df = sample_ohlcv.copy()
        df["signal"] = "HOLD"
        # Simple signal: buy on first day, sell on last day
        df.iloc[50, df.columns.get_loc("signal")] = "BUY"
        df.iloc[100, df.columns.get_loc("signal")] = "SELL"

        engine = BacktestEngine(initial_capital=100000.0, fee_rate=0.001)
        engine.set_market_data(df)
        engine.set_signals(df)
        results = engine.run()

        assert results["total_trades"] >= 1
        assert "sharpe_ratio" in results
        assert "max_drawdown" in results

    def test_run_no_signals(self, sample_ohlcv):
        df = sample_ohlcv.copy()
        df["signal"] = "HOLD"

        engine = BacktestEngine(initial_capital=100000.0)
        engine.set_market_data(df)
        engine.set_signals(df)
        results = engine.run()

        assert results["total_trades"] == 0
        assert results["total_return"] == 0.0

    def test_missing_market_data(self):
        engine = BacktestEngine()
        with pytest.raises(ValueError, match="Market data not set"):
            engine.run()

    def test_missing_signals(self, sample_ohlcv):
        engine = BacktestEngine()
        engine.set_market_data(sample_ohlcv)
        with pytest.raises(ValueError, match="Signals not set"):
            engine.run()
