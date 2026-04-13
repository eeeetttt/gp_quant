"""Tests for harness engine"""
import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np

from gp_quant.harness.engine import (
    HarnessEngine,
    HarnessOrder,
    OrderStatus,
    RiskConfig,
    RiskManager,
    FixedFractionSizer,
    FixedAllocationSizer,
    PaperExecutionEngine,
)
from gp_quant.strategy.base import Signal, Strategy


class DummyStrategy(Strategy):
    """A test strategy that returns pre-configured signals"""

    def __init__(self, signal: Signal = Signal.HOLD):
        super().__init__("Dummy")
        self._signal = signal

    def generate_signal(self, data: pd.DataFrame) -> Signal:
        return self._signal

    def get_parameters(self) -> dict:
        return {}


def make_mock_fetcher(quote_price: float = 10.0):
    """Create a mock fetcher that returns a fixed quote"""
    fetcher = MagicMock()
    fetcher.fetch_quote.return_value = {
        "symbol": "000001.SZ",
        "current_price": quote_price,
        "day_high": quote_price + 0.5,
        "day_low": quote_price - 0.5,
        "volume": 1000000,
    }
    fetcher.fetch.return_value = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=130, freq="D"),
        "open": 10 + np.random.randn(130) * 0.1,
        "high": 10.5 + np.random.randn(130) * 0.1,
        "low": 9.5 + np.random.randn(130) * 0.1,
        "close": 10 + np.random.randn(130) * 0.1,
        "volume": np.random.randint(500000, 2000000, 130),
    })
    return fetcher


class TestHarnessOrder:
    def test_order_creation(self):
        order = HarnessOrder(
            order_id="ORD-001",
            symbol="000001.SZ",
            side=Signal.BUY,
            quantity=1000,
            price=10.0,
        )
        assert order.status == OrderStatus.PENDING

    def test_order_fill(self):
        order = HarnessOrder(
            order_id="ORD-001",
            symbol="000001.SZ",
            side=Signal.BUY,
            quantity=1000,
            price=10.0,
        )
        order.fill(10.1)
        assert order.status == OrderStatus.FILLED
        assert order.filled_price == 10.1
        assert order.filled_quantity == 1000

    def test_order_reject(self):
        order = HarnessOrder(
            order_id="ORD-001",
            symbol="000001.SZ",
            side=Signal.BUY,
            quantity=1000,
            price=10.0,
        )
        order.reject("Insufficient capital")
        assert order.status == OrderStatus.REJECTED
        assert "capital" in order.reason.lower()

    def test_order_to_dict(self):
        order = HarnessOrder(
            order_id="ORD-001",
            symbol="000001.SZ",
            side=Signal.BUY,
            quantity=1000,
            price=10.0,
        )
        d = order.to_dict()
        assert d["order_id"] == "ORD-001"
        assert d["side"] == "buy"


class TestRiskManager:
    def test_check_order_pass(self):
        risk = RiskManager(RiskConfig(max_position_pct=0.1))
        order = HarnessOrder(
            order_id="ORD-001",
            symbol="000001.SZ",
            side=Signal.BUY,
            quantity=1000,
            price=10.0,
        )
        passed, _ = risk.check_order(order, 100000, 0)
        assert passed

    def test_check_order_position_limit(self):
        risk = RiskManager(RiskConfig(max_position_pct=0.05))
        order = HarnessOrder(
            order_id="ORD-001",
            symbol="000001.SZ",
            side=Signal.BUY,
            quantity=10000,
            price=10.0,
        )
        passed, reason = risk.check_order(order, 100000, 0)
        assert not passed
        assert "limit" in reason.lower()

    def test_check_order_daily_limit(self):
        risk = RiskManager(RiskConfig(max_daily_trades=1))
        risk.record_trade()
        order = HarnessOrder(
            order_id="ORD-001",
            symbol="000001.SZ",
            side=Signal.BUY,
            quantity=1000,
            price=10.0,
        )
        passed, reason = risk.check_order(order, 100000, 0)
        assert not passed
        assert "limit" in reason.lower()

    def test_portfolio_drawdown(self):
        risk = RiskManager(RiskConfig(max_drawdown_pct=0.1))
        passed, _ = risk.check_portfolio_risk(95000, 100000)
        assert passed

        passed, _ = risk.check_portfolio_risk(80000, 100000)
        assert not passed

    def test_portfolio_min_capital(self):
        risk = RiskManager(RiskConfig(min_capital=10000))
        passed, _ = risk.check_portfolio_risk(5000, 100000)
        assert not passed


class TestPositionSizer:
    def test_fixed_fraction(self):
        sizer = FixedFractionSizer(risk_pct=0.02)
        qty = sizer.calculate_size(Signal.BUY, 10.0, 100000, 0.02)
        assert qty > 0
        assert qty % 100 == 0  # A股整手

    def test_fixed_fraction_no_buy(self):
        sizer = FixedFractionSizer()
        qty = sizer.calculate_size(Signal.SELL, 10.0, 100000, 0.02)
        assert qty == 0

    def test_fixed_allocation(self):
        sizer = FixedAllocationSizer(allocation_pct=0.1)
        qty = sizer.calculate_size(Signal.BUY, 10.0, 100000, 0.02)
        assert qty == 1000  # 100000 * 0.1 / 10 = 1000, rounded to lot

    def test_fixed_allocation_no_buy(self):
        sizer = FixedAllocationSizer()
        qty = sizer.calculate_size(Signal.HOLD, 10.0, 100000, 0.02)
        assert qty == 0


class TestHarnessEngine:
    def test_engine_init(self):
        fetcher = make_mock_fetcher()
        executor = MagicMock()
        strategy = DummyStrategy()

        engine = HarnessEngine(
            strategy=strategy,
            fetcher=fetcher,
            executor=executor,
            symbols=["000001.SZ"],
        )
        assert engine.current_capital == 100000.0
        assert len(engine.symbols) == 1

    def test_tick_no_trades_on_hold(self):
        """When strategy returns HOLD, no orders should be created"""
        fetcher = make_mock_fetcher()
        executor = MagicMock()
        strategy = DummyStrategy(Signal.HOLD)

        engine = HarnessEngine(
            strategy=strategy,
            fetcher=fetcher,
            executor=executor,
            symbols=["000001.SZ"],
        )
        orders = engine.tick()
        # No orders since signal is HOLD
        assert len(orders) == 0

    def test_tick_creates_buy_order(self):
        """When strategy returns BUY, a buy order should be created"""
        fetcher = make_mock_fetcher(quote_price=10.0)
        executor = MagicMock()
        strategy = DummyStrategy(Signal.BUY)

        engine = HarnessEngine(
            strategy=strategy,
            fetcher=fetcher,
            executor=executor,
            symbols=["000001.SZ"],
        )
        orders = engine.tick()
        assert len(orders) >= 1
        assert orders[0].side == Signal.BUY

    def test_portfolio_value(self):
        fetcher = make_mock_fetcher(quote_price=10.0)
        executor = MagicMock()
        strategy = DummyStrategy()

        engine = HarnessEngine(
            strategy=strategy,
            fetcher=fetcher,
            executor=executor,
            symbols=["000001.SZ"],
        )
        assert engine._portfolio_value() == 100000.0

    def test_summary(self):
        fetcher = make_mock_fetcher()
        executor = MagicMock()
        strategy = DummyStrategy()

        engine = HarnessEngine(
            strategy=strategy,
            fetcher=fetcher,
            executor=executor,
        )
        summary = engine.summary()
        assert "initial_capital" in summary
        assert "portfolio_value" in summary
        assert summary["initial_capital"] == 100000.0

    def test_get_volatility(self):
        fetcher = make_mock_fetcher()
        executor = MagicMock()
        strategy = DummyStrategy()

        engine = HarnessEngine(
            strategy=strategy,
            fetcher=fetcher,
            executor=executor,
        )
        # Load history first
        engine.load_history("000001.SZ")
        vol = engine._get_volatility("000001.SZ")
        assert vol > 0

    def test_get_volatility_no_data(self):
        fetcher = make_mock_fetcher()
        executor = MagicMock()
        strategy = DummyStrategy()

        engine = HarnessEngine(
            strategy=strategy,
            fetcher=fetcher,
            executor=executor,
        )
        vol = engine._get_volatility("UNKNOWN")
        assert vol == 0.02  # default
