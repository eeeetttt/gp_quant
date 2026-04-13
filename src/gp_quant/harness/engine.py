"""
Harness 引擎模块
连接策略研究和实盘交易的中间层：调度器、风控、订单执行、仓位管理
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..data.fetcher import DataFetcher
from ..data.processor import StockDataProcessor
from ..strategy.base import Signal, Strategy

logger = logging.getLogger(__name__)


# ──────────────────────────────────────
# 核心数据类
# ──────────────────────────────────────

class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    PARTIAL = "partial"


@dataclass
class HarnessOrder:
    """统一订单对象，贯穿信号 → 下单 → 成交全流程"""
    order_id: str
    symbol: str
    side: Signal
    quantity: float
    price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_price: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = ""
    reason: str = ""  # reject / cancel 原因

    def __post_init__(self):
        if not self.updated_at:
            self.updated_at = self.created_at

    def fill(self, price: float, quantity: Optional[float] = None):
        self.filled_price = price
        self.filled_quantity = quantity or self.quantity
        self.status = OrderStatus.FILLED
        self.updated_at = datetime.now().isoformat()

    def reject(self, reason: str):
        self.status = OrderStatus.REJECTED
        self.reason = reason
        self.updated_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "price": self.price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "status": self.status.value,
            "filled_quantity": self.filled_quantity,
            "filled_price": self.filled_price,
            "created_at": self.created_at,
            "reason": self.reason,
        }


@dataclass
class RiskConfig:
    """风控配置"""
    max_position_pct: float = 0.1        # 单标的最大仓位占比
    max_drawdown_pct: float = 0.15       # 最大回撤触发线
    max_daily_trades: int = 50           # 单日最大交易次数
    max_total_orders: int = 1000         # 总最大订单数
    stop_loss_pct: float = 0.05          # 默认止损百分比
    take_profit_pct: float = 0.10        # 默认止盈百分比
    min_capital: float = 10000.0         # 最低资金要求


@dataclass
class ScheduleConfig:
    """调度配置"""
    interval_seconds: int = 60           # 轮询间隔 (秒)
    market_open: str = "09:25"           # A 股开盘集合竞价
    market_close: str = "15:05"          # A 股收盘
    trading_days_only: bool = True       # 仅交易日运行


# ──────────────────────────────────────
# 风控引擎
# ──────────────────────────────────────

class RiskManager:
    """风控检查器，在订单提交前拦截危险交易"""

    def __init__(self, config: RiskConfig):
        self.config = config
        self._daily_trade_count: Dict[str, int] = {}

    def check_order(
        self,
        order: HarnessOrder,
        portfolio_value: float,
        current_position_qty: float,
    ) -> Tuple[bool, str]:
        """
        检查订单是否合规

        Returns:
            (通过, 拒绝原因)
        """
        # 1. 单标的仓位上限
        notional = order.quantity * order.price
        if portfolio_value > 0 and notional / portfolio_value > self.config.max_position_pct:
            return False, f"Position would exceed {self.config.max_position_pct:.0%} limit"

        # 2. 止损 / 止盈校验
        if order.stop_loss and order.price < order.stop_loss:
            return False, f"Price {order.price} below stop-loss {order.stop_loss}"

        # 3. 日交易频次
        today = datetime.now().strftime("%Y-%m-%d")
        daily_count = self._daily_trade_count.get(today, 0)
        if daily_count >= self.config.max_daily_trades:
            return False, f"Daily trade limit ({self.config.max_daily_trades}) reached"

        # 4. 总订单数
        # (由 HarnessEngine 在外部检查)

        return True, ""

    def check_portfolio_risk(
        self,
        portfolio_value: float,
        initial_capital: float,
    ) -> Tuple[bool, str]:
        """检查组合级风险"""
        if initial_capital <= 0:
            return True, ""

        drawdown = (initial_capital - portfolio_value) / initial_capital
        if drawdown > self.config.max_drawdown_pct:
            return False, f"Drawdown {drawdown:.2%} exceeds limit {self.config.max_drawdown_pct:.2%}"

        if portfolio_value < self.config.min_capital:
            return False, f"Capital {portfolio_value:.0f} below minimum {self.config.min_capital:.0f}"

        return True, ""

    def record_trade(self):
        today = datetime.now().strftime("%Y-%m-%d")
        self._daily_trade_count[today] = self._daily_trade_count.get(today, 0) + 1


# ──────────────────────────────────────
# 仓位管理器
# ──────────────────────────────────────

class PositionSizer(ABC):
    """仓位大小计算器基类"""

    @abstractmethod
    def calculate_size(
        self,
        signal: Signal,
        price: float,
        portfolio_value: float,
        volatility: float,
        **kwargs: Any,
    ) -> float:
        """返回建议下单数量"""
        ...


class FixedFractionSizer(PositionSizer):
    """固定比例仓位管理器"""

    def __init__(self, risk_pct: float = 0.02):
        """
        Args:
            risk_pct: 单笔风险占总资金比例
        """
        self.risk_pct = risk_pct

    def calculate_size(
        self,
        signal: Signal,
        price: float,
        portfolio_value: float,
        volatility: float,
        **kwargs: Any,
    ) -> float:
        if price <= 0 or signal != Signal.BUY:
            return 0.0

        risk_amount = portfolio_value * self.risk_pct
        # 以 volatility 作为止损参考
        stop_distance = price * volatility if volatility > 0 else price * 0.05
        if stop_distance <= 0:
            return 0.0

        quantity = risk_amount / stop_distance
        # A 股整手
        return int(quantity / 100) * 100


class FixedAllocationSizer(PositionSizer):
    """固定比例分配仓位管理器"""

    def __init__(self, allocation_pct: float = 0.1):
        self.allocation_pct = allocation_pct

    def calculate_size(
        self,
        signal: Signal,
        price: float,
        portfolio_value: float,
        volatility: float,
        **kwargs: Any,
    ) -> float:
        if price <= 0 or signal != Signal.BUY:
            return 0.0
        quantity = portfolio_value * self.allocation_pct / price
        return int(quantity / 100) * 100


# ──────────────────────────────────────
# 执行器抽象
# ──────────────────────────────────────

class ExecutionEngine(ABC):
    """订单执行器基类"""

    @abstractmethod
    def submit_order(self, order: HarnessOrder) -> HarnessOrder:
        """提交订单并返回更新后的状态"""
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        ...

    @abstractmethod
    def get_fill_price(self, symbol: str) -> Optional[float]:
        """获取实际成交价"""
        ...


class PaperExecutionEngine(ExecutionEngine):
    """模拟盘执行器 — 以最新 bar 的 close 价成交"""

    def __init__(self, fetcher: DataFetcher, fee_rate: float = 0.001):
        self.fetcher = fetcher
        self.fee_rate = fee_rate
        self._order_book: Dict[str, HarnessOrder] = {}
        self._order_counter = 0

    def submit_order(self, order: HarnessOrder) -> HarnessOrder:
        self._order_counter += 1
        self._order_book[order.order_id] = order
        order.status = OrderStatus.SUBMITTED

        try:
            quote = self.fetcher.fetch_quote(order.symbol)
            if "error" in quote:
                order.reject(f"Quote error: {quote['error']}")
                return order

            fill_price = quote["current_price"]
            if fill_price <= 0:
                order.reject("Invalid fill price")
                return order

            order.fill(fill_price)
            logger.info(
                "[PAPER] FILLED %s %s %.0f @ %.2f",
                order.side.value.upper(), order.symbol,
                order.filled_quantity, order.filled_price,
            )
        except Exception as e:
            order.reject(str(e))
            logger.error("[PAPER] REJECTED %s: %s", order.order_id, e)

        return order

    def cancel_order(self, order_id: str) -> bool:
        order = self._order_book.get(order_id)
        if order and order.status in (OrderStatus.PENDING, OrderStatus.SUBMITTED):
            order.status = OrderStatus.CANCELLED
            return True
        return False

    def get_fill_price(self, symbol: str) -> Optional[float]:
        try:
            quote = self.fetcher.fetch_quote(symbol)
            return quote.get("current_price")
        except Exception:
            return None


# ──────────────────────────────────────
# Harness Engine — 核心调度器
# ──────────────────────────────────────

class HarnessEngine:
    """
    实时交易调度引擎

    职责:
    1. 定时拉取行情数据
    2. 运行策略生成信号
    3. 风控检查
    4. 仓位计算
    5. 提交订单给执行器
    6. 维护组合状态
    """

    def __init__(
        self,
        strategy: Strategy,
        fetcher: DataFetcher,
        executor: ExecutionEngine,
        *,
        risk_config: Optional[RiskConfig] = None,
        schedule_config: Optional[ScheduleConfig] = None,
        position_sizer: Optional[PositionSizer] = None,
        initial_capital: float = 100_000.0,
        symbols: Optional[List[str]] = None,
    ):
        self.strategy = strategy
        self.fetcher = fetcher
        self.executor = executor
        self.risk_manager = RiskManager(risk_config or RiskConfig())
        self.schedule_config = schedule_config or ScheduleConfig()
        self.position_sizer = position_sizer or FixedFractionSizer()
        self.symbols = symbols or ["000001.SZ"]

        # 组合状态
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self._positions: Dict[str, Dict[str, Any]] = {}  # symbol → {qty, avg_price, ...}
        self._orders: Dict[str, HarnessOrder] = {}
        self._order_counter = 0

        # 数据处理
        self._processor = StockDataProcessor()
        self._history: Dict[str, pd.DataFrame] = {}  # symbol → historical data
        self._lookback_days = 120

    # ── 数据层 ──────────────────────

    def load_history(self, symbol: str) -> pd.DataFrame:
        """加载历史数据用于策略计算"""
        if symbol in self._history:
            return self._history[symbol]

        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - pd.Timedelta(days=self._lookback_days)).strftime("%Y-%m-%d")

        df = self.fetcher.fetch(symbol, start, end)
        df = self._processor.clean(df)
        self._history[symbol] = df
        logger.info("Loaded %d bars for %s", len(df), symbol)
        return df

    def refresh_data(self) -> Dict[str, pd.DataFrame]:
        """刷新所有标的的数据"""
        for symbol in self.symbols:
            try:
                df = self.load_history(symbol)
                # 追加最新行情 (fetch_quote 模拟盘用)
                quote = self.fetcher.fetch_quote(symbol)
                if "error" not in quote and quote.get("current_price", 0) > 0:
                    last_date = datetime.now().strftime("%Y-%m-%d")
                    if str(df["date"].iloc[-1])[:10] != last_date:
                        new_row = pd.DataFrame([{
                            "date": last_date,
                            "open": quote.get("current_price", 0),
                            "high": quote.get("day_high", 0),
                            "low": quote.get("day_low", 0),
                            "close": quote.get("current_price", 0),
                            "volume": quote.get("volume", 0),
                        }])
                        self._history[symbol] = pd.concat(
                            [df, new_row], ignore_index=True
                        )
            except Exception as e:
                logger.error("Failed to refresh data for %s: %s", symbol, e)

        return self._history

    # ── 策略层 ──────────────────────

    def evaluate_strategy(self, symbol: str) -> Signal:
        """对单个标的运行策略"""
        df = self._history.get(symbol)
        if df is None or len(df) == 0:
            return Signal.HOLD

        return self.strategy.generate_signal(df)

    # ── 订单管理 ────────────────────

    def _create_order(
        self,
        symbol: str,
        signal: Signal,
        price: float,
        quantity: float,
    ) -> HarnessOrder:
        self._order_counter += 1
        order_id = f"ORD-{self._order_counter:06d}"

        risk_cfg = self.risk_manager.config
        stop_loss = price * (1 - risk_cfg.stop_loss_pct) if signal == Signal.BUY else None
        take_profit = price * (1 + risk_cfg.take_profit_pct) if signal == Signal.BUY else None

        order = HarnessOrder(
            order_id=order_id,
            symbol=symbol,
            side=signal,
            quantity=quantity,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        self._orders[order_id] = order
        return order

    def process_signal(self, symbol: str, signal: Signal) -> Optional[HarnessOrder]:
        """处理策略信号，经风控和仓位计算后下单"""
        if signal == Signal.HOLD:
            return None

        # 获取当前价格
        quote = self.fetcher.fetch_quote(symbol)
        price = quote.get("current_price", 0) if isinstance(quote, dict) else 0
        if price <= 0:
            logger.warning("Invalid price for %s, skipping", symbol)
            return None

        portfolio_value = self._portfolio_value()

        # 已有持仓 → SELL 信号触发平仓
        if symbol in self._positions:
            pos = self._positions[symbol]
            if signal == Signal.SELL and pos["quantity"] > 0:
                order = self._create_order(symbol, Signal.SELL, price, pos["quantity"])
                order.status = OrderStatus.SUBMITTED
                return self._submit_order(order)
            return None

        # 无持仓 → BUY 信号
        if signal == Signal.BUY:
            volatility = self._get_volatility(symbol)
            quantity = self.position_sizer.calculate_size(
                signal, price, portfolio_value, volatility,
            )
            if quantity <= 0:
                logger.debug("Position size is 0 for %s", symbol)
                return None

            order = self._create_order(symbol, Signal.BUY, price, quantity)

            # 风控检查
            passed, reason = self.risk_manager.check_order(
                order, portfolio_value,
                self._positions.get(symbol, {}).get("quantity", 0),
            )
            if not passed:
                order.reject(reason)
                logger.info("Order REJECTED: %s", reason)
                return order

            # 组合级风控
            passed, reason = self.risk_manager.check_portfolio_risk(
                portfolio_value, self.initial_capital,
            )
            if not passed:
                order.reject(reason)
                logger.warning("Portfolio risk: %s", reason)
                return order

            return self._submit_order(order)

        return None

    def _submit_order(self, order: HarnessOrder) -> HarnessOrder:
        """提交订单到执行器"""
        result = self.executor.submit_order(order)

        if result.status == OrderStatus.FILLED:
            self._update_position(result)
            self.risk_manager.record_trade()
        return result

    def _update_position(self, order: HarnessOrder):
        """根据成交订单更新持仓"""
        symbol = order.symbol

        if order.side == Signal.BUY:
            if symbol in self._positions:
                pos = self._positions[symbol]
                total_qty = pos["quantity"] + order.filled_quantity
                pos["avg_price"] = (
                    pos["avg_price"] * pos["quantity"]
                    + order.filled_price * order.filled_quantity
                ) / total_qty
                pos["quantity"] = total_qty
            else:
                self._positions[symbol] = {
                    "quantity": order.filled_quantity,
                    "avg_price": order.filled_price,
                    "stop_loss": order.stop_loss,
                    "take_profit": order.take_profit,
                }
            self.current_capital -= order.filled_quantity * order.filled_price

        elif order.side == Signal.SELL:
            if symbol in self._positions:
                pos = self._positions[symbol]
                sell_qty = min(order.filled_quantity, pos["quantity"])
                realized = (order.filled_price - pos["avg_price"]) * sell_qty
                self.current_capital += sell_qty * order.filled_price
                pos["quantity"] -= sell_qty
                if pos["quantity"] <= 0:
                    del self._positions[symbol]
                logger.info(
                    "Realized PnL for %s: %.2f", symbol, realized,
                )

    # ── 止损 / 止盈检查 ─────────────

    def check_stops(self) -> List[HarnessOrder]:
        """遍历持仓，触发止损止盈"""
        orders: List[HarnessOrder] = []

        for symbol, pos in list(self._positions.items()):
            quote = self.fetcher.fetch_quote(symbol)
            price = quote.get("current_price", 0) if isinstance(quote, dict) else 0
            if price <= 0:
                continue

            avg = pos["avg_price"]
            should_sell = False

            # 止损
            if pos.get("stop_loss") and price <= pos["stop_loss"]:
                logger.warning("STOP LOSS triggered for %s: %.2f <= %.2f",
                               symbol, price, pos["stop_loss"])
                should_sell = True

            # 止盈
            if pos.get("take_profit") and price >= pos["take_profit"]:
                logger.info("TAKE PROFIT triggered for %s: %.2f >= %.2f",
                            symbol, price, pos["take_profit"])
                should_sell = True

            # 组合级回撤止损
            ok, reason = self.risk_manager.check_portfolio_risk(
                self._portfolio_value(), self.initial_capital,
            )
            if not ok:
                logger.warning("Portfolio-level stop: %s", reason)
                should_sell = True

            if should_sell:
                order = self._create_order(symbol, Signal.SELL, price, pos["quantity"])
                orders.append(self._submit_order(order))

        return orders

    # ── 辅助 ────────────────────────

    def _portfolio_value(self) -> float:
        """当前组合总市值"""
        market_value = 0.0
        for symbol, pos in self._positions.items():
            quote = self.fetcher.fetch_quote(symbol)
            price = quote.get("current_price", 0) if isinstance(quote, dict) else pos["avg_price"]
            market_value += pos["quantity"] * price
        return self.current_capital + market_value

    def _get_volatility(self, symbol: str) -> float:
        """计算标的的日收益率波动率"""
        df = self._history.get(symbol)
        if df is None or len(df) < 20:
            return 0.02  # 默认 2%
        returns = df["close"].pct_change().dropna()
        return float(returns.std()) if len(returns) > 0 else 0.02

    def get_positions(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._positions)

    def get_orders(self) -> Dict[str, Dict[str, Any]]:
        return {oid: o.to_dict() for oid, o in self._orders.items()}

    def summary(self) -> Dict[str, Any]:
        return {
            "initial_capital": self.initial_capital,
            "current_capital": self.current_capital,
            "portfolio_value": self._portfolio_value(),
            "positions": {
                sym: {**pos, "current_value": pos["quantity"] * self.fetcher.fetch_quote(sym).get("current_price", 0)}
                for sym, pos in self._positions.items()
            },
            "total_orders": len(self._orders),
            "filled_orders": sum(1 for o in self._orders.values() if o.status == OrderStatus.FILLED),
        }

    # ── 运行循环 ────────────────────

    def tick(self) -> List[HarnessOrder]:
        """
        执行一次完整的 tick 循环:
        1. 刷新数据
        2. 评估策略
        3. 检查止损止盈
        4. 生成并执行订单

        Returns:
            本轮产生的订单列表
        """
        orders: List[HarnessOrder] = []

        # 1. 刷新行情数据
        self.refresh_data()

        # 2. 策略评估 + 下单
        for symbol in self.symbols:
            signal = self.evaluate_strategy(symbol)
            order = self.process_signal(symbol, signal)
            if order:
                orders.append(order)

        # 3. 止损止盈检查
        stop_orders = self.check_stops()
        orders.extend(stop_orders)

        return orders
