"""
策略基类
"""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
import pandas as pd
import numpy as np


class Signal(Enum):
    """交易信号"""
    BUY = "buy"           # 买入
    SELL = "sell"         # 卖出
    HOLD = "hold"         # 持有


@dataclass
class Order:
    """订单类"""
    symbol: str
    side: Signal
    quantity: float
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            from datetime import datetime
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "price": self.price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "timestamp": self.timestamp
        }


@dataclass
class Position:
    """持仓类"""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    pnl: float = 0.0
    pnl_pct: float = 0.0

    def update(self, current_price: float):
        self.current_price = current_price
        self.pnl = (current_price - self.entry_price) * self.quantity
        self.pnl_pct = (current_price - self.entry_price) / self.entry_price * 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct
        }


class Strategy(ABC):
    """策略基类"""

    def __init__(self, name: str = "Strategy", params: Optional[Dict] = None):
        """
        初始化策略

        Args:
            name: 策略名称
            params: 策略参数字典
        """
        self.name = name
        self.params = params or {}
        self.initial_capital = 100000.0

    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> Signal:
        """
        生成交易信号

        Args:
            data: 包含 OHLCV 及技术指标的数据

        Returns:
            Signal 类型的信号
        """
        pass

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """获取策略参数"""
        pass

    def set_parameters(self, params: Dict[str, Any]):
        """设置策略参数"""
        self.params.update(params)

    def validate(self, data: pd.DataFrame) -> bool:
        """
        验证数据是否满足策略要求

        Args:
            data: 输入数据

        Returns:
            验证是否通过
        """
        required_columns = {"date", "open", "high", "low", "close", "volume"}
        return required_columns.issubset(set(data.columns))

    def calculate_position_size(self, signal_data: pd.DataFrame,
                                current_price: float,
                                max_allocation: float = 0.1) -> float:
        """
        计算仓位大小

        Args:
            signal_data: 信号相关数据
            current_price: 当前价格
            max_allocation: 最大资金使用比例

        Returns:
            建议购买的数量
        """
        # 简单实现：按资金管理规则
        capital = self.initial_capital * max_allocation
        return capital / current_price


class SimpleMovingAverageStrategy(Strategy):
    """简单移动平均策略示例"""

    def __init__(self, fast_window: int = 5, slow_window: int = 20):
        super().__init__("SMA Strategy", {
            "fast_window": fast_window,
            "slow_window": slow_window
        })

    def generate_signal(self, data: pd.DataFrame) -> Signal:
        """
        基于双均线策略生成信号

        金叉买入，死叉卖出
        """
        if not self.validate(data):
            return Signal.HOLD

        df = data.copy()

        # 计算移动平均线
        df["ma_fast"] = df["close"].rolling(window=self.params["fast_window"]).mean()
        df["ma_slow"] = df["close"].rolling(window=self.params["slow_window"]).mean()

        # 删除 NaN 值
        df = df.dropna()

        if len(df) < 2:
            return Signal.HOLD

        # 获取最新数据
        current = df.iloc[-1]
        previous = df.iloc[-2]

        # 金叉：快线从下向上穿过慢线
        if previous["ma_fast"] <= previous["ma_slow"] and current["ma_fast"] > current["ma_slow"]:
            return Signal.BUY

        # 死叉：快线从上向下穿过慢线
        if previous["ma_fast"] >= previous["ma_slow"] and current["ma_fast"] < current["ma_slow"]:
            return Signal.SELL

        return Signal.HOLD

    def get_parameters(self) -> Dict[str, Any]:
        return self.params.copy()


class MomentumStrategy(Strategy):
    """动量策略示例"""

    def __init__(self, lookback_period: int = 20, threshold: float = 2.0):
        super().__init__("Momentum Strategy", {
            "lookback_period": lookback_period,
            "threshold": threshold
        })

    def generate_signal(self, data: pd.DataFrame) -> Signal:
        """基于收益率动量生成信号"""
        if not self.validate(data):
            return Signal.HOLD

        df = data.copy()

        # 计算收益率
        df["return"] = df["close"].pct_change(self.params["lookback_period"]) * 100

        if len(df) < self.params["lookback_period"] + 1:
            return Signal.HOLD

        current_return = df["return"].iloc[-1]

        if current_return > self.params["threshold"]:
            return Signal.SELL  # 超买，卖出

        if current_return < -self.params["threshold"]:
            return Signal.BUY  # 超卖，买入

        return Signal.HOLD

    def get_parameters(self) -> Dict[str, Any]:
        return self.params.copy()
