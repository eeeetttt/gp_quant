"""
回测引擎模块
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import os


@dataclass
class Trade:
    """交易记录"""
    symbol: str
    entry_date: str
    exit_date: Optional[str] = None
    entry_price: float = 0.0
    exit_price: Optional[float] = None
    quantity: float = 0.0
    side: str = "BUY"  # "BUY" or "SELL"
    pnl: float = 0.0
    pnl_pct: float = 0.0
    status: str = "OPEN"  # "OPEN" or "CLOSED"
    entry_signal: Optional[str] = None
    exit_signal: Optional[str] = None
    fees: float = 0.0

    def close(self, exit_price: float, exit_signal: Optional[str] = None) -> float:
        """平仓"""
        self.exit_date = datetime.now().strftime("%Y-%m-%d")
        self.exit_price = exit_price
        self.status = "CLOSED"
        self.exit_signal = exit_signal

        if self.side == "BUY":
            self.pnl = (exit_price - self.entry_price) * self.quantity
            self.pnl_pct = (exit_price - self.entry_price) / self.entry_price * 100
        else:
            self.pnl = (self.entry_price - exit_price) * self.quantity
            self.pnl_pct = (self.entry_price - exit_price) / self.entry_price * 100

        self.pnl -= self.fees
        return self.pnl

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "entry_date": self.entry_date,
            "exit_date": self.exit_date,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "quantity": self.quantity,
            "side": self.side,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
            "status": self.status,
            "entry_signal": self.entry_signal,
            "exit_signal": self.exit_signal,
            "fees": self.fees
        }


@dataclass
class Portfolio:
    """投资组合"""
    initial_capital: float = 100000.0
    current_capital: float = 100000.0
    trades: List[Trade] = field(default_factory=list)
    positions: Dict[str, Trade] = field(default_factory=dict)

    def add_trade(self, trade: Trade):
        """添加交易"""
        self.trades.append(trade)

    def get_open_positions(self) -> List[Trade]:
        """获取未平仓头寸"""
        return [t for t in self.trades if t.status == "OPEN"]

    def get_closed_trades(self) -> List[Trade]:
        """获取已平仓交易"""
        return [t for t in self.trades if t.status == "CLOSED"]

    def get_total_pnl(self) -> float:
        """获取总盈亏"""
        return sum(t.pnl for t in self.trades if t.status == "CLOSED")

    def get_total_return(self) -> float:
        """获取总收益率"""
        if self.initial_capital == 0:
            return 0.0
        return self.get_total_pnl() / self.initial_capital * 100

    def get_trade_count(self) -> int:
        """获取交易次数"""
        return len([t for t in self.trades if t.status == "CLOSED"])

    def get_win_rate(self) -> float:
        """获取胜率"""
        closed = self.get_closed_trades()
        if len(closed) == 0:
            return 0.0
        winning = sum(1 for t in closed if t.pnl > 0)
        return winning / len(closed) * 100


class BacktestEngine:
    """回测引擎"""

    def __init__(self, initial_capital: float = 100000.0, fee_rate: float = 0.001):
        """
        初始化回测引擎

        Args:
            initial_capital: 初始资金
            fee_rate: 交易费率 (默认 0.1%)
        """
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.portfolio = Portfolio(initial_capital=initial_capital)
        self.market_data: Optional[pd.DataFrame] = None
        self.signals: Optional[pd.DataFrame] = None
        self.results: Dict[str, Any] = {}

    def set_market_data(self, data: pd.DataFrame):
        """设置市场数据"""
        self.market_data = data.copy()

    def set_signals(self, signals: pd.DataFrame):
        """设置交易信号"""
        self.signals = signals.copy()

    def run(self) -> Dict[str, Any]:
        """
        运行回测

        Returns:
            回测结果字典
        """
        if self.market_data is None:
            raise ValueError("Market data not set")

        if self.signals is None:
            raise ValueError("Signals not set")

        # 初始化投资组合
        self.portfolio = Portfolio(initial_capital=self.initial_capital)

        # 遍历信号
        for idx, signal_row in self.signals.iterrows():
            date = signal_row.get("date", idx)
            symbol = signal_row.get("symbol", "default")
            signal = signal_row.get("signal", "HOLD")

            # 获取当前价格
            current_price = self._get_price(date, symbol)
            if current_price is None:
                continue

            # 执行交易逻辑
            self._execute_trade(symbol, signal, current_price, date)

        # 关闭所有未平仓头寸
        self._close_all_positions(self.market_data.iloc[-1]["date"])

        # 计算回测结果
        self.results = self._calculate_results()

        return self.results

    def _get_price(self, date, symbol: str) -> Optional[float]:
        """获取价格"""
        if self.market_data is None:
            return None

        if isinstance(date, pd.Timestamp):
            date_str = date.strftime("%Y-%m-%d")
        else:
            date_str = str(date)

        row = self.market_data[self.market_data["date"] == date_str]
        if len(row) > 0:
            return row["close"].iloc[0]
        return None

    def _execute_trade(self, symbol: str, signal: str, current_price: float, date):
        """执行交易"""
        # 检查是否已有该标的的持仓
        if symbol in self.portfolio.positions:
            # 平仓
            position = self.portfolio.positions[symbol]
            pnl = position.close(current_price, exit_signal=signal)
            self.portfolio.current_capital += position.exit_price * position.quantity * (1 - self.fee_rate) + pnl
            del self.portfolio.positions[symbol]
        elif signal == "BUY" and self.portfolio.current_capital > current_price:
            # 开仓
            quantity = min(self.portfolio.current_capital * 0.9 / current_price, 10000)
            fees = quantity * current_price * self.fee_rate

            trade = Trade(
                symbol=symbol,
                entry_date=str(date),
                entry_price=current_price,
                quantity=quantity,
                side="BUY",
                fees=fees
            )

            self.portfolio.current_capital -= quantity * current_price * (1 + self.fee_rate)
            self.portfolio.positions[symbol] = trade
            self.portfolio.add_trade(trade)

    def _close_all_positions(self, end_date):
        """关闭所有持仓"""
        if self.market_data is None:
            return

        for symbol, position in list(self.portfolio.positions.items()):
            end_price = self._get_price(end_date, symbol)
            if end_price:
                position.close(end_price)

    def _calculate_results(self) -> Dict[str, Any]:
        """计算回测结果"""
        closed_trades = self.portfolio.get_closed_trades()

        if len(closed_trades) == 0:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "trades": []
            }

        # 计算收益序列
        equity_curve = self._calculate_equity_curve(closed_trades)

        # 计算各种指标
        results = {
            "total_trades": self.portfolio.get_trade_count(),
            "winning_trades": sum(1 for t in closed_trades if t.pnl > 0),
            "losing_trades": sum(1 for t in closed_trades if t.pnl <= 0),
            "win_rate": self.portfolio.get_win_rate(),
            "total_pnl": self.portfolio.get_total_pnl(),
            "total_return": self.portfolio.get_total_return(),
            "average_trade_pnl": self.portfolio.get_total_pnl() / len(closed_trades),
            "average_winning_trade": sum(t.pnl for t in closed_trades if t.pnl > 0) / max(1, sum(1 for t in closed_trades if t.pnl > 0)),
            "average_losing_trade": sum(t.pnl for t in closed_trades if t.pnl <= 0) / max(1, sum(1 for t in closed_trades if t.pnl <= 0)),
            "profit_factor": abs(sum(t.pnl for t in closed_trades if t.pnl > 0) / sum(t.pnl for t in closed_trades if t.pnl <= 0)) if sum(t.pnl for t in closed_trades if t.pnl <= 0) != 0 else float("inf"),
            "sharpe_ratio": self._calculate_sharpe_ratio(equity_curve),
            "sortino_ratio": self._calculate_sortino_ratio(equity_curve),
            "max_drawdown": self._calculate_max_drawdown(equity_curve),
            "max_drawdown_duration": self._calculate_max_drawdown_duration(equity_curve),
            "average_drawdown": self._calculate_average_drawdown(equity_curve),
            "final_equity": self.portfolio.current_capital,
            "equity_curve": equity_curve.to_dict(),
            "trades": [t.to_dict() for t in closed_trades],
            "summary": {
                "start_date": str(closed_trades[0].entry_date) if closed_trades else None,
                "end_date": str(closed_trades[-1].exit_date) if closed_trades else None,
                "trading_days": len(equity_curve),
            }
        }

        return results

    def _calculate_equity_curve(self, closed_trades: List[Trade]) -> pd.DataFrame:
        """计算权益曲线"""
        if self.market_data is None:
            return pd.DataFrame()

        dates = self.market_data["date"].tolist()
        equity = []

        for date in dates:
            portfolio_value = self.portfolio.current_capital

            # 加上未平仓头寸的浮动盈亏
            for position in self.portfolio.get_open_positions():
                current_price = self._get_price(date, position.symbol)
                if current_price:
                    if position.side == "BUY":
                        portfolio_value += (current_price - position.entry_price) * position.quantity
                    else:
                        portfolio_value += (position.entry_price - current_price) * position.quantity

            equity.append(portfolio_value)

        return pd.DataFrame({"date": dates, "equity": equity})

    def _calculate_sharpe_ratio(self, equity_curve: pd.DataFrame) -> float:
        """计算夏普比率"""
        if len(equity_curve) < 2:
            return 0.0

        returns = equity_curve["equity"].pct_change()
        if returns.std() == 0:
            return 0.0

        return returns.mean() / returns.std() * np.sqrt(252)

    def _calculate_sortino_ratio(self, equity_curve: pd.DataFrame) -> float:
        """计算索提诺比率"""
        if len(equity_curve) < 2:
            return 0.0

        returns = equity_curve["equity"].pct_change()
        downside_returns = returns[returns < 0]

        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0

        return returns.mean() / downside_returns.std() * np.sqrt(252)

    def _calculate_max_drawdown(self, equity_curve: pd.DataFrame) -> float:
        """计算最大回撤"""
        if len(equity_curve) == 0:
            return 0.0

        cumulative_max = equity_curve["equity"].cummax()
        drawdown = (equity_curve["equity"] - cumulative_max) / cumulative_max * 100

        return drawdown.min()

    def _calculate_max_drawdown_duration(self, equity_curve: pd.DataFrame) -> int:
        """计算最大回撤持续时间"""
        if len(equity_curve) == 0:
            return 0

        cumulative_max = equity_curve["equity"].cummax()
        in_drawdown = equity_curve["equity"] < cumulative_max

        max_duration = 0
        current_duration = 0

        for in_dd in in_drawdown:
            if in_dd:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0

        return max_duration

    def _calculate_average_drawdown(self, equity_curve: pd.DataFrame) -> float:
        """计算平均回撤"""
        if len(equity_curve) == 0:
            return 0.0

        cumulative_max = equity_curve["equity"].cummax()
        drawdown = (equity_curve["equity"] - cumulative_max) / cumulative_max * 100

        drawdown_values = drawdown[drawdown < 0]
        return drawdown_values.mean() if len(drawdown_values) > 0 else 0.0

    def save_results(self, filepath: str):
        """保存回测结果"""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)

    def generate_report(self) -> str:
        """生成回测报告"""
        if not self.results:
            return "No results to report"

        report = [
            "=" * 50,
            "回测报告",
            "=" * 50,
            "",
            f"总交易次数：{self.results['total_trades']}",
            f"胜率：{self.results['win_rate']:.2f}%",
            f"总盈亏：{self.results['total_pnl']:.2f}",
            f"总收益率：{self.results['total_return']:.2f}%",
            f"平均盈亏：{self.results['average_trade_pnl']:.2f}",
            "",
            f"夏普比率：{self.results['sharpe_ratio']:.4f}",
            f"索提诺比率：{self.results['sortino_ratio']:.4f}",
            f"最大回撤：{self.results['max_drawdown']:.2f}%",
            f"最大回撤持续天数：{self.results['max_drawdown_duration']}天",
            f"平均回撤：{self.results['average_drawdown']:.2f}%",
            "",
            f"盈利交易：{self.results['winning_trades']}",
            f"亏损交易：{self.results['losing_trades']}",
            f"平均盈利：{self.results['average_winning_trade']:.2f}",
            f"平均亏损：{self.results['average_losing_trade']:.2f}",
            f"盈亏比：{self.results['profit_factor']:.4f}",
            "",
            f"最终资金：{self.results['final_equity']:.2f}",
            "=" * 50,
        ]

        return "\n".join(report)

    def load_results(self, filepath: str):
        """加载回测结果"""
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                self.results = json.load(f)
