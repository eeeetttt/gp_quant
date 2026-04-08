"""
股票数据获取模块
支持多数据源：AkShare、本地文件等
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import pandas as pd
from datetime import datetime


class DataFetcher(ABC):
    """数据获取基类"""

    @abstractmethod
    def fetch(self, symbol: str, start_date: str, end_date: str,
              indicators: Optional[list] = None) -> pd.DataFrame:
        """获取数据"""
        pass

    @abstractmethod
    def fetch_quote(self, symbol: str) -> Dict[str, Any]:
        """获取实时报价"""
        pass


class AkShareFetcher(DataFetcher):
    """AkShare 数据获取器 - 专用于 A 股"""

    def __init__(self):
        import akshare as ak
        self.ak = ak

    def fetch(self, symbol: str, start_date: str, end_date: str,
              indicators: Optional[list] = None) -> pd.DataFrame:
        """
        获取 A 股历史行情数据

        Args:
            symbol: 股票代码 (如 000001.SZ, 600519.SH)
            start_date: 开始日期 "YYYYMMDD"
            end_date: 结束日期 "YYYYMMDD"
            indicators: 技术指标列表

        Returns:
            包含 OHLCV 数据的 DataFrame
        """
        # 格式化日期
        start_date_fmt = start_date.replace("-", "")
        end_date_fmt = end_date.replace("-", "")

        # 获取数据
        df = self.ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start_date_fmt,
            end_date=end_date_fmt,
            adjust="qfq"  # 前复权
        )

        if df.empty:
            raise ValueError(f"No data found for {symbol}")

        # 重命名列
        df = df.rename(columns={
            "日期": "date",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交量": "volume",
            "成交额": "amount",
            "振幅": "range",
            "涨跌幅": "change_pct",
            "涨跌额": "change",
            "换手": "turnover"
        })

        # 转换日期格式
        df["date"] = pd.to_datetime(df["date"])

        # 添加技术指标
        if indicators:
            from ..strategy.indicators import TechnicalIndicators
            indicator_calculator = TechnicalIndicators(df)
            for ind in indicators:
                df = indicator_calculator.add_indicator(ind)

        return df.reset_index(drop=True)

    def fetch_quote(self, symbol: str) -> Dict[str, Any]:
        """获取实时报价"""
        try:
            df = self.ak.stock_zh_a_spot_em()
            stock_info = df[df["代码"] == symbol]

            if len(stock_info) > 0:
                info = stock_info.iloc[0]
                return {
                    "symbol": symbol,
                    "name": info.get("名称", ""),
                    "current_price": float(info.get("最新价", 0)),
                    "previous_close": float(info.get("昨收", 0)),
                    "day_high": float(info.get("最高", 0)),
                    "day_low": float(info.get("最低", 0)),
                    "volume": int(info.get("成交量", 0)),
                    "amount": float(info.get("成交额", 0)),
                    "change_pct": float(info.get("涨跌幅", 0)),
                    "turnover": float(info.get("换手", 0)),
                    "timestamp": datetime.now().isoformat()
                }
        except Exception:
            pass

        return {"error": "Unable to fetch real-time quote"}


class LocalDataFetcher(DataFetcher):
    """本地 CSV 数据获取器"""

    def __init__(self, data_dir: str = "./data"):
        self.data_dir = data_dir

    def fetch(self, symbol: str, start_date: str, end_date: str,
              indicators: Optional[list] = None) -> pd.DataFrame:
        """从 CSV 文件读取数据"""
        import os
        file_path = os.path.join(self.data_dir, f"{symbol}.csv")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")

        df = pd.read_csv(file_path, parse_dates=["date"])
        df = df.set_index("date")

        # 过滤日期范围
        df = df[(df.index >= start_date) & (df.index <= end_date)]

        if indicators:
            from ..strategy.indicators import TechnicalIndicators
            indicator_calculator = TechnicalIndicators(df)
            for ind in indicators:
                df = indicator_calculator.add_indicator(ind)

        return df.reset_index()

    def fetch_quote(self, symbol: str) -> Dict[str, Any]:
        return {"error": "Local data fetcher doesn't support real-time quotes"}


def create_fetcher(source: str = "akshare", **kwargs) -> DataFetcher:
    """创建数据获取器"""
    if source == "akshare":
        return AkShareFetcher(**kwargs)
    elif source == "local":
        return LocalDataFetcher(**kwargs)
    else:
        raise ValueError(f"Unknown data source: {source}")
