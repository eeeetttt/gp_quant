"""
股票数据获取模块
支持多数据源：Yahoo Finance、Tushare Pro、本地文件等
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta


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


class YahooFinanceFetcher(DataFetcher):
    """Yahoo Finance 数据获取器"""

    def __init__(self, proxy: Optional[str] = None):
        self.proxy = proxy

    def fetch(self, symbol: str, start_date: str, end_date: str,
              indicators: Optional[list] = None) -> pd.DataFrame:
        """
        获取历史行情数据

        Args:
            symbol: 股票代码 (如 AAPL, 000001.SZ)
            start_date: 开始日期 "YYYY-MM-DD"
            end_date: 结束日期 "YYYY-MM-DD"
            indicators: 技术指标列表

        Returns:
            包含 OHLCV 数据的 DataFrame
        """
        ticker = yf.Ticker(symbol)

        # 获取历史数据
        df = ticker.history(start=start_date, end=end_date, proxy=self.proxy)

        if df.empty:
            raise ValueError(f"No data found for {symbol}")

        df.reset_index(inplace=True)
        df = df.rename(columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "Dividends": "dividends",
            "Stock Splits": "split"
        })

        # 添加技术指标
        if indicators:
            from ..strategy.indicators import TechnicalIndicators
            indicator_calculator = TechnicalIndicators(df)
            for ind in indicators:
                df = indicator_calculator.add_indicator(ind)

        return df

    def fetch_quote(self, symbol: str) -> Dict[str, Any]:
        """获取实时报价"""
        ticker = yf.Ticker(symbol)
        info = ticker.fast_info

        return {
            "symbol": symbol,
            "current_price": float(info.current_price),
            "previous_close": float(info.previous_close),
            "day_high": float(info.day_high),
            "day_low": float(info.day_low),
            "volume": int(info.volume),
            "avg_volume": int(info.avg_volume_10d),
            "market_cap": int(info.market_cap),
            "pe_ratio": float(info.trailing_pe),
            "fifty_two_week_high": float(info.fifty_two_week_high),
            "fifty_two_week_low": float(info.fifty_two_week_low),
            "timestamp": datetime.now().isoformat()
        }


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


def create_fetcher(source: str = "yfinance", **kwargs) -> DataFetcher:
    """创建数据获取器"""
    if source == "yfinance":
        return YahooFinanceFetcher(**kwargs)
    elif source == "local":
        return LocalDataFetcher(**kwargs)
    else:
        raise ValueError(f"Unknown data source: {source}")
