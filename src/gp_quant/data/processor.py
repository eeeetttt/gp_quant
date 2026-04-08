"""
数据处理模块
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod


class DataProcessor(ABC):
    """数据处理器基类"""

    @abstractmethod
    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        """数据清洗"""
        pass

    @abstractmethod
    def normalize(self, data: pd.DataFrame, method: str = "minmax") -> pd.DataFrame:
        """数据标准化"""
        pass


class StockDataProcessor(DataProcessor):
    """股票数据处理器"""

    def __init__(self):
        self.required_columns = {"date", "open", "high", "low", "close", "volume"}

    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        数据清洗

        Args:
            data: 原始数据 DataFrame

        Returns:
            清洗后的数据
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input must be a DataFrame")

        # 复制数据避免修改原始数据
        df = data.copy()

        # 处理缺失值
        df = df.dropna(subset=["close"])

        # 删除重复日期
        if "date" in df.columns:
            df = df.drop_duplicates(subset=["date"])
            df = df.sort_values("date")
            df.index = pd.to_datetime(df["date"])

        # 计算变动幅度
        df["change"] = df["close"].pct_change() * 100
        df["change_abs"] = df["close"] - df["open"]

        # 计算波动率
        df["range"] = df["high"] - df["low"]
        df["range_pct"] = df["range"] / df["close"] * 100

        # 删除删除无效数据
        df = df[df["range_pct"] < 100]  # 排除极端异常值

        return df.reset_index()

    def normalize(self, data: pd.DataFrame, method: str = "minmax") -> pd.DataFrame:
        """
        数据标准化

        Args:
            data: 输入数据
            method: 标准化方法 ("minmax", "zscore", "robust")

        Returns:
            标准化后的数据
        """
        df = data.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if method == "minmax":
            for col in numeric_cols:
                if col not in ["date"]:
                    min_val = df[col].min()
                    max_val = df[col].max()
                    if max_val - min_val > 0:
                        df[col] = (df[col] - min_val) / (max_val - min_val)

        elif method == "zscore":
            for col in numeric_cols:
                if col not in ["date"]:
                    mean_val = df[col].mean()
                    std_val = df[col].std()
                    if std_val > 0:
                        df[col] = (df[col] - mean_val) / std_val

        elif method == "robust":
            for col in numeric_cols:
                if col not in ["date"]:
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr = q3 - q1
                    if iqr > 0:
                        df[col] = (df[col] - q1) / iqr

        return df

    def calculate_returns(self, data: pd.DataFrame, periods: List[int] = [1, 5, 10, 20]) -> pd.DataFrame:
        """计算多周期收益率"""
        df = data.copy()

        for period in periods:
            col_name = f"return_{period}d"
            df[col_name] = df["close"].pct_change(periods=period) * 100

        return df

    def generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成特征

        Returns:
            包含特征工程的结果
        """
        df = data.copy()

        # 移动平均线
        df["ma5"] = df["close"].rolling(window=5).mean()
        df["ma10"] = df["close"].rolling(window=10).mean()
        df["ma20"] = df["close"].rolling(window=20).mean()
        df["ma60"] = df["close"].rolling(window=60).mean()

        # MACD
        exp1 = df["close"].ewm(span=12, adjust=False).mean()
        exp2 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = exp1 - exp2
        df["signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["histogram"] = df["macd"] - df["signal"]

        # RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))

        # 布林带
        df["bb_middle"] = df["close"].rolling(window=20).mean()
        bb_std = df["close"].rolling(window=20).std()
        df["bb_upper"] = df["bb_middle"] + (bb_std * 2)
        df["bb_lower"] = df["bb_middle"] - (bb_std * 2)
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]

        # 波动率
        df["volatility"] = df["change"].rolling(window=20).std()

        # 成交量相关
        df["vol_ma5"] = df["volume"].rolling(window=5).mean()
        df["vol_ratio"] = df["volume"] / df["vol_ma5"]

        # 去除 NaN 值
        df = df.dropna()

        return df


def create_processor() -> DataProcessor:
    """创建数据处理器"""
    return StockDataProcessor()
