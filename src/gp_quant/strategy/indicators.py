"""
技术指标计算模块
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional


class TechnicalIndicators:
    """技术指标计算器"""

    def __init__(self, data: pd.DataFrame):
        """
        初始化技术指标计算器

        Args:
            data: 包含 OHLCV 的 DataFrame，索引为日期
        """
        self.data = data.copy()
        self._calculate_basic_indicators()

    def _calculate_basic_indicators(self):
        """计算基础指标"""
        df = self.data

        # 基础价格指标
        if "change" not in df.columns:
            df["change"] = df["close"].pct_change() * 100
        if "change_abs" not in df.columns:
            df["change_abs"] = df["close"] - df["open"]

        # 波动率
        if "range" not in df.columns:
            df["range"] = df["high"] - df["low"]
        if "range_pct" not in df.columns:
            df["range_pct"] = df["range"] / df["close"] * 100

    def add_indicator(self, indicator: str) -> pd.DataFrame:
        """
        添加技术指标

        Args:
            indicator: 指标名称

        Returns:
            包含指标的 DataFrame
        """
        indicator_map = {
            "ma5": self._calculate_ma5,
            "ma10": self._calculate_ma10,
            "ma20": self._calculate_ma20,
            "ma60": self._calculate_ma60,
            "macd": self._calculate_macd,
            "rsi": self._calculate_rsi,
            "bollinger": self._calculate_bollinger,
            "volatility": self._calculate_volatility,
            "volume_ratio": self._calculate_volume_ratio,
            "atr": self._calculate_atr,
            "adx": self._calculate_adx,
            "stochastic": self._calculate_stochastic,
            "cci": self._calculate_cci,
            "wr": self._calculate_wr,
            "obv": self._calculate_obv,
            "mfi": self._calculate_mfi,
        }

        if indicator not in indicator_map:
            raise ValueError(f"Unknown indicator: {indicator}")

        return indicator_map[indicator]()

    # ========== 移动平均线 ==========

    def _calculate_ma5(self) -> pd.DataFrame:
        """5 日移动平均"""
        self.data["ma5"] = self.data["close"].rolling(window=5).mean()
        return self.data

    def _calculate_ma10(self) -> pd.DataFrame:
        """10 日移动平均"""
        self.data["ma10"] = self.data["close"].rolling(window=10).mean()
        return self.data

    def _calculate_ma20(self) -> pd.DataFrame:
        """20 日移动平均"""
        self.data["ma20"] = self.data["close"].rolling(window=20).mean()
        return self.data

    def _calculate_ma60(self) -> pd.DataFrame:
        """60 日移动平均"""
        self.data["ma60"] = self.data["close"].rolling(window=60).mean()
        return self.data

    def get_ma(self, period: int = 20) -> pd.Series:
        """获取指定周期的移动平均"""
        return self.data["close"].rolling(window=period).mean()

    # ========== MACD ==========

    def _calculate_macd(self) -> pd.DataFrame:
        """
        MACD 指标

        Returns:
            DataFrame with macd, signal, histogram columns
        """
        df = self.data

        # EMA 计算
        exp1 = df["close"].ewm(span=12, adjust=False).mean()
        exp2 = df["close"].ewm(span=26, adjust=False).mean()

        # MACD 线
        df["macd"] = exp1 - exp2

        # 信号线
        df["signal"] = df["macd"].ewm(span=9, adjust=False).mean()

        # 柱状图
        df["histogram"] = df["macd"] - df["signal"]

        return df

    def get_macd_signal(self) -> str:
        """获取 MACD 交易信号"""
        if len(self.data) < 30:
            return "HOLD"

        current = self.data.iloc[-1]
        previous = self.data.iloc[-2]

        # 金叉
        if previous["macd"] <= previous["signal"] and current["macd"] > current["signal"]:
            return "BUY"

        # 死叉
        if previous["macd"] >= previous["signal"] and current["macd"] < current["signal"]:
            return "SELL"

        return "HOLD"

    # ========== RSI ==========

    def _calculate_rsi(self) -> pd.DataFrame:
        """
        RSI 指标

        Returns:
            DataFrame with rsi column
        """
        df = self.data

        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()

        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))

        return df

    def get_rsi_signal(self, period: int = 14, oversold: float = 30, overbought: float = 70) -> str:
        """获取 RSI 交易信号"""
        if "rsi" not in self.data.columns:
            self._calculate_rsi()

        rsi = self.data["rsi"].iloc[-1]

        if rsi > overbought:
            return "SELL"  # 超买
        elif rsi < oversold:
            return "BUY"  # 超卖

        return "HOLD"

    # ========== 布林带 ==========

    def _calculate_bollinger(self) -> pd.DataFrame:
        """
        布林带指标

        Returns:
            DataFrame with bb_upper, bb_middle, bb_lower, bb_width columns
        """
        df = self.data

        # 中轨
        df["bb_middle"] = df["close"].rolling(window=20).mean()

        # 标准差
        bb_std = df["close"].rolling(window=20).std()

        # 上下轨
        df["bb_upper"] = df["bb_middle"] + (bb_std * 2)
        df["bb_lower"] = df["bb_middle"] - (bb_std * 2)

        # 带宽
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]

        # 股价位置 (0-1 之间)
        df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

        return df

    def get_bollinger_signal(self) -> str:
        """获取布林带交易信号"""
        if "bb_upper" not in self.data.columns:
            self._calculate_bollinger()

        df = self.data
        current = df.iloc[-1]

        if current["bb_position"] < 0.1:
            return "BUY"  # 触及下轨
        elif current["bb_position"] > 0.9:
            return "SELL"  # 触及上轨

        return "HOLD"

    # ========== 波动率 ==========

    def _calculate_volatility(self) -> pd.DataFrame:
        """
        波动率计算

        Returns:
            DataFrame with volatility column
        """
        df = self.data

        # 20 日波动率
        df["volatility"] = df["change"].rolling(window=20).std()

        # 年化波动率
        df["volatility_annual"] = df["volatility"] * np.sqrt(252)

        return df

    # ========== 成交量相关 ==========

    def _calculate_volume_ratio(self) -> pd.DataFrame:
        """
        成交量比率

        Returns:
            DataFrame with vol_ratio column
        """
        df = self.data

        df["vol_ma5"] = df["volume"].rolling(window=5).mean()
        df["vol_ma20"] = df["volume"].rolling(window=20).mean()
        df["vol_ratio"] = df["volume"] / df["vol_ma5"]

        return df

    def get_volume_signal(self) -> str:
        """获取成交量信号"""
        if "vol_ratio" not in self.data.columns:
            self._calculate_volume_ratio()

        vol_ratio = self.data["vol_ratio"].iloc[-1]

        if vol_ratio > 2.0:
            return "HIGH_VOLUME"  # 放量
        elif vol_ratio < 0.5:
            return "LOW_VOLUME"  # 缩量

        return "NORMAL_VOLUME"

    # ========== ATR 止损指标 ==========

    def _calculate_atr(self) -> pd.DataFrame:
        """
        ATR (Average True Range) 指标

        Returns:
            DataFrame with atr column
        """
        df = self.data

        high_low = df["high"] - df["low"]
        high_close = abs(df["high"] - df["close"].shift())
        low_close = abs(df["low"] - df["close"].shift())

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        df["atr"] = tr.rolling(window=14).mean()
        df["atr_ratio"] = df["atr"] / df["close"] * 100

        return df

    def get_atr_stop_loss(self, multiplier: float = 2.0) -> float:
        """获取基于 ATR 的止损价"""
        if "atr" not in self.data.columns:
            self._calculate_atr()

        atr = self.data["atr"].iloc[-1]
        close = self.data["close"].iloc[-1]

        return close - (atr * multiplier)

    # ========== ADX 趋势强度 ==========

    def _calculate_adx(self) -> pd.DataFrame:
        """
        ADX 指标

        Returns:
            DataFrame with adx, +di, -di columns
        """
        df = self.data

        high = df["high"]
        low = df["low"]
        close = df["close"]

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = pd.Series(0, index=df.index)
        minus_dm = pd.Series(0, index=df.index)

        for i in range(1, len(df)):
            if up_move[i] > down_move[i] and up_move[i] > 0:
                plus_dm[i] = up_move[i]
            if down_move[i] > up_move[i] and down_move[i] > 0:
                minus_dm[i] = down_move[i]

        # EMA of TR, +DM, -DM
        atr_14 = tr.ewm(span=14).mean()
        plus_di = 100 * (plus_dm.ewm(span=14).mean() / atr_14)
        minus_di = 100 * (minus_dm.ewm(span=14).mean() / atr_14)

        # DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(span=14).mean()

        df["adx"] = adx
        df["+di"] = plus_di
        df["-di"] = minus_di

        return df

    def get_adx_signal(self) -> str:
        """获取 ADX 趋势信号"""
        if "adx" not in self.data.columns:
            self._calculate_adx()

        adx = self.data["adx"].iloc[-1]

        if adx > 25:
            return "TRENDING"  # 有趋势
        else:
            return "RANGING"  # 震荡

    # ========== 随机指标 ==========

    def _calculate_stochastic(self) -> pd.DataFrame:
        """
        随机指标 (Stochastic Oscillator)

        Returns:
            DataFrame with k, d columns
        """
        df = self.data

        low_14 = df["low"].rolling(window=14).min()
        high_14 = df["high"].rolling(window=14).max()

        df["k"] = 100 * (df["close"] - low_14) / (high_14 - low_14)
        df["d"] = df["k"].rolling(window=3).mean()

        return df

    def get_stochastic_signal(self) -> str:
        """获取随机指标信号"""
        if "k" not in self.data.columns:
            self._calculate_stochastic()

        k = self.data["k"].iloc[-1]

        if k < 20:
            return "OVERSOLD"
        elif k > 80:
            return "OVERBOUGHT"

        return "NEUTRAL"

    # ========== CCI 顺势指标 ==========

    def _calculate_cci(self) -> pd.DataFrame:
        """
        CCI 指标

        Returns:
            DataFrame with cci column
        """
        df = self.data

        tp = (df["high"] + df["low"] + df["close"]) / 3
        ma = tp.rolling(window=20).mean()
        mad = tp.rolling(window=20).std()

        df["cci"] = (tp - ma) / (0.015 * mad)

        return df

    def get_cci_signal(self) -> str:
        """获取 CCI 信号"""
        if "cci" not in self.data.columns:
            self._calculate_cci()

        cci = self.data["cci"].iloc[-1]

        if cci > 100:
            return "OVERBOUGHT"
        elif cci < -100:
            return "OVERSOLD"

        return "NEUTRAL"

    # ========== WR 威廉指标 ==========

    def _calculate_wr(self) -> pd.DataFrame:
        """
        威廉指标

        Returns:
            DataFrame with wr column
        """
        df = self.data

        high_14 = df["high"].rolling(window=14).max()
        low_14 = df["low"].rolling(window=14).min()

        df["wr"] = (high_14 - df["close"]) / (high_14 - low_14) * -100

        return df

    def get_wr_signal(self) -> str:
        """获取 WR 信号"""
        if "wr" not in self.data.columns:
            self._calculate_wr()

        wr = self.data["wr"].iloc[-1]

        if wr > 80:
            return "OVERSOLD"
        elif wr < 20:
            return "OVERBOUGHT"

        return "NEUTRAL"

    # ========== OBV 能量潮 ==========

    def _calculate_obv(self) -> pd.DataFrame:
        """
        OBV 指标

        Returns:
            DataFrame with obv column
        """
        df = self.data

        condition = df["close"] > df["close"].shift(1)
        df["obv"] = np.where(condition, df["volume"],
                             np.where(df["close"] < df["close"].shift(1), -df["volume"], 0)).cumsum()

        return df

    def get_obv_signal(self) -> str:
        """获取 OBV 信号"""
        if "obv" not in self.data.columns:
            self._calculate_obv()

        obv = self.data["obv"].rolling(window=20).mean().iloc[-1]
        price_trend = self.data["close"].pct_change(20).iloc[-1]

        if price_trend > 0 and obv > 0:
            return "BULLISH"
        elif price_trend < 0 and obv < 0:
            return "BEARISH"

        return "NEUTRAL"

    # ========== MFI 资金流量 ==========

    def _calculate_mfi(self) -> pd.DataFrame:
        """
        MFI 指标

        Returns:
            DataFrame with mfi column
        """
        df = self.data

        tp = (df["high"] + df["low"] + df["close"]) / 3
        raw_money_flow = tp * df["volume"]

        condition = raw_money_flow > raw_money_flow.shift(1)
        positive_flow = np.where(condition, raw_money_flow, 0)
        negative_flow = np.where(~condition, raw_money_flow, 0)

        mfi_14 = positive_flow.rolling(window=14).sum() / negative_flow.rolling(window=14).sum()
        df["mfi"] = 100 - (100 / (1 + mfi_14))

        return df

    def get_mfi_signal(self) -> str:
        """获取 MFI 信号"""
        if "mfi" not in self.data.columns:
            self._calculate_mfi()

        mfi = self.data["mfi"].iloc[-1]

        if mfi > 80:
            return "OVERBOUGHT"
        elif mfi < 20:
            return "OVERSOLD"

        return "NEUTRAL"

    # ========== 综合信号 ==========

    def get_comprehensive_signal(self) -> Dict[str, Any]:
        """
        获取综合交易信号

        Returns:
            包含各指标信号的综合字典
        """
        # 确保计算所有指标
        self._calculate_rsi()
        self._calculate_macd()
        self._calculate_bollinger()

        signals = {
            "rsi": self.get_rsi_signal(),
            "macd": self.get_macd_signal(),
            "bollinger": self.get_bollinger_signal(),
            "volume": self.get_volume_signal(),
            "trend": self.get_adx_signal() if "adx" in self.data.columns else "NEUTRAL",
        }

        # 综合判断
        buy_count = sum(1 for s in signals.values() if s in ["BUY", "OVERSOLD", "BULLISH"])
        sell_count = sum(1 for s in signals.values() if s in ["SELL", "OVERBOUGHT", "BEARISH"])

        if buy_count >= 3:
            overall = "BUY"
        elif sell_count >= 3:
            overall = "SELL"
        else:
            overall = "HOLD"

        return {
            "overall": overall,
            "detailed_signals": signals,
            "buy_signals": buy_count,
            "sell_signals": sell_count,
        }

    def get_all_indicators(self) -> pd.DataFrame:
        """计算并返回所有技术指标"""
        self._calculate_rsi()
        self._calculate_macd()
        self._calculate_bollinger()
        self._calculate_atr()
        self._calculate_adx()
        self._calculate_stochastic()
        self._calculate_cci()
        self._calculate_wr()
        self._calculate_obv()
        self._calculate_mfi()
        self._calculate_volume_ratio()
        self._calculate_volatility()

        return self.data.dropna()
