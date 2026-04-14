"""
Phase 4: 特征工程

在现有 63 个特征基础上，新增：
- 大盘指标（上证指数走势/技术指标）
- 相对强弱（个股 vs 大盘超额收益/alpha/beta）
- 资金流（量价配合/换手率/资金净流入）
- 时间特征（月份/星期/月初月末）
- 价格动量（N日高低点位置/趋势强度）
- 特征去冗余（剔除高度相关的重复特征）

输入：experiments/data/indicators_pool.csv
输出：experiments/ml/features.csv（合并版）
     experiments/ml/feature_report.json（特征报告）
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import logging
import json

import pandas as pd
import numpy as np
import baostock as bs

from gp_quant.ml.features import FeatureEngineer, FeatureConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

INDICATORS_PATH = os.path.join(os.path.dirname(__file__), "data", "indicators_pool.csv")
ML_DIR = os.path.join(os.path.dirname(__file__), "ml")
FEATURES_PATH = os.path.join(ML_DIR, "features.csv")
REPORT_PATH = os.path.join(ML_DIR, "feature_report.json")

INDEX_CODE = "sh.000001"  # 上证指数


def get_market_index(start_date: str, end_date: str) -> pd.DataFrame:
    """获取上证指数日线数据"""
    bs.login()
    rs = bs.query_history_k_data_plus(
        INDEX_CODE,
        "date,close,volume,amount",
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="2"
    )
    df = rs.get_data()
    for col in ['close', 'volume', 'amount']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['close'])
    bs.logout()
    return df


def add_market_features(df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
    """添加大盘相关特征"""
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    # 大盘收益率
    market_df = market_df.copy()
    market_df['date'] = pd.to_datetime(market_df['date'])
    market_df['market_return_1d'] = market_df['close'].pct_change() * 100
    market_df['market_return_5d'] = market_df['close'].pct_change(periods=5) * 100
    market_df['market_vol_change'] = market_df['volume'].pct_change() * 100

    for dcol in ['market_return_1d', 'market_return_5d', 'market_vol_change', 'close']:
        map_dict = dict(zip(market_df['date'], market_df[dcol]))
        df[dcol] = df['date'].map(map_dict)

    # 相对强弱（个股超额收益 vs 大盘）
    df['alpha_1d'] = df['return_1d'] - df['market_return_1d']
    df['alpha_5d'] = df['return_5d'] - df['market_return_5d']

    # 滚动 beta（20日窗口）
    for sym in df['symbol'].unique():
        mask = df['symbol'] == sym
        stock = df.loc[mask]
        if len(stock) < 25:
            continue
        rolling_beta = stock['return_1d'].rolling(20).cov(
            stock['market_return_1d']
        ) / stock['market_return_1d'].rolling(20).var()
        df.loc[mask, 'beta_20d'] = rolling_beta.values

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """添加时间特征"""
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
    df['quarter'] = df['date'].dt.quarter
    return df


def add_moneyflow_features(df: pd.DataFrame) -> pd.DataFrame:
    """添加资金流特征"""
    df = df.copy()

    # 量价配合：放量上涨 vs 缩量下跌
    df['vol_up'] = (df['return_1d'] > 0) & (df['volume'] > df['volume'].rolling(5).mean())
    df['vol_down'] = (df['return_1d'] < 0) & (df['volume'] > df['volume'].rolling(5).mean())
    df['moneyflow_ratio'] = df['vol_up'].astype(int) - df['vol_down'].astype(int)

    # 5日/20日资金流比率
    df['moneyflow_5d'] = df['moneyflow_ratio'].rolling(5).sum()
    df['moneyflow_20d'] = df['moneyflow_ratio'].rolling(20).sum()

    # 换手率代理（成交量 / 20日均量）
    df['turnover_ratio'] = df['volume'] / df['volume'].rolling(20).mean()

    return df


def add_price_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """添加价格动量特征"""
    df = df.copy()

    # N 日最高/最低价位置
    for window in [20, 60]:
        rolling_high = df['high'].rolling(window).max()
        rolling_low = df['low'].rolling(window).min()
        df[f'price_position_{window}d'] = (df['close'] - rolling_low) / (rolling_high - rolling_low + 1e-10)
        df[f'trend_strength_{window}d'] = df['close'].pct_change(window) * 100

    # 趋势强度（线性回归斜率）
    for window in [10, 20]:
        def linreg_slope(series):
            x = np.arange(len(series))
            if series.std() == 0:
                return 0
            slope = np.polyfit(x, series.values, 1)[0]
            return slope

        df[f'slope_{window}d'] = df['close'].rolling(window).apply(linreg_slope, raw=False)

    return df


def remove_redundant_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    剔除高度相关的冗余特征
    - change 和 return_1d 重复，保留 return_1d
    - 不直接丢弃（后续可能有其他用途），但在 ML 中不列
    """
    # 这里只做标记，不做删除，让后续 ML 阶段决定
    return df


def build_features():
    os.makedirs(ML_DIR, exist_ok=True)

    logger.info("读取指标数据: %s", INDICATORS_PATH)
    df = pd.read_csv(INDICATORS_PATH, dtype={'symbol': str})
    logger.info("原始数据: %d 行, %d 只股票", len(df), df['symbol'].nunique())

    # 获取大盘数据
    date_min = df['date'].min()
    date_max = df['date'].max()
    logger.info("获取上证指数数据 (%s ~ %s)...", date_min, date_max)
    market_df = get_market_index(date_min, date_max)
    logger.info("上证指数: %d 个交易日", len(market_df))

    # 逐只股票处理
    config = FeatureConfig(
        target_column="close",
        target_horizon=5,
        use_technical_indicators=True,
        use_price_features=True,
        use_lag_features=True,
        use_volume_features=True,
        scaling="standard",
        drop_na=True,
    )

    symbols = sorted(df['symbol'].unique())
    feature_frames = []
    all_warnings = []

    for i, sym in enumerate(symbols):
        stock_df = df[df['symbol'] == sym].copy()
        stock_df = stock_df.sort_values('date').reset_index(drop=True)

        # 基础特征
        engineer = FeatureEngineer(config)
        features = engineer.create_features(stock_df)

        if features.empty:
            logger.warning("[%s] 特征工程后无数据", sym)
            continue

        # 新增特征
        features = add_market_features(features, market_df)
        features = add_time_features(features)
        features = add_moneyflow_features(features)
        features = add_price_momentum_features(features)

        # 验证
        warnings = engineer.validate_features(features)
        if warnings:
            all_warnings.extend([f"[{sym}] {w}" for w in warnings])

        feature_frames.append(features)

        if (i + 1) % 20 == 0:
            logger.info("进度: %d/%d", i + 1, len(symbols))

    merged = pd.concat(feature_frames, ignore_index=True)

    # 报告
    feature_cols = [c for c in merged.columns if c not in ['symbol', 'date', 'close']]
    report = {
        "total_stocks": len(feature_frames),
        "total_samples": len(merged),
        "feature_columns": feature_cols,
        "feature_count": len(feature_cols),
        "feature_categories": {
            "基础指标": [c for c in feature_cols if c in ['open', 'high', 'low', 'volume', 'amount', 'change', 'change_abs', 'range', 'range_pct']],
            "多周期收益": [c for c in feature_cols if c.startswith('return_')],
            "技术指标": [c for c in feature_cols if c in ['rsi', 'macd', 'signal', 'histogram', 'bb_middle', 'bb_upper', 'bb_lower', 'bb_width', 'bb_position', 'atr', 'atr_ratio', 'adx', '+di', '-di', 'k', 'd', 'cci', 'wr', 'obv', 'mfi', 'vol_ma5', 'vol_ma20', 'vol_ratio', 'volatility', 'volatility_annual']],
            "价格特征": [c for c in feature_cols if 'price_' in c or 'gap' in c],
            "波动率": [c for c in feature_cols if 'volatility' in c],
            "滞后特征": [c for c in feature_cols if 'lag_' in c],
            "大盘特征": [c for c in feature_cols if c.startswith('market_')],
            "相对强弱": [c for c in feature_cols if c.startswith('alpha_') or c.startswith('beta_')],
            "时间特征": [c for c in feature_cols if c in ['month', 'day_of_week', 'is_month_start', 'is_month_end', 'quarter']],
            "资金流": [c for c in feature_cols if c.startswith('moneyflow_') or c.startswith('vol_up') or c.startswith('vol_down') or c == 'turnover_ratio'],
            "动量特征": [c for c in feature_cols if 'position_' in c or 'trend_' in c or 'slope_' in c],
        },
        "target_columns": [c for c in merged.columns if c.startswith('target')],
        "validation_warnings": all_warnings[:10],
        "missing_values": {k: v for k, v in merged.isnull().sum().items() if v > 0},
    }

    with open(REPORT_PATH, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    merged.to_csv(FEATURES_PATH, index=False)

    logger.info("=" * 50)
    logger.info("特征工程完成!")
    logger.info("输出: %s (%.1f MB)", FEATURES_PATH,
                 os.path.getsize(FEATURES_PATH) / 1024 / 1024)
    logger.info("总样本: %d", len(merged))
    logger.info("总特征数: %d", len(feature_cols))
    logger.info("目标列: %s", report['target_columns'])
    logger.info("特征分类:")
    for cat, cols in report['feature_categories'].items():
        logger.info("  %s: %d 个", cat, len(cols))


if __name__ == "__main__":
    build_features()
