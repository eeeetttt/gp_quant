"""
Phase 4: 特征工程

在现有 63 个特征基础上，新增：
- 大盘指标（上证指数走势/技术指标）
- 相对强弱（个股 vs 大盘超额收益/alpha/beta）
- 资金流（量价配合/换手率/资金净流入）
- 时间特征（月份/星期/月初月末）
- 价格动量（N日高低点位置/趋势强度）
- 截面排名（个股在全市场中的相对位置）
- 截面分位数目标（top 30% vs bottom 30%）
- 特征去冗余（剔除高度相关的重复特征）

输入：experiments/data/indicators_pool.csv
输出：experiments/ml/features.csv（合并版）
     experiments/ml/feature_report.json（特征报告）
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import logging
import json
import argparse

import pandas as pd
import numpy as np
import baostock as bs
import matplotlib
matplotlib.use('Agg')  # 无头渲染
import matplotlib.pyplot as plt
import seaborn as sns

from gp_quant.ml.features import FeatureEngineer, FeatureConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="特征工程")
parser.add_argument("--pool", choices=["small", "500", "full"], default="small")
args = parser.parse_args()

from pipeline.utils import get_pool_input_path

# Note: --pool accepted for CLI consistency. Input is always indicators_pool.csv (from step3).

INDICATORS_PATH = os.path.join(os.path.dirname(__file__), "data", "indicators_pool.csv")
ML_DIR = os.path.join(os.path.dirname(__file__), "ml")
FEATURES_PATH = os.path.join(ML_DIR, "features.csv")
REPORT_PATH = os.path.join(ML_DIR, "feature_report.json")

CORR_REPORT_PATH = os.path.join(ML_DIR, "correlation_report.json")
CORR_HEATMAP_PATH = os.path.join(ML_DIR, "feature_correlation.png")

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
        col_name = f'market_{dcol}' if dcol == 'close' else dcol
        df[col_name] = df['date'].map(map_dict)

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


def add_cross_sectional_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """
    截面排名：每日计算每只股票在全市场中的百分位排名。
    新增 5 个特征，值域 [0, 100]。
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    rank_columns = {
        'rank_rsi': 'rsi',
        'rank_return_5d': 'return_5d',
        'rank_vol_ratio': 'vol_ratio',
        'rank_turnover': 'turnover_ratio',
        'rank_momentum': 'trend_strength_20d',
    }

    for rank_col, src_col in rank_columns.items():
        if src_col not in df.columns:
            logger.warning("  缺少列 %s，跳过截面排名 %s", src_col, rank_col)
            continue
        df[rank_col] = df.groupby('date')[src_col].rank(pct=True) * 100

    return df


def add_quantile_target(df: pd.DataFrame, horizon: int = 5, top_pct: float = 0.30) -> pd.DataFrame:
    """
    截面分位数目标：
    - 每天 top_pct 的股票 → 1（买入信号）
    - 每天 bottom_pct 的股票 → 0（卖出信号）
    - 中间部分 → -1（中性，训练时排除）
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    # 未来 N 日收益率
    df[f'future_{horizon}d_ret'] = (
        df.groupby('symbol')['close'].transform(lambda x: x.shift(-horizon))
        - df['close']
    ) / df['close'] * 100

    def label_quantile(x):
        try:
            return pd.qcut(x, q=[0, top_pct, 1.0 - top_pct, 1.0], labels=[0, -1, 1]).astype(float)
        except (ValueError, KeyError):
            # 样本太少或值相同，无法分位
            return pd.Series(-1, index=x.index, dtype=float)

    df[f'target_quantile_{horizon}d'] = df.groupby('date')[f'future_{horizon}d_ret'].transform(label_quantile)

    return df


def analyze_feature_correlation(df, feature_cols, target_col="target_5d",
                                 corr_threshold=0.85, max_samples=100000):
    """
    特征相关性分析：
    1. 特征-特征相关矩阵（Spearman，捕获非线性关系）
    2. 高相关特征对（|corr| > threshold），建议剔除
    3. 特征-目标相关性（找出最有预测力的特征）
    4. 热力图可视化
    5. 报告 JSON
    """
    logger.info("开始特征相关性分析...")

    # 采样加速（1M+ 行算相关性太慢）
    if len(df) > max_samples:
        sample_df = df.sample(n=max_samples, random_state=42).copy()
        logger.info("  数据采样: %d → %d 行", len(df), len(sample_df))
    else:
        sample_df = df.copy()

    # 只取数值特征
    numeric_cols = [c for c in feature_cols if c in sample_df.columns
                    and sample_df[c].dtype in ['float64', 'float32', 'int64', 'int32']
                    and sample_df[c].nunique() > 2]

    # 剔除目标列和中间计算列
    exclude = {'target_5d', 'target_direction_5d', 'target_quantile_5d',
               'future_5d_ret', 'vol_up', 'vol_down',
               'is_month_start', 'is_month_end'}
    numeric_cols = [c for c in numeric_cols if c not in exclude]

    # 去 NaN
    feat_df = sample_df[numeric_cols].dropna()
    if len(feat_df) < 100:
        logger.warning("  有效样本不足，跳过相关性分析")
        return

    logger.info("  特征数: %d, 有效样本: %d", len(numeric_cols), len(feat_df))

    # 1. 特征-特征相关矩阵
    corr_matrix = feat_df.corr(method='spearman')

    # 2. 高相关特征对
    high_corr_pairs = []
    seen = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            c1, c2 = corr_matrix.columns[i], corr_matrix.columns[j]
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > corr_threshold and (c1, c2) not in seen:
                seen.add((c1, c2))
                seen.add((c2, c1))
                high_corr_pairs.append({
                    "feature1": c1,
                    "feature2": c2,
                    "correlation": round(float(corr_val), 4),
                })
    high_corr_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)

    logger.info("  高相关特征对 (|corr| > %.2f): %d 对", corr_threshold, len(high_corr_pairs))

    # 3. 特征-目标相关性
    feat_target_corr = {}
    top_pos = []
    top_neg = []
    if target_col in sample_df.columns:
        target_series = sample_df[target_col].dropna()
        # 取共同索引
        common_idx = feat_df.index.intersection(target_series.index)
        if len(common_idx) > 100:
            for col in numeric_cols:
                c = feat_df.loc[common_idx, col].corr(target_series.loc[common_idx], method='spearman')
                feat_target_corr[col] = round(float(c), 4)

            feat_target_sorted = sorted(feat_target_corr.items(), key=lambda x: abs(x[1]), reverse=True)
            top_pos = [(c, v) for c, v in feat_target_sorted if v > 0][:15]
            top_neg = [(c, v) for c, v in feat_target_sorted if v < 0][:15]
    else:
        feat_target_corr = {}
        top_pos = []
        top_neg = []

    logger.info("  最强正向特征 (Top 5):")
    for c, v in top_pos[:5]:
        logger.info("    %-35s corr=%.4f", c, v)
    logger.info("  最强负向特征 (Top 5):")
    for c, v in top_neg[:5]:
        logger.info("    %-35s corr=%.4f", c, v)

    # 4. 热力图（只取 Top 30 最强相关的特征，否则图太密）
    try:
        top_n = min(30, len(numeric_cols))
        all_sorted = sorted(feat_target_corr.items(), key=lambda x: abs(x[1]), reverse=True) if feat_target_corr else []
        top_features = [c for c, v in all_sorted[:top_n]]
        if len(top_features) < top_n:
            # 如果相关特征不足，用前面的补充
            remaining = [c for c in numeric_cols[:top_n] if c not in top_features]
            top_features.extend(remaining)

        plot_df = feat_df[top_features].copy()
        if target_col in sample_df.columns:
            target_for_plot = sample_df[target_col].dropna()
            common = plot_df.index.intersection(target_for_plot.index)
            if len(common) > 100:
                plot_df[target_col] = target_for_plot.loc[common]

        corr_sub = plot_df.corr(method='spearman')

        plt.figure(figsize=(16, 14))
        sns.heatmap(corr_sub, cmap='RdBu_r', center=0, annot=False,
                    square=True, linewidths=0.5, cbar_kws={'shrink': 0.8})
        plt.title('Feature Correlation Matrix (Spearman) — Top 30 vs Target', fontsize=14, pad=15)
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        plt.savefig(CORR_HEATMAP_PATH, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("  热力图已保存: %s", CORR_HEATMAP_PATH)
    except Exception as e:
        logger.warning("  热力图生成失败: %s", e)

    # 5. 保存报告
    corr_report = {
        "threshold": corr_threshold,
        "n_features_analyzed": len(numeric_cols),
        "n_samples_used": len(feat_df),
        "high_correlation_pairs": high_corr_pairs[:50],  # 只保留前 50 对
        "n_high_corr_pairs_total": len(high_corr_pairs),
        "feature_target_correlation": dict(feat_target_corr),
        "top_positive_features": [{"feature": c, "correlation": v} for c, v in top_pos],
        "top_negative_features": [{"feature": c, "correlation": v} for c, v in top_neg],
        "heatmap_path": CORR_HEATMAP_PATH,
    }

    with open(CORR_REPORT_PATH, 'w') as f:
        json.dump(corr_report, f, indent=2, ensure_ascii=False, default=str)
    logger.info("  报告已保存: %s", CORR_REPORT_PATH)


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

    # 截面排名特征（需要全市场数据）
    logger.info("计算截面排名特征...")
    merged = add_cross_sectional_ranks(merged)

    # 截面分位数目标
    logger.info("计算截面分位数目标...")
    merged = add_quantile_target(merged, horizon=5, top_pct=0.30)

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
            "截面排名": [c for c in feature_cols if c.startswith('rank_')],
        },
        "target_columns": [c for c in merged.columns if c.startswith('target')],
        "validation_warnings": all_warnings[:10],
        "missing_values": {k: v for k, v in merged.isnull().sum().items() if v > 0},
    }

    with open(REPORT_PATH, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    merged.to_csv(FEATURES_PATH, index=False)

    # 相关性分析
    analyze_feature_correlation(merged, feature_cols, target_col="target_5d")

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
