"""
Sector/Industry Features (使用 baostock 行业分类)

板块特征对 A 股定价影响很大。基于 baostock 的证监会行业分类：
- query_stock_industry() 获取 stock -> industry 映射
- 计算行业板块日收益率（行业内等权平均）
- 新增特征：行业收益、行业动量、行业波动率、个股vs行业超额、行业内排名

输入：pipeline/data/indicators_pool.csv
输出：pipeline/ml/sector_features.csv（只含板块特征列，step5 merge 使用）
"""
import sys, os, logging, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
import baostock as bs

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

INDICATORS_PATH = os.path.join(os.path.dirname(__file__), "data", "indicators_pool.csv")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "ml", "sector_features.csv")


def get_sector_mapping():
    """
    获取股票-行业映射（baostock 证监会行业分类）
    返回 {symbol: industry} 字典
    """
    bs.login()
    rs = bs.query_stock_industry()
    if rs.error_code != '0':
        logger.error("获取行业分类失败: %s", rs.error_msg)
        bs.logout()
        return {}

    df = rs.get_data()
    # 转换代码格式：sh.600000 → 600000
    df['symbol'] = df['code'].str.replace(r'^sh\.|^sz\.', '', regex=True)
    mapping = dict(zip(df['symbol'], df['industry']))
    bs.logout()

    n_with_industry = sum(1 for v in mapping.values() if v)
    logger.info("行业分类: %d 只股票, %d 个行业", len(mapping),
                len(set(v for v in mapping.values() if v)))
    return mapping


def add_sector_features(df, sector_mapping):
    """
    添加板块相关特征：
    - industry: 所属行业（one-hot 编码用，但这里先存字符串）
    - sector_return_1d: 行业当日等权平均收益率
    - sector_return_5d: 行业5日收益率
    - sector_momentum_20d: 行业20日动量
    - sector_volatility_20d: 行业20日波动率
    - sector_alpha_1d: 个股收益 - 行业收益
    - sector_rank_in_industry: 个股在行业内的当日收益率排名百分位
    - sector_breadth_5d: 行业近5日上涨股票比例
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['industry'] = df['symbol'].map(sector_mapping)

    # 没有行业信息的标记为 NaN，后续会被 dropna 过滤
    no_sector = df['industry'].isna().sum()
    if no_sector > 0:
        logger.info("  %d 只股票无行业分类，后续会被过滤", no_sector)

    # 计算每个行业每日的等权平均收益率
    logger.info("  计算行业日收益率...")
    # 行业内所有股票 return_1d 的均值
    sector_daily = df.groupby(['date', 'industry'])['return_1d'].mean().reset_index()
    sector_daily.columns = ['date', 'industry', 'sector_return_1d']
    sector_daily['sector_return_5d'] = sector_daily.groupby('industry')['sector_return_1d'].transform(
        lambda x: x.rolling(5, min_periods=1).sum()
    )
    sector_daily['sector_momentum_20d'] = sector_daily.groupby('industry')['sector_return_1d'].transform(
        lambda x: x.rolling(20, min_periods=1).sum()
    )
    sector_daily['sector_volatility_20d'] = sector_daily.groupby('industry')['sector_return_1d'].transform(
        lambda x: x.rolling(20, min_periods=1).std()
    )
    # 行业近5日上涨股票比例（广度指标）
    daily_up = df[df['return_1d'] > 0].groupby(['date', 'industry']).size().reset_index(name='up_count')
    daily_total = df.groupby(['date', 'industry']).size().reset_index(name='total_count')
    breadth = daily_up.merge(daily_total, on=['date', 'industry'])
    breadth['sector_breadth_5d'] = (breadth['up_count'] / breadth['total_count']).rolling(5, min_periods=1).mean()
    breadth = breadth[['date', 'industry', 'sector_breadth_5d']]

    sector_daily = sector_daily.merge(breadth, on=['date', 'industry'], how='left')

    # merge 回原数据
    df = df.merge(sector_daily, on=['date', 'industry'], how='left')

    # 个股超额收益（vs 行业）
    df['sector_alpha_1d'] = df['return_1d'] - df['sector_return_1d']

    # 个股在行业内的当日收益率排名百分位
    logger.info("  计算行业内排名...")
    df['sector_rank_in_industry'] = df.groupby('date').apply(
        lambda g: g.groupby('industry')['return_1d'].transform(
            lambda x: x.rank(pct=True) * 100
        ),
        include_groups=False
    ).reset_index(level=0, drop=True)

    return df


def run_sector_features():
    logger.info("步骤 1: 获取行业分类...")
    sector_mapping = get_sector_mapping()
    if not sector_mapping:
        logger.error("无法获取行业分类")
        return

    logger.info("步骤 2: 读取指标数据...")
    df = pd.read_csv(INDICATORS_PATH, dtype={'symbol': str})
    logger.info("  原始数据: %d 行, %d 只股票", len(df), df['symbol'].nunique())

    logger.info("步骤 3: 计算板块特征...")
    df = add_sector_features(df, sector_mapping)

    # 保存板块特征（只含新增列）
    sector_cols = [
        'symbol', 'date', 'industry',
        'sector_return_1d', 'sector_return_5d', 'sector_momentum_20d',
        'sector_volatility_20d', 'sector_alpha_1d', 'sector_rank_in_industry',
        'sector_breadth_5d',
    ]
    df[sector_cols].to_csv(OUTPUT_PATH, index=False)
    logger.info("步骤 4: 板块特征已保存: %s (%.1f MB)",
                 OUTPUT_PATH, os.path.getsize(OUTPUT_PATH) / 1024 / 1024)
    logger.info("  新增特征: %d 个", len(sector_cols) - 2)  # minus symbol, date


if __name__ == "__main__":
    run_sector_features()
