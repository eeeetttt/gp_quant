"""
数据清洗脚本

按 symbol 分组清洗，输出：
- experiments/data/cleaned/<symbol>.csv（每只独立）
- experiments/data/cleaned_pool.csv（合并版）
"""
import sys
import os
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import logging

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="数据清洗")
parser.add_argument("--pool", choices=["small", "500", "full"], default="small",
                    help="股票池: small=30只, 500=500只, full=5502只")
args = parser.parse_args()

from pipeline.utils import get_pool_input_path

RAW_PATH = get_pool_input_path(args.pool)
CLEANED_DIR = os.path.join(os.path.dirname(__file__), "data", "cleaned")
MERGED_PATH = os.path.join(os.path.dirname(__file__), "data", "cleaned_pool.csv")


def clean_single(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """清洗单只股票"""
    df = df.copy()

    # 1. 转数值类型（baostock 返回的是字符串）
    for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 2. 去空值
    df = df.dropna(subset=['close'])

    # 3. 按日期排序 + 去重
    df = df.sort_values('date').reset_index(drop=True)
    df = df.drop_duplicates(subset=['date'], keep='last')

    # 4. OHLC 逻辑一致性检查
    #    high >= low, high >= open, high >= close
    #    low <= open, low <= close
    #    open/close 不应为负
    mask = (
        (df['high'] >= df['low']) &
        (df['high'] >= df['open']) &
        (df['high'] >= df['close']) &
        (df['low'] <= df['open']) &
        (df['low'] <= df['close']) &
        (df['open'] > 0) & (df['close'] > 0)
    )
    n_bad = (~mask).sum()
    df = df[mask]

    # 5. 计算衍生列
    df['change'] = df['close'].pct_change() * 100
    df['change_abs'] = df['close'] - df['open']
    df['range'] = df['high'] - df['low']
    df['range_pct'] = df['range'] / df['close'] * 100

    # 6. 多周期收益率
    for period in [1, 5, 10, 20]:
        df[f'return_{period}d'] = df['close'].pct_change(periods=period) * 100

    # 7. 排除极端异常（range_pct > 50%）
    df = df[df['range_pct'] < 50]

    # 8. 排除 close <= 0
    df = df[df['close'] > 0]

    # 9. 排除停牌日（volume 为 NaN，价格不变）
    n_suspended = df['volume'].isna().sum()
    df = df.dropna(subset=['volume'])

    return df


def run():
    os.makedirs(CLEANED_DIR, exist_ok=True)

    logger.info("读取原始数据: %s", RAW_PATH)
    raw = pd.read_csv(RAW_PATH, dtype={'symbol': str})
    raw['symbol'] = raw['symbol'].str.zfill(6)
    logger.info("原始数据: %d 行, %d 只股票", len(raw), raw['symbol'].nunique())

    symbols = sorted(raw['symbol'].unique())
    cleaned_frames = []

    for sym in symbols:
        df = raw[raw['symbol'] == sym]
        cleaned = clean_single(df, sym)

        if cleaned.empty:
            logger.warning("[%s] 清洗后无数据，跳过", sym)
            continue

        # 保存独立文件
        out_path = os.path.join(CLEANED_DIR, f"{sym}.csv")
        cleaned.to_csv(out_path, index=False)

        cleaned_frames.append(cleaned)

        before = len(df)
        after = len(cleaned)
        if before != after:
            logger.info("[%s] %d 行 → %d 行 (删除 %d)", sym, before, after, before - after)

    # 合并版
    merged = pd.concat(cleaned_frames, ignore_index=True)
    merged.to_csv(MERGED_PATH, index=False)

    logger.info("=" * 50)
    logger.info("清洗完成!")
    logger.info("独立文件: %d 只 → %s/", len(cleaned_frames), CLEANED_DIR)
    logger.info("合并文件: %s (%.1f MB)", MERGED_PATH,
                 os.path.getsize(MERGED_PATH) / 1024 / 1024)
    logger.info("总行数: %d", len(merged))
    logger.info("日期范围: %s ~ %s", merged['date'].min(), merged['date'].max())
    logger.info("列: %s", list(merged.columns))


if __name__ == "__main__":
    run()
