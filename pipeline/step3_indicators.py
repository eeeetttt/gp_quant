"""
Phase 3: 技术指标计算

输入：experiments/data/cleaned_pool.csv
输出：experiments/data/indicators_pool.csv（合并版）
     experiments/data/indicators/<symbol>.csv（独立文件）
"""
import sys
import os
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import logging

import pandas as pd
import numpy as np

from gp_quant.strategy.indicators import TechnicalIndicators

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="技术指标计算")
parser.add_argument("--pool", choices=["small", "500", "full"], default="small",
                    help="股票池: small=30只, 500=500只, full=5502只")
args = parser.parse_args()
# Note: --pool accepted for CLI consistency. Input is always cleaned_pool.csv (from step2).

CLEANED_PATH = os.path.join(os.path.dirname(__file__), "data", "cleaned_pool.csv")
IND_DIR = os.path.join(os.path.dirname(__file__), "data", "indicators")
MERGED_PATH = os.path.join(os.path.dirname(__file__), "data", "indicators_pool.csv")


def calc_indicators_single(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """对单只股票计算所有技术指标"""
    if len(df) < 60:
        logger.warning("[%s] 仅 %d 行，跳过（需要至少60行）", symbol, len(df))
        return pd.DataFrame()

    # 确保按日期排序
    df = df.sort_values('date').reset_index(drop=True)

    ti = TechnicalIndicators(df)
    result = ti.get_all_indicators()

    return result


def run():
    os.makedirs(IND_DIR, exist_ok=True)

    logger.info("读取清洗后数据: %s", CLEANED_PATH)
    cleaned = pd.read_csv(CLEANED_PATH, dtype={'symbol': str})
    logger.info("原始数据: %d 行, %d 只股票", len(cleaned), cleaned['symbol'].nunique())

    symbols = sorted(cleaned['symbol'].unique())
    ind_frames = []
    skipped = 0

    for i, sym in enumerate(symbols):
        df = cleaned[cleaned['symbol'] == sym]
        result = calc_indicators_single(df, sym)

        if result.empty:
            skipped += 1
            continue

        # 保存独立文件
        out_path = os.path.join(IND_DIR, f"{sym}.csv")
        result.to_csv(out_path, index=False)
        ind_frames.append(result)

        if (i + 1) % 20 == 0:
            logger.info("进度: %d/%d", i + 1, len(symbols))

    merged = pd.concat(ind_frames, ignore_index=True)
    merged.to_csv(MERGED_PATH, index=False)

    logger.info("=" * 50)
    logger.info("指标计算完成!")
    logger.info("独立文件: %d 只 → %s/", len(ind_frames), IND_DIR)
    logger.info("合并文件: %s (%.1f MB)", MERGED_PATH,
                 os.path.getsize(MERGED_PATH) / 1024 / 1024)
    logger.info("总行数: %d", len(merged))
    logger.info("列: %s", list(merged.columns))
    if skipped:
        logger.info("跳过: %d 只（数据不足）", skipped)


if __name__ == "__main__":
    run()
