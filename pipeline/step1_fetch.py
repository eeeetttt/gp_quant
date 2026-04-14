"""
批量获取股票历史数据（使用 baostock）
从股票池中拉取近1年日线数据，清洗后保存为CSV
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import logging

import pandas as pd
import baostock as bs

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── 从股票池文件读取 ──
POOL_FILE = os.path.join(os.path.dirname(__file__), "data", "stock_pool.txt")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")

with open(POOL_FILE) as f:
    SYMBOLS = [line.strip() for line in f if line.strip()]


def code_to_baostock(code: str) -> str:
    """6位代码 → baostock格式"""
    if code.startswith(('6', '9')):
        return f"sh.{code}"
    elif code.startswith(('0', '3')):
        return f"sz.{code}"
    else:
        return f"sz.{code}"


def fetch_all(
    start_date: str = "2025-04-01",
    end_date: str = "2026-04-13",
) -> pd.DataFrame:
    """批量拉取数据，返回合并后的 DataFrame"""
    bs.login()

    all_frames = []
    success = 0
    failed = []

    for i, sym in enumerate(SYMBOLS):
        try:
            bs_code = code_to_baostock(sym)
            rs = bs.query_history_k_data_plus(
                bs_code,
                "date,open,high,low,close,volume,amount",
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag="2"  # 前复权
            )
            if rs.error_code == '0':
                df = rs.get_data()
                if not df.empty and len(df) > 0:
                    # 转换数值列
                    for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    df['symbol'] = sym
                    all_frames.append(df)
                    success += 1
                    logger.info("[%d/%d] %s → %d 条记录", i + 1, len(SYMBOLS), sym, len(df))
                else:
                    failed.append(sym)
                    logger.warning("[%d/%d] %s → 无数据", i + 1, len(SYMBOLS), sym)
            else:
                failed.append(sym)
                logger.warning("[%d/%d] %s → ERROR: %s", i + 1, len(SYMBOLS), sym, rs.error_msg)
        except Exception as e:
            failed.append(sym)
            logger.warning("[%d/%d] %s → EXCEPTION: %s", i + 1, len(SYMBOLS), sym, e)

    bs.logout()

    if not all_frames:
        logger.error("没有成功获取任何数据")
        return pd.DataFrame()

    combined = pd.concat(all_frames, ignore_index=True)
    logger.info(
        "完成：成功 %d / %d，总记录 %d 条",
        success, len(SYMBOLS), len(combined),
    )
    if failed:
        logger.info("失败股票: %s", ", ".join(failed))

    return combined


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger.info("股票池: %d 只", len(SYMBOLS))
    logger.info("前 10 只: %s", SYMBOLS[:10])

    combined = fetch_all(
        start_date="2016-01-01",
        end_date="2026-04-13",
    )

    if not combined.empty:
        out_path = os.path.join(OUTPUT_DIR, "stock_pool_1y.csv")
        combined.to_csv(out_path, index=False)
        logger.info("已保存到 %s (%.1f MB)", out_path,
                     os.path.getsize(out_path) / 1024 / 1024)
        print(f"\n数据概览：")
        print(f"  股票数: {combined['symbol'].nunique()}")
        print(f"  总行数: {len(combined)}")
        print(f"  日期范围: {combined['date'].min()} ~ {combined['date'].max()}")
        print(f"  列: {list(combined.columns)}")
        print(f"\n前 5 行：")
        print(combined.head())
