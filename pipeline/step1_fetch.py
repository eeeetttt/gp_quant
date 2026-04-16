"""
批量获取股票历史数据（使用 baostock，多线程并行）
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import logging
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import baostock as bs

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

import argparse

parser = argparse.ArgumentParser(description="批量获取股票历史数据")
parser.add_argument("--pool", choices=["small", "500", "full"], default="small",
                    help="股票池: small=30只, 500=500只, full=5502只")
parser.add_argument("--workers", type=int, default=3, help="并发进程数（baostock 限制，建议 ≤3）")
args = parser.parse_args()

from pipeline.utils import get_pool_file, get_pool_input_path

POOL_FILE = get_pool_file(args.pool)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_NAME = f"stock_pool_{args.pool}.csv"

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


def fetch_single(sym: str, start_date: str, end_date: str, max_retries: int = 3):
    """
    单只股票数据拉取（每个进程独立 baostock 连接）
    自动重试 max_retries 次
    """
    bs_code = code_to_baostock(sym)

    for attempt in range(max_retries):
        login_ok = bs.login()
        if login_ok.error_code != '0':
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return None, sym

        try:
            rs = bs.query_history_k_data_plus(
                bs_code,
                "date,open,high,low,close,volume,amount",
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag="2"
            )
            if rs.error_code == '0':
                df = rs.get_data()
                if not df.empty and len(df) > 0:
                    for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    df['symbol'] = sym
                    return df, None
        except Exception:
            pass
        finally:
            bs.logout()

        if attempt < max_retries - 1:
            time.sleep(2)  # 失败后等一会再重试

    return None, sym


def fetch_all(
    start_date: str = "2016-01-01",
    end_date: str = "2026-04-13",
    max_workers: int = 3,
) -> pd.DataFrame:
    """多进程并行拉取数据，返回合并后的 DataFrame"""
    all_frames = []
    success = 0
    failed = []
    done = 0
    total = len(SYMBOLS)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fetch_single, sym, start_date, end_date): sym
            for sym in SYMBOLS
        }

        for future in as_completed(futures):
            done += 1
            result, fail_sym = future.result()
            if result is not None:
                all_frames.append(result)
                success += 1
                if done % 50 == 0 or done == total:
                    logger.info("进度: %d/%d (成功 %d, 失败 %d)", done, total, success, done - success)
            else:
                failed.append(fail_sym)

    if not all_frames:
        logger.error("没有成功获取任何数据")
        return pd.DataFrame()

    combined = pd.concat(all_frames, ignore_index=True)
    logger.info(
        "完成：成功 %d / %d，总记录 %d 条",
        success, total, len(combined),
    )
    if failed:
        logger.info("失败股票: %s", ", ".join(failed))

    return combined


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger.info("股票池: %d 只, 并发数: %d", len(SYMBOLS), args.workers)
    logger.info("前 10 只: %s", SYMBOLS[:10])

    combined = fetch_all(
        start_date="2016-01-01",
        end_date="2026-04-13",
        max_workers=args.workers,
    )

    if not combined.empty:
        out_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
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
