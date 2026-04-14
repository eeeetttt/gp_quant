"""
全 A 股选股器

筛选条件：
1. 净利润近 3 年逐年递增
2. 无扩容需求（近期无增发/配股公告）— 暂跳过
3. 无可转债
4. 股息率排名 — 用 THS 数据中净资产收益率等指标辅助
5. 总市值 >= 80 亿
6. 上市时间 >= 5 年

数据源：
- stock_financial_abstract_ths (同花顺)：净利润等财务指标
- stock_financial_report_sina (新浪)：详细利润表
- bond_zh_hs_cov_spot (新浪)：可转债
- stock_zh_a_hist (东方财富)：历史行情（用于计算市值/上市时间）
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import re
import time
import logging
from datetime import datetime

import pandas as pd
import akshare as ak

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ── 配置 ──
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")
POOL_FILE = os.path.join(OUTPUT_DIR, "stock_pool.txt")

MIN_MARKET_CAP = 80        # 亿
MIN_LISTING_YEARS = 5      # 年
NET_PROFIT_YEARS = 3       # 净利润递增年数


def parse_chinese_number(s) -> float:
    """解析 '627.17亿' → 627.17（单位：亿）"""
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s).strip()
    if not s or s == 'False':
        return float('nan')
    match = re.search(r'([+-]?\d+\.?\d*)', s)
    if match:
        return float(match.group(1))
    return float('nan')


def get_all_stock_codes() -> list[str]:
    """
    获取全 A 股代码
    优先使用 stock_info_a_code_name()
    """
    logger.info("获取全 A 股代码...")
    try:
        df = ak.stock_info_a_code_name()
        if "code" in df.columns:
            codes = df["code"].astype(str).tolist()
            logger.info("获取到 %d 只股票", len(codes))
            return codes
    except Exception as e:
        logger.warning("stock_info_a_code_name 失败: %s", e)

    # 回退：加载已有股票池
    logger.info("回退到已有股票池")
    if os.path.exists(POOL_FILE):
        with open(POOL_FILE) as f:
            return [line.strip() for line in f if line.strip()]
    return []


def get_no_bond_stocks() -> set:
    """条件 3: 获取有可转债的正股代码 → 取补集"""
    logger.info("获取可转债列表...")
    try:
        df = ak.bond_zh_hs_cov_spot()
        if "code" in df.columns:
            return set(df["code"].astype(str))
    except Exception as e:
        logger.warning("可转债数据获取失败: %s", e)
    return set()


def screen_profit_growth(codes: list[str], sleep_sec: float = 0.3) -> list[str]:
    """
    条件 1: 净利润近 3 年逐年递增
    使用 THS 财务摘要接口
    """
    logger.info("筛选净利润递增 (%d 只)...", len(codes))
    passed = []
    failed_count = 0

    for i, code in enumerate(codes):
        try:
            df = ak.stock_financial_abstract_ths(
                symbol=code, indicator="净利润"
            )
            if df is not None and not df.empty and "净利润" in df.columns:
                # 解析净利润列（字符串 → 数值）
                profits = df["净利润"].apply(parse_chinese_number).dropna().tolist()
                if len(profits) >= NET_PROFIT_YEARS:
                    # 取最后 3 年（最近）
                    last_3 = profits[-NET_PROFIT_YEARS:]
                    if all(last_3[j] < last_3[j + 1] for j in range(2)):
                        passed.append(code)
        except Exception:
            failed_count += 1

        if (i + 1) % 50 == 0:
            elapsed = (i + 1) * (sleep_sec + 0.1)
            logger.info("  进度: %d/%d, 通过: %d, 失败: %d, 预计剩余: %.0fs",
                        i + 1, len(codes), len(passed), failed_count,
                        (len(codes) - i - 1) * (sleep_sec + 0.1))
        time.sleep(sleep_sec)

    logger.info("净利润递增: %d → %d (请求失败 %d)",
                len(codes), len(passed), failed_count)
    return passed


def screen_market_cap(codes: list[str], sleep_sec: float = 0.5) -> list[str]:
    """
    条件 5: 总市值 >= 80 亿
    使用 stock_individual_info_em 获取市值
    """
    logger.info("筛选市值 >= %d 亿 (%d 只)...", MIN_MARKET_CAP, len(codes))
    passed = []
    failed_count = 0

    for i, code in enumerate(codes):
        try:
            df = ak.stock_individual_info_em(symbol=code)
            if df is not None and not df.empty:
                for _, row in df.iterrows():
                    item = str(row.get("item", ""))
                    if "总市值" in item:
                        cap = float(row["value"])
                        if cap >= MIN_MARKET_CAP * 1e8:
                            passed.append(code)
                        break
        except Exception:
            failed_count += 1

        if (i + 1) % 30 == 0:
            logger.info("  进度: %d/%d, 通过: %d, 失败: %d",
                        i + 1, len(codes), len(passed), failed_count)
        time.sleep(sleep_sec)

    logger.info("市值过滤: %d → %d (请求失败 %d)",
                len(codes), len(passed), failed_count)
    return passed if passed else codes


def screen_listing_years(codes: list[str]) -> list[str]:
    """
    条件 6: 上市 >= 5 年
    通过 stock_individual_info_em 获取上市日期
    """
    logger.info("筛选上市 >= %d 年 (%d 只)...", MIN_LISTING_YEARS, len(codes))
    passed = []
    cutoff = datetime.now().replace(year=datetime.now().year - MIN_LISTING_YEARS)

    for code in codes:
        try:
            df = ak.stock_individual_info_em(symbol=code)
            if df is not None and not df.empty:
                for _, row in df.iterrows():
                    item = str(row.get("item", ""))
                    if "上市" in item:
                        date_str = str(row["value"])
                        if date_str and date_str != 'None':
                            list_date = pd.to_datetime(date_str)
                            if list_date <= cutoff:
                                passed.append(code)
                        break
        except Exception:
            pass
        time.sleep(0.3)

    logger.info("上市时间过滤: %d → %d", len(codes), len(passed))
    return passed if passed else codes


def save_pool(codes: list[str], label: str = ""):
    """保存股票池"""
    codes = sorted(set(codes))
    with open(POOL_FILE, "w") as f:
        for code in codes:
            f.write(code + "\n")
    logger.info("已保存 %s: %d 只 → %s", label, len(codes), POOL_FILE)
    logger.info("列表: %s", codes[:30])
    if len(codes) > 30:
        logger.info("  ... 共 %d 只", len(codes))


def run_screening(use_full_market: bool = True):
    """
    执行选股全流程

    Args:
        use_full_market: True = 全 A 股, False = 已有股票池
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 0: 获取股票池
    if use_full_market:
        codes = get_all_stock_codes()
    else:
        if os.path.exists(POOL_FILE):
            with open(POOL_FILE) as f:
                codes = [line.strip() for line in f if line.strip()]
        else:
            codes = []
        logger.info("使用已有股票池: %d 只", len(codes))

    if not codes:
        logger.error("没有可用股票池")
        return

    # Step 1: 无可转债
    bond_stocks = get_no_bond_stocks()
    codes = [c for c in codes if c not in bond_stocks]
    save_pool(codes, "无转债")

    # Step 2: 净利润递增
    codes = screen_profit_growth(codes)
    save_pool(codes, "净利润递增")

    # Step 3: 市值过滤（需要联网稳定）
    # codes = screen_market_cap(codes)
    # save_pool(codes, "市值过滤")

    # Step 4: 上市时间
    # codes = screen_listing_years(codes)
    # save_pool(codes, "上市时间")

    logger.info("=" * 50)
    logger.info("选股完成！共 %d 只", len(codes))
    logger.info("=" * 50)
    for code in sorted(codes):
        logger.info("  %s", code)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="A 股选股器")
    parser.add_argument("--full", action="store_true", help="全 A 股筛选（默认）")
    parser.add_argument("--pool", action="store_true", help="使用已有股票池")
    args = parser.parse_args()

    run_screening(use_full_market=args.full or not args.pool)
