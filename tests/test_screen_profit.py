"""测试净利润递增筛选在 173 只样本上的效果"""
import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import akshare as ak

# 加载股票池
with open('experiments/data/stock_pool.txt') as f:
    codes = [line.strip() for line in f if line.strip()]
print(f"股票池: {len(codes)} 只")

# 可转债
try:
    bond_df = ak.bond_zh_hs_cov_spot()
    bond_codes = set(bond_df['code'].astype(str))
    print(f"有可转债的正股: {len(bond_codes)} 只")
except Exception as e:
    bond_codes = set()
    print(f"可转债数据获取失败: {e}")

no_bond = [c for c in codes if c not in bond_codes]
print(f"无转债: {len(no_bond)} 只")

# 净利润递增测试 (全部 173 只)
print(f"\n测试净利润递增 (173 只，预计 ~1 分钟)...")
passed = []
failed = []
for i, code in enumerate(codes):
    try:
        df = ak.stock_financial_abstract_ths(symbol=code, indicator='净利润')
        if df is not None and not df.empty and '净利润' in df.columns:
            profits = []
            for v in df['净利润'].head(3):
                try:
                    profits.append(float(v))
                except (ValueError, TypeError):
                    break
            if len(profits) >= 3:
                increasing = all(profits[j] < profits[j + 1] for j in range(2))
                if increasing:
                    passed.append(code)
                    print(f"  {code}: {profits} PASS")
                else:
                    failed.append(code)
            else:
                failed.append(code)
                print(f"  {code}: 数据不足 ({len(profits)} 条)")
        else:
            failed.append(code)
            print(f"  {code}: 无数据")
    except Exception as e:
        failed.append(code)
        print(f"  {code}: ERROR - {str(e)[:80]}")
    time.sleep(0.3)

print(f"\n=== 结果 ===")
print(f"净利润递增: {len(passed)}/{len(codes)}")
print(f"通过: {passed}")

# 写入最终 pool
out_file = 'experiments/data/stock_pool.txt'
with open(out_file, 'w') as f:
    for code in sorted(passed):
        f.write(code + '\n')
print(f"已保存到 {out_file}")
