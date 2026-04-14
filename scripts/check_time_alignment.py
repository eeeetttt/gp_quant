"""检查特征和目标变量的时间对齐"""
import pandas as pd
import numpy as np

df = pd.read_csv('experiments/ml/features.csv', dtype={'symbol': str})
sym = '600519'
stock = df[df['symbol'] == sym].sort_values('date').reset_index(drop=True)

print(f"=== {sym} 时间对齐检查 ===")
print(f"总行数: {len(stock)}")

# 看连续几行的关系
for i in range(5, len(stock) - 5):
    t = stock.iloc[i]
    t_plus_5 = stock.iloc[i + 5]
    t_minus_5 = stock.iloc[i - 5]

    # 手动计算 target
    target_manual = (t_plus_5['close'] - t['close']) / t['close'] * 100
    return_5d_manual = (t['close'] - t_minus_5['close']) / t_minus_5['close'] * 100

    if abs(target_manual - t['target_5d']) > 0.01 or abs(return_5d_manual - t['return_5d']) > 0.01:
        print(f"第 {i} 行 时间对齐错误!")
        print(f"  target_5d={t['target_5d']:.4f}  manual={target_manual:.4f}")
        print(f"  return_5d={t['return_5d']:.4f}  manual={return_5d_manual:.4f}")
        break
else:
    print("时间对齐正确")

# 关键检查：return_5d 和 target_5d 是否在同一时间窗口
print(f"\n=== 检查 return_5d 和 target_5d 的相关性 ===")
# 如果 return_5d 真的只是历史数据，和 target 的相关性不应该超过 0.3-0.4
print(f"  return_5d vs target_5d: {stock['return_5d'].corr(stock['target_5d']):.4f}")

# 更关键的：检查是否有某个特征和 target 几乎完全一样
print(f"\n=== 特征与 target_5d 的差值 ===")
features = ['return_5d', 'return_1d', 'return_10d', 'return_20d',
            'bb_position', 'price_bb_position', 'k', 'wr',
            'close_lag_1', 'close_lag_2', 'close_lag_3', 'close_lag_5']

for col in features:
    diff = (stock[col] - stock['target_5d']).abs()
    print(f"  {col:20s} min_diff={diff.min():.6f}  mean_diff={diff.mean():.4f}  max_diff={diff.max():.4f}")
    if diff.min() < 0.001:
        print(f"    *** {col} 和 target_5d 几乎相同！***")
