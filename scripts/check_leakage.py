"""检查数据泄漏"""
import pandas as pd
import numpy as np

df = pd.read_csv('experiments/ml/features.csv', dtype={'symbol': str})
sym = df['symbol'].iloc[0]
stock = df[df['symbol'] == sym].sort_values('date').tail(20)

print("=== 检查 target 和特征的关系 ===")
print(f"target_direction_5d = {'向上' if stock['target_direction_5d'].iloc[-1] == 1 else '向下'}")
print(f"return_5d (过去5日收益) = {stock['return_5d'].iloc[-1]:.4f}")
print(f"return_1d (昨日收益) = {stock['return_1d'].iloc[-1]:.4f}")
print(f"alpha_5d (超额收益) = {stock['alpha_5d'].iloc[-1]:.4f}")

print("\n=== target 的定义 ===")
# 看看 target_direction_5d 到底和哪个特征完全相关
for col in ['return_1d', 'return_5d', 'return_10d', 'return_20d',
            'alpha_1d', 'alpha_5d', 'change', 'close_lag_1',
            'target_5d']:
    corr = df[col].corr(df['target_direction_5d'], method='pearson')
    print(f"  {col} vs target: corr = {corr:.4f}")

print("\n=== 随机抽样检查 ===")
sample = df.sample(n=50, random_state=42)
for _, row in sample.head(5).iterrows():
    print(f"  return_5d={row['return_5d']:.4f}  target_dir={row['target_direction_5d']}  target_5d={row['target_5d']:.4f}")
