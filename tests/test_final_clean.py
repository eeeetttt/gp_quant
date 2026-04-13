"""最终验证清洗数据"""
import pandas as pd

df = pd.read_csv('experiments/data/cleaned_pool.csv', dtype={'symbol': str})

print(f"总行数: {len(df)}")
print(f"股票数: {df['symbol'].nunique()}")
print(f"日期范围: {df['date'].min()} ~ {df['date'].max()}")
print()

# NaN 检查
print("NaN 统计:")
for col in df.columns:
    n = df[col].isna().sum()
    if n > 0:
        print(f"  {col}: {n} ({n/len(df)*100:.2f}%)")
print()

# 停牌日确认（不应再有）
na_vol = df[df['volume'].isna()]
print(f"停牌日残留: {len(na_vol)} 行")

# 各股票行数
counts = df.groupby('symbol').size()
print(f"\n行数分布: min={counts.min()}, max={counts.max()}, avg={counts.mean():.0f}")

# 列
print(f"\n列: {list(df.columns)}")
