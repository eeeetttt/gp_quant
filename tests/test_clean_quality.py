"""检查清洗后数据质量"""
import pandas as pd

# 看合并版
df = pd.read_csv('experiments/data/cleaned_pool.csv')
print(f"合并版: {df.shape}")
print(f"列: {list(df.columns)}")
print()

# 看单只股票的样例
single = pd.read_csv('experiments/data/cleaned/000088.csv', dtype={'symbol': str})
print(f"000088: {single.shape}")
print(single.head(10).to_string())
print()
print(single.tail(5).to_string())
print()

# 检查 NaN
print("NaN 统计:")
has_nan = False
for col in df.columns:
    n = df[col].isna().sum()
    if n > 0:
        print(f"  {col}: {n}")
        has_nan = True
if not has_nan:
    print("  无 NaN")

# 检查每只股票的行数
print()
print("各股票行数分布:")
counts = df.groupby('symbol').size()
print(f"  最小: {counts.min()}")
print(f"  最大: {counts.max()}")
print(f"  平均: {counts.mean():.0f}")
