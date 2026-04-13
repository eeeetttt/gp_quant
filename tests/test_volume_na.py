"""检查 volume/amount 缺失的情况"""
import pandas as pd

df = pd.read_csv('experiments/data/cleaned_pool.csv', dtype={'symbol': str})

# 找出 volume 为 NaN 的行
na_rows = df[df['volume'].isna()]
print(f"volume/amount 缺失: {len(na_rows)} 行")
print("按股票分布:")
print(na_rows.groupby('symbol').size().sort_values(ascending=False).head(20))

print()
print("缺失行示例:")
print(na_rows.head(10).to_string())
