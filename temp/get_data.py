"""获取 A 股数据"""
import akshare as ak
import pandas as pd

# 获取平安银行数据
df = ak.stock_zh_a_hist(symbol="000001.SZ", period="daily", start_date="20230101", end_date="20241231")

print(f"数据形状：{df.shape}")
print(f"日期范围：{df['日期'].min()} 至 {df['日期'].max()}")
print(f"\n前 5 行数据:")
print(df.head())

# 保存测试数据
df.to_csv("/Users/et/program/gp-quant/temp/sample_data.csv", index=False)
print(f"\n数据已保存到 sample_data.csv")
