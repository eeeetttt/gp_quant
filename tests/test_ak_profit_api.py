"""调试 stock_financial_abstract_ths 的返回结构"""
import akshare as ak

for symbol in ["600519", "600036", "601318", "600000"]:
    print(f"=== {symbol} ===")
    try:
        df = ak.stock_financial_abstract_ths(symbol=symbol, indicator='净利润')
        print(f"  shape: {df.shape}")
        print(f"  columns: {list(df.columns)}")
        if not df.empty:
            print(f"  head:\n{df.head(5)}")
        else:
            print("  空 DataFrame")
    except Exception as e:
        print(f"  ERROR: {e}")
    print()

# 也试试新浪的利润表
print("\n=== 新浪利润表 (600519) ===")
try:
    df2 = ak.stock_financial_report_sina(stock="600519", symbol="利润表")
    print(f"  shape: {df2.shape}")
    print(f"  columns: {list(df2.columns)}")
    if '净利润' in df2.columns:
        print(f"  净利润前 5:\n{df2[['报告日', '净利润']].head(5)}")
except Exception as e:
    print(f"  ERROR: {e}")
