"""调试 THS 净利润列的数据格式"""
import akshare as ak

for symbol in ["600519", "600036", "601318", "000333", "600000"]:
    print(f"=== {symbol} ===")
    try:
        df = ak.stock_financial_abstract_ths(symbol=symbol, indicator='净利润')
        if not df.empty and '净利润' in df.columns:
            print(f"  净利润列 dtype: {df['净利润'].dtype}")
            print(f"  前 5 行净利润:")
            for i, v in enumerate(df['净利润'].head(5)):
                print(f"    [{i}] {v!r} (type: {type(v).__name__})")
            # 检查最后 3 行（最近的年度数据）
            print(f"  最后 5 行净利润:")
            for i, v in enumerate(df['净利润'].tail(5)):
                print(f"    [{-5+i}] {v!r}")
        else:
            print("  空")
    except Exception as e:
        print(f"  ERROR: {e}")
    print()
