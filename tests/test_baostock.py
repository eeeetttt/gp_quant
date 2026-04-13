"""测试 baostock 接口可用性"""
import baostock as bs

# 登录
lg = bs.login()
print("登录结果:", lg)

# 查询股票基本信息
rs = bs.query_stock_basic(code="sh.600519")
print("\n=== sh.600519 基本信息 ===")
if rs.error_code == '0':
    print(f"fields: {rs.fields}")
    data = rs.get_data()
    print(data.to_string())
else:
    print(f"ERROR: {rs.error_msg}")

# 获取日K线
rs2 = bs.query_history_k_data_plus(
    "sh.600519",
    "date,open,high,low,close,volume",
    start_date="2025-04-01",
    end_date="2026-04-13",
    frequency="d",
    adjustflag="2"  # 前复权
)
print("\n=== sh.600519 日K线 ===")
if rs2.error_code == '0':
    print(f"fields: {rs2.fields}")
    df = rs2.get_data()
    print(f"shape: {df.shape}")
    print(df.head(5))
else:
    print(f"ERROR: {rs2.error_msg}")

# 查几只别的
for code in ["sz.000333", "sh.600000", "sz.002867"]:
    rs3 = bs.query_history_k_data_plus(
        code,
        "date,open,high,low,close,volume",
        start_date="2025-04-01",
        end_date="2026-04-13",
        frequency="d",
        adjustflag="2"
    )
    if rs3.error_code == '0':
        df3 = rs3.get_data()
        print(f"{code}: {len(df3)} rows")
    else:
        print(f"{code}: ERROR {rs3.error_msg}")

bs.logout()
