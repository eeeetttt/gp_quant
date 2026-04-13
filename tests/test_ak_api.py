"""测试 AkShare 各接口可用性"""
import akshare as ak
import time

tests = [
    ("stock_zh_a_spot_em", lambda: ak.stock_zh_a_spot_em()),
    ("stock_individual_info_em('600519')", lambda: ak.stock_individual_info_em(symbol="600519")),
    ("stock_zh_a_hist('600519')", lambda: ak.stock_zh_a_hist(symbol="600519", period="daily", adjust="qfq")),
    ("stock_zh_a_dividend_yield_em", lambda: ak.stock_zh_a_dividend_yield_em() if hasattr(ak, "stock_zh_a_dividend_yield_em") else "not found"),
    ("bond_zh_hs_cov_spot", lambda: ak.bond_zh_hs_cov_spot()),
    ("stock_financial_abstract_ths", lambda: ak.stock_financial_abstract_ths(symbol="600519", indicator="净利润")),
    ("stock_financial_analysis_indicator", lambda: ak.stock_financial_analysis_indicator(symbol="600519")),
    ("stock_financial_report_sina", lambda: ak.stock_financial_report_sina(stock="600519", symbol="利润表")),
    ("stock_profit_sheet_by_report_em", lambda: ak.stock_profit_sheet_by_report_em(symbol="600519")),
    ("stock_a_pe_and_mrq_em", lambda: ak.stock_a_pe_and_mrq_em() if hasattr(ak, "stock_a_pe_and_mrq_em") else "not found"),
]

for name, fn in tests:
    try:
        time.sleep(0.5)
        result = fn()
        if isinstance(result, str):
            print(f"SKIP: {name} -> {result}")
        elif hasattr(result, "shape"):
            print(f"OK: {name} shape={result.shape}, cols={list(result.columns)}")
        else:
            print(f"OK: {name} type={type(result)}")
    except AttributeError as e:
        print(f"NOT_FOUND: {name} -> {e}")
    except Exception as e:
        print(f"ERROR: {name} -> {type(e).__name__}: {str(e)[:100]}")
