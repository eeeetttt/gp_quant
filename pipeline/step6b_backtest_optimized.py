"""
Phase 6b: 回测优化模型

用优化后的 LightGBM/XGBoost 模型生成买卖信号。

策略逻辑：
  - 模型预测未来 5 日上涨概率 > 阈值 → BUY
  - 已持仓股票，预测概率 < 阈值 → SELL（平仓）
  - 最大持仓数限制，分散风险

输入：experiments/ml/features.csv, models/model_optimized.pkl
输出：experiments/results/backtest_optimized_report.json
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import logging
import json
import joblib
import pandas as pd
import numpy as np
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FEATURES_PATH = os.path.join(os.path.dirname(__file__), "ml", "features.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "model_optimized.pkl")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
BACKTEST_REPORT_PATH = os.path.join(RESULTS_DIR, "backtest_optimized_report.json")

MAX_POSITIONS = 5  # 最大同时持仓数


def backtest_optimized():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 加载模型
    logger.info("加载优化模型: %s", MODEL_PATH)
    model_data = joblib.load(MODEL_PATH)
    model = model_data['model']
    scaler = model_data['scaler']
    feature_cols = model_data['feature_cols']
    threshold = model_data.get('threshold', 0.50)
    model_type = model_data.get('model_type', str(type(model).__name__))

    logger.info("模型类型: %s, 优化阈值: %.2f", model_type, threshold)
    logger.info("特征数: %d", len(feature_cols))

    # 加载特征数据
    logger.info("加载特征数据: %s", FEATURES_PATH)
    df = pd.read_csv(FEATURES_PATH, dtype={'symbol': str})
    df = df.dropna(subset=feature_cols)

    # 生成模型预测
    logger.info("生成预测 (%d 行, %d 特征)...", len(df), len(feature_cols))
    X = scaler.transform(df[feature_cols].values)
    df['prob_up'] = model.predict_proba(X)[:, 1]
    df['signal'] = df['prob_up'].apply(lambda p: "BUY" if p >= threshold else "HOLD")

    logger.info("BUY 信号: %d (%.1f%%)", (df['signal'] == 'BUY').sum(),
                (df['signal'] == 'BUY').mean() * 100)

    # 按日期遍历，强制执行 MAX_POSITIONS 限制
    symbols = sorted(df['symbol'].unique())
    logger.info("回测 %d 只股票 (最大持仓 %d)...", len(symbols), MAX_POSITIONS)

    all_trades = []
    open_positions = {}  # symbol -> {entry_date, entry_price, entry_prob, entry_idx}

    dates = sorted(df['date'].unique())
    for date in dates:
        day_df = df[df['date'] == date]
        if day_df.empty:
            continue

        # 先检查已持仓的平仓信号
        for sym in list(open_positions.keys()):
            row = day_df[day_df['symbol'] == sym]
            if row.empty:
                continue
            row = row.iloc[0]
            prob = row['prob_up']
            price = row['close']
            pos = open_positions[sym]
            days_held = day_df.index[0] - pos['entry_idx']

            should_sell = (prob < threshold)
            max_hold = days_held >= 10

            if should_sell or max_hold:
                pnl_pct = (price - pos['entry_price']) / pos['entry_price'] * 100
                all_trades.append({
                    "symbol": sym,
                    "entry_date": str(pos['entry_date']),
                    "exit_date": str(date),
                    "entry_price": round(float(pos['entry_price']), 4),
                    "exit_price": round(float(price), 4),
                    "entry_prob": round(float(pos['entry_prob']), 4),
                    "exit_prob": round(float(prob), 4),
                    "pnl_pct": round(float(pnl_pct), 4),
                    "is_win": bool(pnl_pct > 0),
                    "days_held": int(days_held),
                })
                del open_positions[sym]

        # 再处理新买入信号
        buy_signals = day_df[day_df['signal'] == 'BUY'].sort_values('prob_up', ascending=False)
        for _, row in buy_signals.iterrows():
            if len(open_positions) >= MAX_POSITIONS:
                break
            sym = row['symbol']
            if sym not in open_positions:
                open_positions[sym] = {
                    'entry_date': date,
                    'entry_price': row['close'],
                    'entry_prob': row['prob_up'],
                    'entry_idx': day_df.index[0],
                }

    # 清算剩余持仓
    for sym, pos in open_positions.items():
        stock_df = df[df['symbol'] == sym].sort_values('date')
        last_row = stock_df.iloc[-1]
        pnl_pct = (last_row['close'] - pos['entry_price']) / pos['entry_price'] * 100
        all_trades.append({
            "symbol": sym,
            "entry_date": str(pos['entry_date']),
            "exit_date": str(last_row['date']),
            "entry_price": round(float(pos['entry_price']), 4),
            "exit_price": round(float(last_row['close']), 4),
            "entry_prob": round(float(pos['entry_prob']), 4),
            "exit_prob": round(float(last_row['prob_up']), 4),
            "pnl_pct": round(float(pnl_pct), 4),
            "is_win": bool(pnl_pct > 0),
            "days_held": int(len(stock_df) - 1 - pos['entry_idx']),
        })

    # 计算指标
    total_trades = len(all_trades)
    if total_trades == 0:
        logger.warning("没有产生任何交易")
        return

    pnls = [t['pnl_pct'] for t in all_trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    winning_trades = len(wins)

    win_rate = winning_trades / total_trades * 100
    avg_pnl = np.mean(pnls)
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    profit_factor = abs(sum(wins) / sum(losses)) if sum(losses) != 0 else float('inf')

    cumulative = np.cumsum(pnls)
    peak = np.maximum.accumulate(cumulative)
    drawdowns = cumulative - peak
    max_drawdown = float(np.min(drawdowns))

    # 按月份统计
    monthly = defaultdict(list)
    for t in all_trades:
        month = str(t['entry_date'])[:7]
        monthly[month].append(t['pnl_pct'])

    monthly_stats = {}
    for month, pnl_list in sorted(monthly.items()):
        monthly_stats[month] = {
            "trades": len(pnl_list),
            "avg_pnl": round(float(np.mean(pnl_list)), 4),
            "win_rate": round(sum(1 for p in pnl_list if p > 0) / len(pnl_list) * 100, 2),
        }

    # 按预测概率分组
    prob_buckets = {
        "0.50-0.55": [], "0.55-0.60": [], "0.60-0.65": [],
        "0.65-0.70": [], "0.70-0.75": [], "0.75-0.80": [], "0.80+": []
    }
    for t in all_trades:
        p = t['entry_prob']
        if p < 0.55:
            prob_buckets["0.50-0.55"].append(t['pnl_pct'])
        elif p < 0.60:
            prob_buckets["0.55-0.60"].append(t['pnl_pct'])
        elif p < 0.65:
            prob_buckets["0.60-0.65"].append(t['pnl_pct'])
        elif p < 0.70:
            prob_buckets["0.65-0.70"].append(t['pnl_pct'])
        elif p < 0.75:
            prob_buckets["0.70-0.75"].append(t['pnl_pct'])
        elif p < 0.80:
            prob_buckets["0.75-0.80"].append(t['pnl_pct'])
        else:
            prob_buckets["0.80+"].append(t['pnl_pct'])

    prob_analysis = {}
    for bucket, pnl_list in prob_buckets.items():
        if pnl_list:
            prob_analysis[bucket] = {
                "trades": len(pnl_list),
                "avg_pnl": round(float(np.mean(pnl_list)), 4),
                "win_rate": round(sum(1 for p in pnl_list if p > 0) / len(pnl_list) * 100, 2),
            }

    # 持有天数分布
    days_held_list = [t['days_held'] for t in all_trades]
    hold_dist = {
        "1-2天": sum(1 for d in days_held_list if d <= 2),
        "3-5天": sum(1 for d in days_held_list if 3 <= d <= 5),
        "6-10天": sum(1 for d in days_held_list if 6 <= d <= 10),
    }

    report = {
        "strategy": "Optimized %s - BUY when prob >= %.2f" % (model_type, threshold),
        "model_type": model_type,
        "threshold": round(threshold, 2),
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": total_trades - winning_trades,
        "win_rate": round(win_rate, 2),
        "avg_pnl_pct": round(float(avg_pnl), 4),
        "avg_win_pct": round(float(avg_win), 4),
        "avg_loss_pct": round(float(avg_loss), 4),
        "profit_factor": round(float(profit_factor), 4),
        "max_drawdown_pct": round(max_drawdown, 4),
        "total_return_pct": round(float(np.sum(pnls)), 4),
        "sharpe_approx": round(float(np.mean(pnls) / np.std(pnls)) if np.std(pnls) > 0 else 0, 4),
        "avg_days_held": round(float(np.mean(days_held_list)), 2),
        "hold_distribution": hold_dist,
        "monthly_stats": monthly_stats,
        "prob_analysis": prob_analysis,
        "sample_trades": all_trades[:20],
    }

    with open(BACKTEST_REPORT_PATH, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # 打印报告
    print("\n" + "=" * 60)
    print("回测报告（优化模型）")
    print("=" * 60)
    print(f"策略:           {report['strategy']}")
    print(f"模型:           {report['model_type']}")
    print(f"总交易次数:     {total_trades}")
    print(f"胜率:           {win_rate:.2f}%")
    print(f"平均盈亏:       {avg_pnl:.4f}%")
    print(f"平均盈利:       {avg_win:.4f}%")
    print(f"平均亏损:       {avg_loss:.4f}%")
    print(f"盈亏比:         {profit_factor:.4f}")
    print(f"最大回撤:       {max_drawdown:.4f}%")
    print(f"总收益率:       {np.sum(pnls):.4f}%")
    print(f"Sharpe (approx): {report['sharpe_approx']:.4f}")
    print(f"平均持有天数:   {report['avg_days_held']:.1f}")
    print(f"持有分布:       1-2天={hold_dist['1-2天']}, 3-5天={hold_dist['3-5天']}, 6-10天={hold_dist['6-10天']}")
    print("=" * 60)

    print("\n按预测概率分组分析:")
    for bucket, stats in prob_analysis.items():
        print(f"  {bucket}: {stats['trades']}笔, 胜率={stats['win_rate']}%, 平均盈亏={stats['avg_pnl']}%")

    print(f"\n月度统计 (前10个月):")
    for month, stats in list(monthly_stats.items())[:10]:
        print(f"  {month}: {stats['trades']}笔, 胜率={stats['win_rate']}%, 平均盈亏={stats['avg_pnl']}%")

    print(f"\n报告已保存: {BACKTEST_REPORT_PATH}")


if __name__ == "__main__":
    backtest_optimized()
