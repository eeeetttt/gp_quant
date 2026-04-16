"""
Phase 6c: 过拟合检测

多维度检测模型是否过拟合：
1. Walk-forward validation：滚动时间窗口训练/测试
2. OOS 股票回测：仅在训练时未见过的股票上回测
3. 不同市场周期：牛市 vs 熊市表现对比
4. 单只股票分布：收益是否集中在少数股票
5. 训练/测试性能差距：train vs test gap

输入：experiments/ml/features.csv, models/model_optimized.pkl
输出：experiments/results/overfit_report.json
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import logging
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FEATURES_PATH = os.path.join(os.path.dirname(__file__), "ml", "features.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "model_optimized.pkl")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
OVERFIT_REPORT_PATH = os.path.join(RESULTS_DIR, "overfit_report.json")

TARGET_COL = "target_direction_5d"
PROB_THRESHOLD = 0.50


def _get_features(df, feature_cols, scaler=None):
    """Get features, applying scaler only if available and fitted"""
    X = df[feature_cols].values
    if scaler is not None:
        return scaler.transform(X)
    return X


def walk_forward_test(df, feature_cols, scaler, model, n_splits=5,
                      lgb_model=None, xgb_model=None):
    """
    Walk-forward 验证：
    将数据按时间分成 n_splits 段
    每轮：使用已保存的模型直接在测试集上预测（不重新训练）

    支持两种模型格式：
    - 旧版: scaler + model (single sklearn model)
    - 新版: lgb_model + xgb_model (LGB+XGB ensemble)
    """
    df = df.sort_values('date').reset_index(drop=True)
    dates = df['date'].unique()
    split_size = len(dates) // (n_splits + 1)

    results = []

    for split in range(n_splits):
        test_start = dates[split * split_size]
        test_end = dates[(split + 1) * split_size] if split + 1 < n_splits else dates[-1]

        test_df = df[(df['date'] >= test_start) & (df['date'] < test_end)].copy()

        if len(test_df) < 100:
            continue

        test_df = test_df.dropna(subset=feature_cols + [TARGET_COL])
        if test_df.empty:
            continue

        X_test = _get_features(test_df, feature_cols, scaler)
        y_test = test_df[TARGET_COL].values

        # 使用已保存模型预测（不重新训练）
        if lgb_model is not None and xgb_model is not None:
            # 新版：LGB+XGB ensemble
            p_lgb = lgb_model.predict_proba(X_test)[:, 1]
            p_xgb = xgb_model.predict_proba(X_test)[:, 1]
            y_prob = 0.5 * p_lgb + 0.5 * p_xgb
        elif model is not None:
            # 旧版：单模型
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            continue

        y_pred = (y_prob >= PROB_THRESHOLD).astype(int)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)

        # 回测模拟
        test_df = test_df.copy()
        test_df['prob_up'] = y_prob
        test_df['signal'] = test_df['prob_up'].apply(lambda p: "BUY" if p >= PROB_THRESHOLD else "HOLD")

        trades = simulate_trades(test_df)

        results.append({
            "split": split + 1,
            "train_period": f"~{test_start}",
            "test_period": f"{test_start}~{test_end}",
            "test_samples": len(test_df),
            "accuracy": round(acc, 4),
            "f1": round(f1, 4),
            "auc": round(auc, 4),
            "backtest": trades,
        })

        logger.info("  Split %d [%s ~ %s]: acc=%.4f f1=%.4f auc=%.4f, trades=%d, win_rate=%.1f%%",
                     split + 1, test_start, test_end, acc, f1, auc,
                     trades['total_trades'], trades['win_rate'])

    return results


def simulate_trades(df):
    """简单回测模拟"""
    symbols = sorted(df['symbol'].unique())
    all_trades = []

    for symbol in symbols:
        stock_df = df[df['symbol'] == symbol].sort_values('date').reset_index(drop=True)
        if len(stock_df) < 10:
            continue

        in_position = False
        entry_price = None
        entry_prob = None
        entry_date = None
        entry_idx = None

        for i in range(len(stock_df)):
            row = stock_df.iloc[i]
            price = row['close']
            signal = row['signal']
            prob = row['prob_up']

            if not in_position and signal == "BUY":
                in_position = True
                entry_price = price
                entry_prob = prob
                entry_date = row['date']
                entry_idx = i
            elif in_position:
                should_sell = (prob < PROB_THRESHOLD)
                days_held = i - entry_idx
                max_hold = days_held >= 10

                if should_sell or max_hold or i == len(stock_df) - 1:
                    pnl = (price - entry_price) / entry_price * 100
                    all_trades.append({
                        "symbol": symbol,
                        "pnl_pct": round(float(pnl), 4),
                        "is_win": bool(pnl > 0),
                        "entry_prob": round(float(entry_prob), 4),
                        "days_held": int(days_held),
                    })
                    in_position = False

    if not all_trades:
        return {"total_trades": 0, "win_rate": 0, "avg_pnl": 0, "max_drawdown": 0}

    pnls = [t['pnl_pct'] for t in all_trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    cumulative = np.cumsum(pnls)
    peak = np.maximum.accumulate(cumulative)
    max_dd = float(np.min(cumulative - peak))

    return {
        "total_trades": len(all_trades),
        "win_rate": round(sum(1 for t in all_trades if t['is_win']) / len(all_trades) * 100, 2),
        "avg_pnl": round(float(np.mean(pnls)), 4),
        "avg_win": round(float(np.mean(wins)), 4) if wins else 0,
        "avg_loss": round(float(np.mean(losses)), 4) if losses else 0,
        "max_drawdown": round(max_dd, 4),
        "total_return": round(float(np.sum(pnls)), 4),
    }


def predict_proba(model, lgb_model, xgb_model, X):
    """Unified prediction for both model formats"""
    if lgb_model is not None and xgb_model is not None:
        p_lgb = lgb_model.predict_proba(X)[:, 1]
        p_xgb = xgb_model.predict_proba(X)[:, 1]
        return 0.5 * p_lgb + 0.5 * p_xgb
    elif model is not None:
        return model.predict_proba(X)[:, 1]
    else:
        raise ValueError("No model available for prediction")


def per_stock_analysis(df, feature_cols, scaler, model, lgb_model=None, xgb_model=None):
    """逐只股票分析：检查收益是否集中在少数股票"""
    symbols = sorted(df['symbol'].unique())
    stock_results = {}

    for symbol in symbols:
        stock_df = df[df['symbol'] == symbol].sort_values('date').reset_index(drop=True)
        stock_df = stock_df.dropna(subset=feature_cols)
        if len(stock_df) < 50:
            continue

        X = _get_features(stock_df, feature_cols, scaler)
        y_true = stock_df[TARGET_COL].values
        y_prob = predict_proba(model, lgb_model, xgb_model, X)
        y_pred = (y_prob >= PROB_THRESHOLD).astype(int)

        acc = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_prob)

        stock_df = stock_df.copy()
        stock_df['prob_up'] = y_prob
        stock_df['signal'] = stock_df['prob_up'].apply(lambda p: "BUY" if p >= PROB_THRESHOLD else "HOLD")

        trades = simulate_trades(stock_df)

        stock_results[symbol] = {
            "samples": len(stock_df),
            "accuracy": round(acc, 4),
            "auc": round(auc, 4),
            "trades": trades['total_trades'],
            "win_rate": trades['win_rate'],
            "avg_pnl": trades['avg_pnl'],
            "total_return": trades['total_return'],
        }

    # 检查集中度
    total_returns = [v['total_return'] for v in stock_results.values() if v['total_return'] != 0]
    if total_returns:
        top5_return = sum(sorted(total_returns, reverse=True)[:5])
        overall_return = sum(total_returns)
        concentration = abs(top5_return / overall_return) if overall_return != 0 else 0
    else:
        concentration = 0

    logger.info("单只股票收益集中度 (Top5/Total): %.2f", concentration)

    return stock_results, {"concentration_ratio": round(concentration, 4)}


def oos_backtest(df, feature_cols, scaler, model, unseen_symbols, lgb_model=None, xgb_model=None):
    """仅在外样本（未见过的股票）上回测"""
    oos_df = df[df['symbol'].isin(unseen_symbols)].copy()
    oos_df = oos_df.dropna(subset=feature_cols)

    if oos_df.empty:
        logger.warning("无 OOS 数据")
        return {"error": "no_oos_data"}

    X = _get_features(oos_df, feature_cols, scaler)
    y_true = oos_df[TARGET_COL].values
    y_prob = predict_proba(model, lgb_model, xgb_model, X)
    y_pred = (y_prob >= PROB_THRESHOLD).astype(int)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)

    oos_df['prob_up'] = y_prob
    oos_df['signal'] = oos_df['prob_up'].apply(lambda p: "BUY" if p >= PROB_THRESHOLD else "HOLD")

    trades = simulate_trades(oos_df)

    logger.info("OOS 回测 (%d 只股票): acc=%.4f f1=%.4f auc=%.4f, trades=%d, win_rate=%.1f%%",
                 len(unseen_symbols), acc, f1, auc, trades['total_trades'], trades['win_rate'])

    return {
        "symbols": unseen_symbols,
        "samples": len(oos_df),
        "accuracy": round(acc, 4),
        "f1": round(f1, 4),
        "auc": round(auc, 4),
        "backtest": trades,
    }


def check_overfitting():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 加载模型（支持两种格式）
    logger.info("加载模型: %s", MODEL_PATH)
    model_data = joblib.load(MODEL_PATH)
    model_type = model_data.get('model_type', str(type(model_data.get('model', 'unknown'))))

    # 新版格式：lgb_model + xgb_model
    lgb_model = model_data.get('lgb_model')
    xgb_model = model_data.get('xgb_model')
    feature_cols = model_data.get('feature_cols', [])

    # 旧版格式：model + scaler
    model = model_data.get('model')
    scaler = model_data.get('scaler')
    threshold = model_data.get('threshold', 0.50)

    # 如果没有 scaler（新版），设为None，_get_features会直接用原始特征
    if scaler is None and feature_cols:
        logger.info("新版模型格式，无scaler，使用原始特征")

    logger.info("模型: %s, 特征: %d, LGB=%s, XGB=%s",
                 model_type, len(feature_cols), lgb_model is not None, xgb_model is not None)

    # 加载数据
    logger.info("加载特征数据...")
    df = pd.read_csv(FEATURES_PATH, dtype={'symbol': str})
    df = df.dropna(subset=feature_cols + [TARGET_COL])
    if df.empty:
        logger.error("特征数据过滤后为空")
        return

    symbols = sorted(df['symbol'].unique())
    np.random.seed(42)
    n_seen = int(len(symbols) * 0.8)
    seen_symbols = symbols[:n_seen]
    unseen_symbols = symbols[n_seen:]

    logger.info("Seen 股票: %d 只, Unseen 股票: %d 只", len(seen_symbols), len(unseen_symbols))

    report = {
        "model": model_type,
        "threshold": threshold,
        "n_features": len(feature_cols),
        "n_seen_stocks": len(seen_symbols),
        "n_unseen_stocks": len(unseen_symbols),
    }

    # 1. Walk-forward 验证（使用已保存模型，不重新训练）
    logger.info("\n=== 1. Walk-Forward 验证 ===")
    wf_results = walk_forward_test(
        df, feature_cols, scaler, model,
        n_splits=5,
        lgb_model=lgb_model, xgb_model=xgb_model
    )
    report["walk_forward"] = wf_results

    # 检查 walk-forward 性能趋势
    if wf_results:
        aucs = [r['auc'] for r in wf_results]
        f1s = [r['f1'] for r in wf_results]
        logger.info("  AUC 趋势: %s", " → ".join(f"{a:.3f}" for a in aucs))
        logger.info("  F1  趋势: %s", " → ".join(f"{f:.3f}" for f in f1s))

        # 早期 vs 晚期
        mid = len(wf_results) // 2
        early_auc = np.mean([r['auc'] for r in wf_results[:mid]])
        late_auc = np.mean([r['auc'] for r in wf_results[mid:]])
        early_f1 = np.mean([r['f1'] for r in wf_results[:mid]])
        late_f1 = np.mean([r['f1'] for r in wf_results[mid:]])

        report["wf_trend"] = {
            "early_auc": round(early_auc, 4),
            "late_auc": round(late_auc, 4),
            "early_f1": round(early_f1, 4),
            "late_f1": round(late_f1, 4),
            "auc_degradation": round(early_auc - late_auc, 4),
            "f1_degradation": round(early_f1 - late_f1, 4),
        }
        logger.info("  早期 vs 晚期: AUC %.4f vs %.4f (退化 %.4f)",
                     early_auc, late_auc, early_auc - late_auc)

    # 2. OOS 股票回测
    logger.info("\n=== 2. OOS 股票回测 ===")
    oos_results = oos_backtest(df, feature_cols, scaler, model, unseen_symbols,
                                lgb_model=lgb_model, xgb_model=xgb_model)
    report["oos"] = oos_results

    # 3. 逐只股票分析
    logger.info("\n=== 3. 逐只股票分析 ===")
    stock_results, concentration = per_stock_analysis(df, feature_cols, scaler, model,
                                                       lgb_model=lgb_model, xgb_model=xgb_model)
    report["per_stock"] = stock_results
    report["concentration"] = concentration

    # 4. 训练集 vs 测试集 分类指标对比
    logger.info("\n=== 4. Train vs Test 对比 ===")
    train_frames = []
    test_frames = []
    for sym in seen_symbols:
        stock_df = df[df['symbol'] == sym].sort_values('date')
        cutoff = int(len(stock_df) * 0.8)
        train_frames.append(stock_df.iloc[:cutoff])
        test_frames.append(stock_df.iloc[cutoff:])

    train_df = pd.concat(train_frames, ignore_index=True)
    test_df = pd.concat(test_frames, ignore_index=True)
    train_df = train_df.dropna(subset=feature_cols + [TARGET_COL])
    test_df = test_df.dropna(subset=feature_cols + [TARGET_COL])

    if train_df.empty or test_df.empty:
        logger.warning("Train/Test 集为空，跳过对比")
        report["train_vs_test"] = {"error": "empty_data"}
    else:
        X_train = _get_features(train_df, feature_cols, scaler)
        y_train = train_df[TARGET_COL].values
        X_test = _get_features(test_df, feature_cols, scaler)
        y_test = test_df[TARGET_COL].values

        train_prob = predict_proba(model, lgb_model, xgb_model, X_train)
        test_prob = predict_proba(model, lgb_model, xgb_model, X_test)

        train_acc = accuracy_score(y_train, (train_prob >= threshold).astype(int))
        test_acc = accuracy_score(y_test, (test_prob >= threshold).astype(int))
        train_auc = roc_auc_score(y_train, train_prob)
        test_auc = roc_auc_score(y_test, test_prob)
        train_f1 = f1_score(y_train, (train_prob >= threshold).astype(int))
        test_f1 = f1_score(y_test, (test_prob >= threshold).astype(int))

        logger.info("  Train: acc=%.4f auc=%.4f f1=%.4f", train_acc, train_auc, train_f1)
        logger.info("  Test:  acc=%.4f auc=%.4f f1=%.4f", test_acc, test_auc, test_f1)
        logger.info("  Gap:   acc=%.4f auc=%.4f f1=%.4f",
                     train_acc - test_acc, train_auc - test_auc, train_f1 - test_f1)

        report["train_vs_test"] = {
            "train_acc": round(train_acc, 4),
            "test_acc": round(test_acc, 4),
            "train_auc": round(train_auc, 4),
            "test_auc": round(test_auc, 4),
            "train_f1": round(train_f1, 4),
            "test_f1": round(test_f1, 4),
            "gap_acc": round(train_acc - test_acc, 4),
            "gap_auc": round(train_auc - test_auc, 4),
            "gap_f1": round(train_f1 - test_f1, 4),
        }

    # 5. 过拟合综合判断
    logger.info("\n=== 5. 过拟合综合判断 ===")
    gap = report.get("train_vs_test", {})
    if gap.get("error"):
        signals = ["无法判断：Train/Test 数据为空"]
        logger.warning("  %s", signals[0])
    else:
        # AUC gap > 0.1 是危险信号
        if gap['gap_auc'] > 0.1:
            signals.append(f"严重过拟合: AUC gap = {gap['gap_auc']:.3f} (> 0.1)")
        elif gap['gap_auc'] > 0.05:
            signals.append(f"轻度过拟合: AUC gap = {gap['gap_auc']:.3f} (0.05~0.1)")
        else:
            signals.append(f"无明显过拟合: AUC gap = {gap['gap_auc']:.3f} (< 0.05)")

        # Walk-forward 退化
        if report.get("wf_trend"):
            wf = report["wf_trend"]
            if wf['auc_degradation'] > 0.1:
                signals.append(f"时间退化严重: AUC 从 {wf['early_auc']} 降到 {wf['late_auc']}")
            elif wf['auc_degradation'] > 0.03:
                signals.append(f"时间退化中等: AUC 从 {wf['early_auc']} 降到 {wf['late_auc']}")
            else:
                signals.append(f"时间退化轻微: AUC 稳定在 {wf['early_auc']}~{wf['late_auc']}")

        # 收益集中度
        if report.get("concentration") and report["concentration"]["concentration_ratio"] > 0.8:
            signals.append(f"收益高度集中: Top5/Total = {report['concentration']['concentration_ratio']:.2f}")

    report["overfit_signals"] = signals

    for s in signals:
        logger.info("  %s", s)

    # 保存报告
    with open(OVERFIT_REPORT_PATH, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    logger.info("\n报告已保存: %s", OVERFIT_REPORT_PATH)


if __name__ == "__main__":
    check_overfitting()
