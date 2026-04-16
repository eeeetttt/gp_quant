"""
Phase 5e v4: 多周期混合模型 + 阈值目标

Y 定义：个股独立信号（非截面排名）
- 买入：未来 N 天收益 >= 4%
- 卖出：未来 N 天收益 <= -5%
- 中性：-5% ~ 4% 之间（训练时排除）

模型：
- 主模型：10d、20d（样本充足）
- 辅助：5d（样本偏少但可作为预警）
- Hybrid：加权投票 + 信号确认

输入：experiments/ml/features.csv
配置：pipeline/config.yaml
"""
import sys, os, logging, json, warnings, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
import optuna
import joblib, lightgbm as lgb, xgboost as xgb
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── 路径 ──
FEATURES_PATH = os.path.join(os.path.dirname(__file__), "ml", "features.csv")
SECTOR_PATH = os.path.join(os.path.dirname(__file__), "ml", "sector_features.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
REPORT_PATH = os.path.join(os.path.dirname(__file__), "ml", "all_report.json")
BACKTEST_PATH = os.path.join(os.path.dirname(__file__), "results", "all_backtest.json")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

# ── 加载 YAML 配置 ──
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        CFG = yaml.safe_load(f)
    logger.info("配置已加载: %s", CONFIG_PATH)
else:
    CFG = {}
    logger.warning("未找到配置文件 %s，使用硬编码默认值", CONFIG_PATH)

POOL_EXPECTED_STOCKS = CFG.get("pool_expected_stocks", {
    "small": 30, "500": 400, "full": 4000,
})

# 模块级默认参数（供 import 时使用，CLI 解析在 main() 中覆盖）
bt_cfg = CFG.get("backtest", {})
args = argparse.Namespace(
    pool="small", n_trials=30,
    stop_loss=bt_cfg.get("stop_loss", -0.05),
    take_profit=bt_cfg.get("take_profit", 0.04),
    max_hold=bt_cfg.get("max_hold_days", 20),
    max_positions=bt_cfg.get("max_positions", 5),
    commission=bt_cfg.get("commission", 0.001),
    stamp_duty=bt_cfg.get("stamp_duty", 0.001),
)

# ── 目标定义（从 YAML） ──
target_cfg = CFG.get("target", {})
BUY_THRESHOLD = target_cfg.get("buy_threshold", 4.0)
SELL_THRESHOLD = target_cfg.get("sell_threshold", -5.0)

# ── 模型配置（从 YAML） ──
models_cfg = CFG.get("models", {})
PRIMARY_MODELS = models_cfg.get("primary", {})
AUXILIARY_MODELS = models_cfg.get("auxiliary", {"short_5d": 5})
ALL_MODELS = {**PRIMARY_MODELS, **AUXILIARY_MODELS}

# 各模型 trial 数（从 YAML）
MODEL_TRIALS = CFG.get("model_trials", {"mid_10d": 20, "mid_20d": 20, "short_5d": 100})

# ── 去冗余特征（从 YAML） ──
DUPLICATE_FEATURES = set(CFG.get("duplicate_features", [
    'wr', 'volatility_annual', 'market_return_1d', 'market_return_5d'
]))

# ── 超参搜索空间（从 YAML） ──
HYPERPARAM_SPACE = CFG.get("hyperparams", {})
DEFAULT_PARAMS = CFG.get("default_params", {
    'n_estimators': 300, 'max_depth': 5, 'learning_rate': 0.05,
    'num_leaves': 31, 'min_child_samples': 100, 'subsample': 0.8,
    'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 1.0,
})

BOUNDED_FEATURES = [
    'return_1d', 'return_5d', 'return_10d', 'return_20d',
    'rsi', 'macd', 'signal', 'histogram',
    'bb_width', 'bb_position',
    'atr_ratio', 'adx', '+di', '-di',
    'k', 'd', 'cci', 'mfi',
    'vol_ratio', 'volatility', 'volatility_5d', 'volatility_10d',
    'price_ma_ratio', 'price_bb_position', 'gap_pct', 'range_pct',
    'return_lag_1', 'return_lag_2', 'return_lag_3', 'return_lag_5',
    'rsi_lag_1', 'rsi_lag_5',
    'macd_lag_1', 'macd_signal_lag_1',
    'volatility_lag_1',
    'market_vol_change', 'alpha_1d', 'alpha_5d', 'beta_20d',
    'moneyflow_ratio', 'moneyflow_5d', 'moneyflow_20d',
    'vol_up', 'vol_down', 'turnover_ratio',
    'price_position_20d', 'price_position_60d',
    'trend_strength_20d', 'trend_strength_60d',
    'slope_10d', 'slope_20d',
    'month', 'day_of_week', 'quarter',
    'rank_rsi', 'rank_return_5d', 'rank_vol_ratio', 'rank_turnover', 'rank_momentum',
    # 板块特征
    'sector_return_1d', 'sector_return_5d', 'sector_momentum_20d',
    'sector_volatility_20d', 'sector_alpha_1d', 'sector_rank_in_industry',
    'sector_breadth_5d',
]


def select_features(feature_list):
    """剔除高相关重复特征"""
    return [f for f in feature_list if f not in DUPLICATE_FEATURES]


# ── 目标计算 ──
def compute_targets(df, horizons, buy_thresh=BUY_THRESHOLD, sell_thresh=SELL_THRESHOLD):
    """计算多周期三分类目标"""
    for h in horizons:
        future_ret = (df.groupby('symbol')['close'].transform(
            lambda x: (x.shift(-h) - x) / x * 100
        ))
        df[f'target_{h}d'] = np.where(
            future_ret >= buy_thresh, 1,
            np.where(future_ret <= sell_thresh, 0, -1)
        )
    return df


def compute_target_distribution(df, horizons):
    """打印目标分布"""
    dist = {}
    for h in horizons:
        col = f'target_{h}d'
        if col not in df.columns:
            continue
        total = df[col].notna().sum()
        n_buy = (df[col] == 1).sum()
        n_sell = (df[col] == 0).sum()
        n_neutral = (df[col] == -1).sum()
        dist[f'{h}d'] = {
            "buy_pct": round(n_buy / total * 100, 2),
            "sell_pct": round(n_sell / total * 100, 2),
            "neutral_pct": round(n_neutral / total * 100, 2),
            "buy_count": int(n_buy), "sell_count": int(n_sell),
            "neutral_count": int(n_neutral), "total": int(total),
        }
        logger.info("  %d d: 买入 %d (%.1f%%), 卖出 %d (%.1f%%), 中性 %d (%.1f%%)",
                     h, n_buy, n_buy/total*100, n_sell, n_sell/total*100,
                     n_neutral, n_neutral/total*100)
    return dist


# ── 超参优化 ──
def _suggest_param(trial, name, spec):
    """根据 YAML 规格生成单个超参"""
    ptype = spec.get("type", "float")
    if ptype == "int":
        return trial.suggest_int(name, spec["min"], spec["max"], step=spec.get("step", 1))
    elif ptype == "float":
        return trial.suggest_float(name, spec["min"], spec["max"], log=spec.get("log", False))
    else:
        raise ValueError(f"Unknown param type: {ptype}")


def optimize_hyperparams(X_train, y_train, X_val, y_val, n_trials=30):
    """Optuna 超参搜索（在第一个 split 上，搜索空间来自 config.yaml）"""
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos

    def objective(trial):
        params = {}
        for pname, pspec in HYPERPARAM_SPACE.items():
            params[pname] = _suggest_param(trial, pname, pspec)
        model = lgb.LGBMClassifier(
            **params, scale_pos_weight=n_neg / max(n_pos, 1),
            random_state=42, n_jobs=-1, verbose=-1
        )
        model.fit(X_train, y_train)
        return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


# ── 集成预测 ──
def predict_ensemble(X_train, y_train, X_test, params):
    """LightGBM + XGBoost 简单平均"""
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos

    lgb_model = lgb.LGBMClassifier(
        **params, scale_pos_weight=n_neg / max(n_pos, 1),
        random_state=42, n_jobs=-1, verbose=-1
    )
    xgb_model = xgb.XGBClassifier(
        max_depth=params['max_depth'], learning_rate=params['learning_rate'],
        n_estimators=params['n_estimators'], min_child_weight=params.get('min_child_samples', 200),
        subsample=params['subsample'], colsample_bytree=params['colsample_bytree'],
        reg_alpha=params['reg_alpha'], reg_lambda=params['reg_lambda'],
        random_state=42, n_jobs=-1, eval_metric='logloss', tree_method='hist', verbosity=1
    )

    logger.info("  [ensemble] LGB training (%d samples, CPU OpenMP) ...", len(y_train))
    lgb_model.fit(X_train, y_train)
    logger.info("  [ensemble] XGB training (%d samples, Metal GPU) ...", len(y_train))
    xgb_model.fit(X_train, y_train)
    logger.info("  [ensemble] XGB eval_metric=logloss, device=metal")

    p_lgb = lgb_model.predict_proba(X_test)[:, 1]
    p_xgb = xgb_model.predict_proba(X_test)[:, 1]

    return 0.5 * p_lgb + 0.5 * p_xgb, lgb_model, xgb_model


# ── Walk-forward ──
def walk_forward_eval(df, feature_cols, target_col, n_splits=4, name="",
                      params=None, n_trials=30, use_ensemble=False):
    """Walk-forward 验证 + 超参搜索"""
    dates = sorted(df['date'].unique())
    split_size = len(dates) // (n_splits + 1)
    results = []
    models_saved = []
    all_test_dates = set()  # 收集所有 WF 测试日期，防止最终模型数据泄漏

    # 第一个 split 做超参搜索
    if params is None:
        split0_train_end = split_size
        split0_test_start = split_size
        split0_test_end = min(2 * split_size, len(dates))
        tr0 = df[df['date'].isin(dates[:split0_train_end])].dropna(subset=feature_cols + [target_col])
        te0 = df[df['date'].isin(dates[split0_test_start:split0_test_end])].dropna(subset=feature_cols + [target_col])

        if len(tr0) >= 1000 and len(te0) >= 100:
            logger.info("  [%s] Optuna 超参搜索 (%d trials)...", name, n_trials)
            best_params, best_auc = optimize_hyperparams(
                tr0[feature_cols].values, tr0[target_col].values,
                te0[feature_cols].values, te0[target_col].values,
                n_trials=n_trials
            )
            logger.info("  [%s] 最佳 AUC=%.4f, params=%s", name, best_auc, best_params)
            params = best_params
        else:
            # 默认参数（来自 config.yaml）
            params = dict(DEFAULT_PARAMS)

    for split_i in range(n_splits):
        train_end_idx = (split_i + 1) * split_size
        test_start_idx = train_end_idx
        test_end_idx = min((split_i + 2) * split_size, len(dates))
        if test_end_idx <= test_start_idx:
            continue

        train_dates = dates[:train_end_idx]
        test_dates = dates[test_start_idx:test_end_idx]
        all_test_dates.update(test_dates)  # 累积 WF 测试日期

        train_df = df[df['date'].isin(train_dates)].dropna(subset=feature_cols + [target_col])
        test_df = df[df['date'].isin(test_dates)].dropna(subset=feature_cols + [target_col])

        if len(train_df) < 1000 or len(test_df) < 100:
            continue

        X_train, y_train = train_df[feature_cols].values, train_df[target_col].values
        X_test, y_test = test_df[feature_cols].values, test_df[target_col].values

        if use_ensemble:
            y_prob, lgb_m, xgb_m = predict_ensemble(X_train, y_train, X_test, params)
        else:
            n_pos, n_neg = int(y_train.sum()), len(y_train) - int(y_train.sum())
            model = lgb.LGBMClassifier(
                **params, scale_pos_weight=n_neg / max(n_pos, 1),
                random_state=42, n_jobs=-1, verbose=-1
            )
            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_test)[:, 1]
            lgb_m, xgb_m = model, None

        auc = roc_auc_score(y_test, y_prob)
        acc = accuracy_score(y_test, (y_prob >= 0.5).astype(int))
        f1 = f1_score(y_test, (y_prob >= 0.5).astype(int), zero_division=0)
        prec = precision_score(y_test, (y_prob >= 0.5).astype(int), zero_division=0)
        rec = recall_score(y_test, (y_prob >= 0.5).astype(int), zero_division=0)

        results.append({
            "split": split_i + 1,
            "test_period": f"{str(test_dates[0])[:10]} ~ {str(test_dates[-1])[:10]}",
            "auc": round(auc, 4), "accuracy": round(acc, 4),
            "f1": round(f1, 4), "precision": round(prec, 4), "recall": round(rec, 4),
        })
        logger.info("  %s WF-%d [%s]: AUC=%.4f Acc=%.4f F1=%.4f P=%.4f R=%.4f",
                     name, split_i + 1, results[-1]["test_period"], auc, acc, f1, prec, rec)

        models_saved.append((lgb_m, xgb_m))

    avg_auc = np.mean([r['auc'] for r in results]) if results else 0.0
    return results, avg_auc, params, models_saved, all_test_dates


# ── Hybrid 集成 ──
def hybrid_evaluate(df_dict, feature_cols, model_dicts, method="weighted"):
    """
    多周期 Hybrid 集成
    df_dict: {model_name: df_with_target}
    model_dicts: {model_name: [model_pairs_per_split]}
    method: "weighted" | "vote" | "confirm"
    """
    # 取所有模型共同的测试日期范围
    all_results = {}
    for name, (wf_res, avg_auc) in model_dicts.items():
        all_results[name] = {"wf": wf_res, "avg_auc": avg_auc}

    if method == "weighted":
        # 按 AUC 分配权重
        total_auc = sum(r['avg_auc'] for r in all_results.values())
        weights = {k: v['avg_auc'] / total_auc for k, v in all_results.items()}
        logger.info("  加权权重: %s", {k: round(v, 3) for k, v in weights.items()})
        return {"method": "weighted", "weights": weights,
                "combined_auc": round(sum(v['avg_auc'] * w for v, w in zip(all_results.values(), weights.values())), 4)}

    elif method == "vote":
        avg_aucs = [r['avg_auc'] for r in all_results.values()]
        logger.info("  简单投票, 各模型 AUC: %s", [round(a, 4) for a in avg_aucs])
        return {"method": "vote", "avg_auc": round(np.mean(avg_aucs), 4)}

    elif method == "confirm":
        # 双模型确认：只有 10d 和 20d 同时看多才买入
        logger.info("  信号确认模式")
        return {"method": "confirm"}

    return {}


# ── 回测（多模型信号） ──
def run_backtest_multi(df, feature_cols, models_dict, stop_loss=-0.05, take_profit=0.04,
                       max_hold_days=20, max_positions=5, hybrid_weights=None,
                       commission=0.001, stamp_duty=0.001):
    """
    多模型混合回测（含交易成本）
    models_dict: {name: [lgb_model, xgb_model_or_None]}
    hybrid_weights: {name: weight} 或 None（用单模型）
    commission: 单边佣金比例（默认 0.1%）
    stamp_duty: 卖出印花税比例（默认 0.1%）
    """
    dates = sorted(df['date'].unique())
    capital = 100000.0
    initial_capital = capital
    positions = {}
    all_trades = []
    equity_curve = []
    total_commission_paid = 0.0  # 累计交易成本

    # 预计算所有模型的概率
    df = df.copy()
    for name, (lgb_m, xgb_m) in models_dict.items():
        df[f'prob_{name}'] = np.nan
        valid = df[feature_cols].notna().all(axis=1)
        if xgb_m is not None:
            p_lgb = lgb_m.predict_proba(df.loc[valid, feature_cols].values)[:, 1]
            p_xgb = xgb_m.predict_proba(df.loc[valid, feature_cols].values)[:, 1]
            df.loc[valid, f'prob_{name}'] = 0.5 * p_lgb + 0.5 * p_xgb
        else:
            df.loc[valid, f'prob_{name}'] = lgb_m.predict_proba(df.loc[valid, feature_cols].values)[:, 1]

    # 如果有 hybrid 权重，计算综合概率
    if hybrid_weights:
        df['prob_combined'] = 0.0
        for name, w in hybrid_weights.items():
            col = f'prob_{name}'
            if col in df.columns:
                df['prob_combined'] += w * df[col]

    score_col = 'prob_combined' if hybrid_weights else 'prob_mid_20d'

    for date_idx, date in enumerate(dates):
        day_df = df[df['date'] == date].copy()
        if day_df.empty:
            equity_curve.append(capital)
            continue

        # 平仓检查
        for sym in list(positions.keys()):
            if sym not in day_df['symbol'].values:
                continue
            row = day_df[day_df['symbol'] == sym].iloc[0]
            current_price = row['close']
            pos = positions[sym]
            pnl = (current_price - pos['entry_price']) / pos['entry_price']
            days_held = date_idx - pos['entry_day_idx']

            exit_reason = None
            if pnl <= stop_loss: exit_reason = 'stop_loss'
            elif pnl >= take_profit: exit_reason = 'take_profit'
            elif days_held >= max_hold_days: exit_reason = 'max_hold'
            else:
                prob = row.get(score_col, 0.5)
                if not np.isnan(prob) and prob < 0.45:
                    exit_reason = 'signal_down'

            if exit_reason:
                sell_value = pos['shares'] * current_price
                # 卖出成本：佣金 + 印花税
                sell_cost = sell_value * (commission + stamp_duty)
                capital += sell_value - sell_cost
                total_commission_paid += sell_cost
                all_trades.append({
                    'symbol': sym, 'entry_date': str(pos['entry_date'])[:10],
                    'exit_date': str(date)[:10],
                    'entry_price': round(pos['entry_price'], 4),
                    'exit_price': round(current_price, 4),
                    'pnl_pct': round(pnl * 100, 4), 'days_held': days_held,
                    'exit_reason': exit_reason,
                    'commission_cost': round(sell_cost, 2),
                })
                del positions[sym]

        # 买入信号
        if len(positions) < max_positions:
            day_df_sorted = day_df[day_df[score_col].notna()].sort_values(score_col, ascending=False)
            for _, row in day_df_sorted.iterrows():
                if len(positions) >= max_positions:
                    break
                if row[score_col] > 0.55 and row['symbol'] not in positions:
                    price = row['close']
                    shares = int(capital * 0.1 / price / 100) * 100
                    if shares > 0:
                        buy_cost = shares * price * commission  # 买入只收佣金
                        capital -= shares * price + buy_cost
                        total_commission_paid += buy_cost
                        positions[row['symbol']] = {
                            'entry_date': date, 'entry_price': price,
                            'entry_day_idx': date_idx, 'shares': shares,
                        }

        # 权益
        total = capital
        for sym, pos in positions.items():
            if sym in day_df['symbol'].values:
                total += pos['shares'] * day_df[day_df['symbol'] == sym].iloc[0]['close']
        equity_curve.append(total)

    if not all_trades:
        return {"total_trades": 0, "error": "no trades"}

    pnls = [t['pnl_pct'] for t in all_trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    equity_arr = np.array(equity_curve)
    peak = np.maximum.accumulate(equity_arr)
    max_dd = float(np.min((equity_arr - peak) / peak * 100))

    return {
        "total_trades": len(all_trades),
        "win_rate": round(sum(1 for p in pnls if p > 0) / len(pnls) * 100, 2),
        "avg_pnl": round(float(np.mean(pnls)), 4),
        "avg_win": round(float(np.mean(wins)), 4) if wins else 0,
        "avg_loss": round(float(np.mean(losses)), 4) if losses else 0,
        "profit_factor": round(abs(sum(wins) / sum(losses)), 4) if sum(losses) != 0 else float('inf'),
        "total_return": round((equity_curve[-1] - initial_capital) / initial_capital * 100, 4),
        "max_drawdown": round(max_dd, 4),
        "final_equity": round(equity_curve[-1], 2),
        "total_cost_paid": round(total_commission_paid, 2),
        "cost_as_pct_of_capital": round(total_commission_paid / initial_capital * 100, 4),
        "stop_loss_hits": sum(1 for t in all_trades if t['exit_reason'] == 'stop_loss'),
        "take_profit_hits": sum(1 for t in all_trades if t['exit_reason'] == 'take_profit'),
        "max_hold_hits": sum(1 for t in all_trades if t['exit_reason'] == 'max_hold'),
        "signal_down_hits": sum(1 for t in all_trades if t['exit_reason'] == 'signal_down'),
        "avg_days_held": round(float(np.mean([t['days_held'] for t in all_trades])), 2),
        "cost_params": {"commission": commission, "stamp_duty": stamp_duty},
        "sample_trades": all_trades[:20],
    }


def main():
    global args
    parser = argparse.ArgumentParser(description="多周期混合模型训练")
    parser.add_argument("--pool", choices=["small", "500", "full"], default="small")
    parser.add_argument("--n-trials", type=int, default=30, help="Optuna 超参搜索 trial 数")
    parser.add_argument("--stop-loss", type=float, default=-0.05, help="止损阈值（如 -0.05 = -5%）")
    parser.add_argument("--take-profit", type=float, default=0.04, help="止盈阈值")
    parser.add_argument("--max-hold", type=int, default=20, help="最大持有天数")
    parser.add_argument("--max-positions", type=int, default=5, help="最大同时持仓数")
    parser.add_argument("--commission", type=float, default=0.001, help="单边佣金比例（默认 0.1%）")
    parser.add_argument("--stamp-duty", type=float, default=0.001, help="卖出印花税比例（默认 0.1%）")
    args = parser.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(BACKTEST_PATH), exist_ok=True)

    logger.info("读取特征数据...")
    logger.info("  硬件配置: CPU=18核(M5 Max) OpenMP, LGB+XGB 全CPU多线程")
    df = pd.read_csv(FEATURES_PATH, dtype={'symbol': str})
    df['date'] = pd.to_datetime(df['date'])
    logger.info("原始: %d 行, %d 股票", len(df), df['symbol'].nunique())

    # 合并板块特征（如果存在）
    if os.path.exists(SECTOR_PATH):
        sector_df = pd.read_csv(SECTOR_PATH, dtype={'symbol': str})
        sector_df['date'] = pd.to_datetime(sector_df['date'])
        sector_cols = [c for c in sector_df.columns if c not in ('symbol', 'date')]
        df = df.merge(sector_df[['symbol', 'date'] + sector_cols], on=['symbol', 'date'], how='left')
        logger.info("  板块特征已合并: %d 个", len(sector_cols))
        # 打印 sector 特征名
        logger.info("  板块列: %s", sector_cols)
    else:
        logger.warning("  未找到板块特征文件 %s，跳过板块特征", SECTOR_PATH)

    # 验证数据与 --pool 匹配
    n_stocks = df['symbol'].nunique()
    expected = POOL_EXPECTED_STOCKS.get(args.pool, 0)
    if expected and n_stocks < expected:
        logger.error("数据不匹配: --pool=%s 期望 >=%d 只，实际 %d 只",
                     args.pool, expected, n_stocks)
        sys.exit(1)
    logger.info("数据验证通过: %d 只股票 (--pool %s)", n_stocks, args.pool)

    # 去冗余特征
    candidate = select_features([f for f in BOUNDED_FEATURES if f in df.columns])
    logger.info("可用特征: %d (已去冗余 %d 个)", len(candidate), len(BOUNDED_FEATURES) - len(candidate))

    # ── Step 1: 计算多周期目标 ──
    logger.info("\n=== Step 1: 多周期目标 (买入>=4%%, 卖出<=-5%%) ===")
    horizons = list(ALL_MODELS.values())
    df = compute_targets(df, horizons)
    report = {
        "training_timestamp": pd.Timestamp.now().isoformat(),
        "data_range": {"start": str(df['date'].min())[:10], "end": str(df['date'].max())[:10]},
        "n_stocks": int(n_stocks), "n_samples": len(df),
        "buy_threshold": BUY_THRESHOLD, "sell_threshold": SELL_THRESHOLD,
        "version": "v5",
        "git_hash": os.popen("git rev-parse --short HEAD 2>/dev/null").read().strip() or "unknown",
    }
    report["target_distribution"] = compute_target_distribution(df, horizons)

    # ── Step 2: 训练各周期模型 ──
    logger.info("\n=== Step 2: 多周期模型训练 ===")
    single_results = {}
    all_models = {}  # {name: (lgb_model, xgb_model)}

    for name, h in ALL_MODELS.items():
        target_col = f'target_{h}d'
        try:
            # 排除中性样本
            df_h = df[df[target_col] != -1].copy()
            df_h = df_h.dropna(subset=candidate + [target_col])

            n_train = len(df_h)
            if n_train < 5000:
                logger.warning("  [%s] 样本不足 (%d), 跳过", name, n_train)
                continue

            pos_ratio = df_h[target_col].mean()
            logger.info("\n  [%s] horizon=%dd, 样本=%d, 正样本比例=%.1f%%", name, h, n_train, pos_ratio * 100)

            # Walk-forward: 先只用 LightGBM（集成在回测阶段用）
            wf_res, avg_auc, best_params, _, wf_test_dates = walk_forward_eval(
                df_h, candidate, target_col, n_splits=4, name=name,
                n_trials=MODEL_TRIALS.get(name, args.n_trials), use_ensemble=False
            )

            single_results[name] = {
                "horizon": h, "n_samples": n_train,
                "positive_ratio": round(pos_ratio, 4),
                "walk_forward": wf_res, "avg_auc": round(avg_auc, 4),
                "best_params": best_params,
            }
            logger.info("  [%s] AUC=%.4f", name, avg_auc)

            # 用全部历史数据训练最终模型（排除 WF 测试日期，防止数据泄漏）
            df_full = df_h[~df_h['date'].isin(wf_test_dates)].sort_values('date')
            if len(df_full) < 5000:
                logger.warning("  [%s] 排除 WF 测试后样本不足 (%d), 使用全部数据", name, len(df_full))
                df_full = df_h.sort_values('date')
            split_pt = int(len(df_full) * 0.8)
            df_tr, df_te = df_full.iloc[:split_pt], df_full.iloc[split_pt:]
            X_tr, y_tr = df_tr[candidate].values, df_tr[target_col].values
            X_te, y_te = df_te[candidate].values, df_te[target_col].values
            n_pos, n_neg = int(y_tr.sum()), len(y_tr) - int(y_tr.sum())

            lgb_m = lgb.LGBMClassifier(
                **best_params, scale_pos_weight=n_neg / max(n_pos, 1),
                random_state=42, n_jobs=-1, verbose=-1
            )
            xgb_m = xgb.XGBClassifier(
                max_depth=best_params['max_depth'], learning_rate=best_params['learning_rate'],
                n_estimators=best_params['n_estimators'],
                min_child_weight=best_params.get('min_child_samples', 200),
                subsample=best_params['subsample'], colsample_bytree=best_params['colsample_bytree'],
                reg_alpha=best_params['reg_alpha'], reg_lambda=best_params['reg_lambda'],
                random_state=42, n_jobs=-1, eval_metric='logloss', tree_method='hist', verbosity=1
            )
            logger.info("  [%s] 最终模型: LGB+XGB CPU OpenMP (%d train, %d test)",
                       name, len(y_tr), len(y_te))
            lgb_m.fit(X_tr, y_tr)
            xgb_m.fit(X_tr, y_tr)
            all_models[name] = (lgb_m, xgb_m)

            # 测试集验证
            p_lgb = lgb_m.predict_proba(X_te)[:, 1]
            p_xgb = xgb_m.predict_proba(X_te)[:, 1]
            p_ens = 0.5 * p_lgb + 0.5 * p_xgb
            ens_auc = roc_auc_score(y_te, p_ens)
            logger.info("  [%s] 集成测试 AUC: LGB=%.4f, XGB=%.4f, ENS=%.4f",
                         name, roc_auc_score(y_te, p_lgb), roc_auc_score(y_te, p_xgb), ens_auc)

        except Exception as e:
            logger.error("  [%s] 训练失败: %s", name, e)
            single_results[name] = {
                "horizon": h, "error": str(e), "walk_forward": [], "avg_auc": 0.0,
            }

    report["single_models"] = single_results

    # ── Feature Importance ──
    logger.info("\n=== Feature Importance ===")
    feature_importance = {}
    for name, (lgb_m, xgb_m) in all_models.items():
        if hasattr(lgb_m, 'feature_importances_'):
            imp_dict = dict(zip(candidate, lgb_m.feature_importances_.tolist()))
            top10 = sorted(imp_dict.items(), key=lambda x: x[1], reverse=True)[:10]
            feature_importance[name] = {
                "top_10": [{"feature": f, "importance": int(v)} for f, v in top10],
                "all": imp_dict,
            }
            logger.info("  [%s] Top 5:", name)
            for f, v in top10[:5]:
                logger.info("    %-25s %d", f, v)
    report["feature_importance"] = feature_importance

    # ── Step 3: Hybrid 集成 ──
    logger.info("\n=== Step 3: Hybrid 集成 ===")

    # 只用主模型做 hybrid，如果没有主模型就用全部可用模型
    primary_results = {k: v for k, v in single_results.items() if k in PRIMARY_MODELS}
    if not primary_results:
        primary_results = single_results

    if len(primary_results) >= 2:
        # A. 加权
        total_auc = sum(r['avg_auc'] for r in primary_results.values())
        weights = {k: round(v['avg_auc'] / total_auc, 4) for k, v in primary_results.items()}
        weighted_auc = sum(r['avg_auc'] * w for r, w in zip(primary_results.values(), weights.values()))
        logger.info("  A. 加权: %s, 综合 AUC=%.4f", weights, weighted_auc)

        # B. 简单平均
        simple_auc = np.mean([r['avg_auc'] for r in primary_results.values()])
        logger.info("  B. 简单平均: AUC=%.4f", simple_auc)

        # C. 信号确认（双模型都看多才买入）
        logger.info("  C. 信号确认: 10d+20d 同时看多")

        report["hybrid"] = {
            "weighted": {"weights": weights, "combined_auc": round(weighted_auc, 4)},
            "simple_avg": {"avg_auc": round(simple_auc, 4)},
            "confirm": {"models": list(PRIMARY_MODELS.keys())},
        }
    else:
        logger.warning("  主模型数量不足，跳过 Hybrid")
        report["hybrid"] = {}

    # ── Step 4: 回测 ──
    logger.info("\n=== Step 4: 回测 ===")

    # 准备回测数据（排除中性）
    # 用已训练模型中样本最多的那个做回测 baseline
    if not all_models:
        logger.warning("  无可用模型，跳过回测")
        report["backtest_single_20d"] = {"total_trades": 0, "error": "no models trained"}
        report["backtest_hybrid_weighted"] = {"total_trades": 0, "error": "no models trained"}
    else:
        # 选样本最多的模型作为 baseline
        baseline_name = max(all_models, key=lambda k: single_results.get(k, {}).get("n_samples", 0))
        h = ALL_MODELS[baseline_name]
        target_col = f'target_{h}d'
        df_bt = df[df[target_col] != -1].copy()
        logger.info("  Baseline 模型: %s (h=%dd)", baseline_name, h)

        # A. 单模型回测
        logger.info("\n  A. 单模型 (%s):", baseline_name)
        bt_single = run_backtest_multi(
            df_bt, candidate, {baseline_name: all_models[baseline_name]},
            stop_loss=args.stop_loss, take_profit=args.take_profit,
            max_hold_days=args.max_hold, max_positions=args.max_positions,
            commission=args.commission, stamp_duty=args.stamp_duty,
        )
        logger.info("  交易: %d 笔, 胜率: %.1f%%, 总收益: %.1f%%, 最大回撤: %.1f%%, 交易成本: %.0f元",
                     bt_single['total_trades'], bt_single['win_rate'],
                     bt_single['total_return'], bt_single['max_drawdown'],
                     bt_single.get('total_cost_paid', 0))
        report[f"backtest_single_{baseline_name}"] = bt_single

    # B. Hybrid 加权回测（只在有多个模型时跑）
    if report.get("hybrid", {}).get("weighted") and all_models:
        # Hybrid 回测用 baseline 的目标列
        if 'df_bt' not in locals():
            baseline_name = max(all_models, key=lambda k: single_results.get(k, {}).get("n_samples", 0))
            h = ALL_MODELS[baseline_name]
            target_col = f'target_{h}d'
            df_bt = df[df[target_col] != -1].copy()
        logger.info("\n  B. Hybrid 加权:")
        bt_hybrid = run_backtest_multi(
            df_bt, candidate, all_models,
            hybrid_weights=report["hybrid"]["weighted"]["weights"],
            stop_loss=args.stop_loss, take_profit=args.take_profit,
            max_hold_days=args.max_hold, max_positions=args.max_positions,
            commission=args.commission, stamp_duty=args.stamp_duty,
        )
        logger.info("  交易: %d 笔, 胜率: %.1f%%, 总收益: %.1f%%, 最大回撤: %.1f%%, 交易成本: %.0f元",
                     bt_hybrid['total_trades'], bt_hybrid['win_rate'],
                     bt_hybrid['total_return'], bt_hybrid['max_drawdown'],
                     bt_hybrid.get('total_cost_paid', 0))
        report["backtest_hybrid_weighted"] = bt_hybrid

    # ── 保存 ──
    # 保存最佳模型 bundle
    best_model_name = max(primary_results, key=lambda k: primary_results[k]['avg_auc']) if primary_results else None
    if best_model_name and best_model_name in all_models:
        lgb_m, xgb_m = all_models[best_model_name]
        model_bundle = {
            'lgb_model': lgb_m, 'xgb_model': xgb_m,
            'feature_cols': candidate,
            'model_type': f'Ensemble_{best_model_name}',
            'horizon': ALL_MODELS[best_model_name],
            'buy_threshold': BUY_THRESHOLD, 'sell_threshold': SELL_THRESHOLD,
            'hybrid_weights': report.get("hybrid", {}).get("weighted", {}).get("weights"),
            # 新增元数据
            'training_timestamp': pd.Timestamp.now().isoformat(),
            'data_range': {"start": str(df['date'].min())[:10], "end": str(df['date'].max())[:10]},
            'n_stocks_used': int(n_stocks),
            'target_distribution': report.get("target_distribution", {}),
            'walk_forward_auc_history': single_results.get(best_model_name, {}).get("walk_forward", []),
            'best_params': single_results.get(best_model_name, {}).get("best_params", {}),
            'git_hash': os.popen("git rev-parse --short HEAD 2>/dev/null").read().strip() or "unknown",
            'version': "v5",
        }
        joblib.dump(model_bundle, os.path.join(MODEL_DIR, "model_all.pkl"))
        logger.info("\n模型已保存: %s/model_all.pkl", MODEL_DIR)
        logger.info("  元数据: %d 只股票, %d 个特征, 训练时间: %s",
                     model_bundle['n_stocks_used'], len(candidate), model_bundle['training_timestamp'][:19])

    with open(REPORT_PATH, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    logger.info("报告已保存: %s", REPORT_PATH)

    # ── 打印汇总 ──
    print("\n" + "=" * 60)
    print("多周期混合模型训练结果")
    print("=" * 60)

    print("\n目标分布 (买入>=4%%, 卖出<=-5%%):")
    for h_name, dist in report.get("target_distribution", {}).items():
        print(f"  {h_name}: 买入 {dist['buy_pct']}%, 卖出 {dist['sell_pct']}%, 中性 {dist['neutral_pct']}%")

    print("\n单模型 AUC:")
    for name, res in single_results.items():
        print(f"  {name} (h={res['horizon']}d): AUC={res['avg_auc']:.4f}, "
              f"样本={res['n_samples']}, 正样本={res['positive_ratio']*100:.1f}%")

    if report.get("hybrid"):
        h = report["hybrid"]
        print("\nHybrid 集成:")
        if "weighted" in h:
            print(f"  加权: {h['weighted']['weights']}, AUC={h['weighted']['combined_auc']:.4f}")
        if "simple_avg" in h:
            print(f"  简单平均: AUC={h['simple_avg']['avg_auc']:.4f}")

    print("\n回测:")
    # 单模型回测（动态 key）
    for key in report:
        if key.startswith("backtest_single_") and not report[key].get("error"):
            bt = report[key]
            model_label = key.replace("backtest_single_", "")
            cost_info = f", 成本 {bt.get('total_cost_paid', 0):.0f}元" if bt.get('total_cost_paid') else ""
            print(f"  单模型({model_label}): {bt['total_trades']}笔, 胜率 {bt['win_rate']}%, "
                  f"收益 {bt['total_return']}%, 回撤 {bt['max_drawdown']}%, "
                  f"盈亏比 {bt['profit_factor']}{cost_info}")
    if "backtest_hybrid_weighted" in report:
        bt = report["backtest_hybrid_weighted"]
        if 'error' not in bt:
            cost_info = f", 成本 {bt.get('total_cost_paid', 0):.0f}元" if bt.get('total_cost_paid') else ""
            print(f"  Hybrid加权:  {bt['total_trades']}笔, 胜率 {bt['win_rate']}%, "
                  f"收益 {bt['total_return']}%, 回撤 {bt['max_drawdown']}%, "
                  f"盈亏比 {bt['profit_factor']}{cost_info}")


if __name__ == "__main__":
    main()
