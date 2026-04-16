"""
Phase 5d: Walk-Forward 集成训练

不用一个模型预测全部时间，而是：
1. 将数据分成多个时间窗口
2. 每个窗口训练一个模型，预测下一个窗口
3. 最终用多个模型的集成（投票）做预测

这样每个模型都在自己熟悉的时间内训练，不会出现时间外退化。
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import logging
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import joblib
import lightgbm as lgb
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FEATURES_PATH = os.path.join(os.path.dirname(__file__), "ml", "features.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
REPORT_PATH = os.path.join(os.path.dirname(__file__), "ml", "wf_train_report.json")

TARGET_COL = "target_direction_5d"


# 相对特征（不包含绝对值）
RELATIVE_FEATURES = [
    'return_1d', 'return_5d', 'return_10d', 'return_20d',
    'rsi', 'macd', 'signal', 'histogram',
    'bb_width', 'bb_position',
    'atr_ratio', 'adx', '+di', '-di',
    'k', 'd', 'cci', 'wr', 'mfi',
    'vol_ratio', 'volatility', 'volatility_annual',
    'price_ma_ratio', 'gap_pct', 'range_pct',
    'return_lag_1', 'return_lag_2', 'return_lag_3', 'return_lag_5',
    'rsi_lag_1', 'rsi_lag_5',
    'macd_lag_1', 'macd_signal_lag_1',
    'volatility_lag_1',
    'market_return_1d', 'market_return_5d', 'market_vol_change',
    'alpha_1d', 'alpha_5d', 'beta_20d',
    'moneyflow_5d', 'moneyflow_20d', 'turnover_ratio',
    'price_position_20d', 'price_position_60d',
    'trend_strength_20d', 'trend_strength_60d',
    'slope_10d', 'slope_20d',
    'month', 'day_of_week', 'quarter',
]


def cross_sectional_normalize(df, feature_cols):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    normalized = []
    dates = sorted(df['date'].unique())

    for i, date in enumerate(dates):
        day_df = df[df['date'] == date].copy()
        if len(day_df) < 5:
            normalized.append(day_df)
            continue

        for col in feature_cols:
            if col in day_df.columns:
                mean = day_df[col].mean()
                std = day_df[col].std()
                if std > 1e-10:
                    day_df[f'{col}_cs'] = (day_df[col] - mean) / std
                else:
                    day_df[f'{col}_cs'] = 0.0

        normalized.append(day_df)

    return pd.concat(normalized, ignore_index=True)


def build_walkforward_models(df, candidate_features, n_windows=5):
    """
    构建多个 walk-forward 模型。
    每个模型用最近 N 年的数据训练，预测下一个窗口。
    最终用所有模型的预测做集成。
    """
    dates = sorted(df['date'].unique())
    total_days = len(dates)
    window_size = total_days // (n_windows + 1)

    models = []
    window_results = []

    for i in range(n_windows):
        # 训练窗口：最近 (i+1) 个窗口的数据
        train_end_idx = (i + 1) * window_size
        # 测试窗口：下一个窗口
        test_start_idx = train_end_idx
        test_end_idx = min((i + 2) * window_size, total_days)

        if test_start_idx >= total_days or test_end_idx <= test_start_idx:
            continue

        train_dates = dates[:train_end_idx]
        test_dates = dates[test_start_idx:test_end_idx]

        train_df = df[df['date'].isin(train_dates)].dropna(
            subset=candidate_features + [TARGET_COL]
        )
        test_df = df[df['date'].isin(test_dates)].dropna(
            subset=candidate_features + [TARGET_COL]
        )

        if len(train_df) < 1000 or len(test_df) < 100:
            continue

        X_train = train_df[candidate_features].values
        y_train = train_df[TARGET_COL].values
        X_test = test_df[candidate_features].values
        y_test = test_df[TARGET_COL].values

        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        scale = n_neg / max(n_pos, 1)

        # 两个模型：LGBM + XGB
        lgb_model = lgb.LGBMClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.02,
            num_leaves=15, min_child_samples=200, subsample=0.6,
            colsample_bytree=0.4, reg_alpha=1.0, reg_lambda=2.0,
            scale_pos_weight=scale, random_state=42, n_jobs=-1, verbose=-1
        )
        xgb_model = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.02,
            min_child_weight=200, subsample=0.6, colsample_bytree=0.4,
            reg_alpha=1.0, reg_lambda=2.0,
            scale_pos_weight=scale, random_state=42, n_jobs=-1,
            eval_metric='logloss'
        )

        lgb_model.fit(X_train, y_train)
        xgb_model.fit(X_train, y_train)

        # 测试
        lgb_prob = lgb_model.predict_proba(X_test)[:, 1]
        xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
        ens_prob = (lgb_prob + xgb_prob) / 2

        lgb_auc = roc_auc_score(y_test, lgb_prob)
        xgb_auc = roc_auc_score(y_test, xgb_prob)
        ens_auc = roc_auc_score(y_test, ens_prob)
        ens_f1 = f1_score(y_test, (ens_prob >= 0.5).astype(int), zero_division=0)

        result = {
            "window": i + 1,
            "train_period": f"{str(train_dates[0])[:10]} ~ {str(train_dates[-1])[:10]}",
            "test_period": f"{str(test_dates[0])[:10]} ~ {str(test_dates[-1])[:10]}",
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "lgb_auc": round(lgb_auc, 4),
            "xgb_auc": round(xgb_auc, 4),
            "ensemble_auc": round(ens_auc, 4),
            "ensemble_f1": round(ens_f1, 4),
        }
        window_results.append(result)
        logger.info("Window %d [%s]: LGB=%.4f XGB=%.4f ENS=%.4f F1=%.4f",
                     i + 1, result["test_period"], lgb_auc, xgb_auc, ens_auc, ens_f1)

        models.append({
            "lgb": lgb_model,
            "xgb": xgb_model,
            "train_end": train_dates[-1],
            "test_start": test_dates[0],
        })

    avg_auc = np.mean([r['ensemble_auc'] for r in window_results])
    logger.info("平均 Walk-Forward AUC: %.4f", avg_auc)

    return models, window_results, avg_auc


def train_walkforward_ensemble():
    os.makedirs(MODEL_DIR, exist_ok=True)

    logger.info("读取特征数据...")
    df = pd.read_csv(FEATURES_PATH, dtype={'symbol': str})
    logger.info("原始: %d 行, %d 股票", len(df), df['symbol'].nunique())

    # 筛选可用特征
    available = set(df.columns)
    candidate = [f for f in RELATIVE_FEATURES if f in available]
    logger.info("可用特征: %d", len(candidate))

    df = df.dropna(subset=candidate + [TARGET_COL])
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    # 截面标准化
    logger.info("截面标准化...")
    df_norm = cross_sectional_normalize(df, candidate)

    cs_features = [f'{c}_cs' for c in candidate]
    time_features = ['month', 'day_of_week', 'quarter']
    final_features = cs_features + [f for f in time_features if f in df_norm.columns]
    final_features = [f for f in final_features if f in df_norm.columns]
    logger.info("最终特征: %d", len(final_features))

    # Walk-forward 训练
    logger.info("\n=== Walk-Forward 模型训练 ===")
    models, window_results, avg_auc = build_walkforward_models(
        df_norm, final_features, n_windows=5
    )

    # 用所有数据训练一个最终模型（用于实时推理）
    logger.info("\n训练最终模型（全量数据，用于实时推理）...")
    df_final = df_norm.dropna(subset=final_features + [TARGET_COL])
    X_final = df_final[final_features].values
    y_final = df_final[TARGET_COL].values
    n_pos = y_final.sum()
    n_neg = len(y_final) - n_pos

    final_lgb = lgb.LGBMClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.02,
        num_leaves=15, min_child_samples=200, subsample=0.6,
        colsample_bytree=0.4, reg_alpha=1.0, reg_lambda=2.0,
        scale_pos_weight=n_neg / max(n_pos, 1),
        random_state=42, n_jobs=-1, verbose=-1
    )
    final_lgb.fit(X_final, y_final)

    # 阈值优化（用 walk-forward 测试集的加权平均）
    all_probs = []
    all_labels = []
    for i, model_dict in enumerate(models):
        # 对每个模型的测试窗口做预测
        test_start = model_dict['test_start']
        test_df = df_norm[df_norm['date'] >= test_start].sort_values('date')
        # 只取测试窗口（避免重叠）
        if i + 1 < len(models):
            next_start = models[i + 1]['test_start']
            test_df = test_df[test_df['date'] < next_start]
        test_df = test_df.dropna(subset=final_features + [TARGET_COL])

        if test_df.empty:
            continue

        X = test_df[final_features].values
        y = test_df[TARGET_COL].values

        prob = (model_dict['lgb'].predict_proba(X)[:, 1] +
                model_dict['xgb'].predict_proba(X)[:, 1]) / 2
        all_probs.append(prob)
        all_labels.append(y)

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    best_thresh = 0.5
    best_f1 = 0
    for t in np.arange(0.40, 0.65, 0.01):
        f1 = f1_score(all_labels, (all_probs >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    logger.info("最优阈值: %.2f (WF F1=%.4f)", best_thresh, best_f1)

    # 保存模型集合
    model_bundle = {
        'final_model': final_lgb,
        'walkforward_models': models,
        'feature_cols': final_features,
        'threshold': round(float(best_thresh), 2),
        'raw_features': candidate,
        'is_cross_sectional': True,
        'model_type': 'WalkForward_LGBM+XGB_Ensemble',
    }
    bundle_path = os.path.join(MODEL_DIR, "model_wf.pkl")
    joblib.dump(model_bundle, bundle_path)
    logger.info("模型已保存: %s", bundle_path)

    # 报告
    report = {
        "model_type": "WalkForward_LGBM+XGB_Ensemble",
        "n_windows": len(models),
        "avg_wf_auc": round(float(avg_auc), 4),
        "optimal_threshold": round(float(best_thresh), 2),
        "optimal_wf_f1": round(float(best_f1), 4),
        "window_results": window_results,
        "n_features": len(final_features),
        "is_cross_sectional": True,
    }

    with open(REPORT_PATH, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    logger.info("报告已保存: %s", REPORT_PATH)


if __name__ == "__main__":
    train_walkforward_ensemble()
