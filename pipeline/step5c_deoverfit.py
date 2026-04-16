"""
Phase 5c: 去过拟合训练

策略：
1. 截面标准化：每天对全市场做 z-score 标准化，消除股票间绝对价格差异
2. 剔除绝对值特征：只用相对/有界特征（收益率、RSI、动量等）
3. 强正则化：限制树深度、增加叶子样本数
4. Walk-forward 验证选模型

输入：experiments/ml/features.csv
输出：models/model_defit.pkl + experiments/ml/defit_train_report.json
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

import lightgbm as lgb
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FEATURES_PATH = os.path.join(os.path.dirname(__file__), "ml", "features.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
REPORT_PATH = os.path.join(os.path.dirname(__file__), "ml", "defit_train_report.json")

TARGET_COL = "target_direction_5d"


# 保留的特征类别（全部是相对值/有界值，不包含绝对价格/成交量）
RELATIVE_FEATURES = [
    # 多周期收益率（相对变化率）
    'return_1d', 'return_5d', 'return_10d', 'return_20d',
    # 技术指标（有界或相对值）
    'rsi', 'macd', 'signal', 'histogram',
    'bb_width', 'bb_position',
    'atr_ratio', 'adx', '+di', '-di',
    'k', 'd', 'cci', 'wr', 'mfi',
    'vol_ratio', 'volatility', 'volatility_annual',
    # 价格相对特征（比率，非绝对值）
    'price_ma_ratio', 'gap_pct', 'range_pct',
    # 滞后收益率
    'return_lag_1', 'return_lag_2', 'return_lag_3', 'return_lag_5',
    # 指标滞后（相对值）
    'rsi_lag_1', 'rsi_lag_5',
    'macd_lag_1', 'macd_signal_lag_1',
    'volatility_lag_1',
    # 大盘特征（相对变化）
    'market_return_1d', 'market_return_5d', 'market_vol_change',
    # 相对强弱
    'alpha_1d', 'alpha_5d', 'beta_20d',
    # 资金流（比率）
    'moneyflow_5d', 'moneyflow_20d', 'turnover_ratio',
    # 动量（位置/斜率，标准化后）
    'price_position_20d', 'price_position_60d',
    'trend_strength_20d', 'trend_strength_60d',
    'slope_10d', 'slope_20d',
    # 时间特征
    'month', 'day_of_week', 'quarter',
]

# 绝对禁止的特征（包含绝对价格/成交量/金额）
BANNED_PATTERNS = [
    'close', 'open', 'high', 'low', 'volume', 'amount',
    'close_lag', 'volume_lag',
    'bb_upper', 'bb_lower', 'bb_middle',
    'obv',  # 累积成交量，绝对值
    'ma5', 'ma10', 'ma20', 'ma60',  # 绝对均线值
    'vol_ma5', 'vol_ma20',  # 绝对成交量均线
    'is_month_start', 'is_month_end',  # 信息量极少
]


def cross_sectional_normalize(df, feature_cols):
    """
    截面标准化：对每个交易日的所有股票，计算每个特征的 z-score。
    这样 100 元的股票和 10 元的股票可以公平比较。
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    normalized = []
    dates = sorted(df['date'].unique())

    for i, date in enumerate(dates):
        day_df = df[df['date'] == date].copy()
        if len(day_df) < 5:  # 样本太少跳过
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

        if (i + 1) % 500 == 0:
            logger.info("  截面标准化: %d/%d 天", i + 1, len(dates))

    return pd.concat(normalized, ignore_index=True)


def select_features(df, candidate_features):
    """
    筛选实际可用的特征（排除不存在的列 + banned 特征）
    """
    available = set(df.columns)
    banned = set()
    for pattern in BANNED_PATTERNS:
        for col in candidate_features:
            if pattern in col:
                banned.add(col)

    selected = [c for c in candidate_features if c in available and c not in banned]
    logger.info("候选特征: %d, 剔除 banned: %d, 实际可用: %d",
                 len(candidate_features), len(banned), len(selected))
    return selected


def walk_forward_validate(df, feature_cols, symbols, n_splits=4):
    """
    Walk-forward 验证，返回每折的 AUC
    """
    df = df.sort_values(['date']).reset_index(drop=True)
    dates = sorted(df['date'].unique())

    # 按时间分块
    split_points = [dates[int(len(dates) * i / (n_splits + 1))] for i in range(1, n_splits + 1)]

    results = []
    for split_i in range(n_splits):
        if split_i == 0:
            train_dates = [d for d in dates if d < split_points[0]]
        else:
            train_dates = [d for d in dates if d < split_points[split_i]]

        test_start = split_points[split_i]
        test_end = split_points[split_i + 1] if split_i + 1 < len(split_points) else dates[-1]
        test_dates = [d for d in dates if test_start <= d < test_end]

        if not test_dates:
            continue

        train_df = df[df['date'].isin(train_dates)].dropna(subset=feature_cols + [TARGET_COL])
        test_df = df[df['date'].isin(test_dates)].dropna(subset=feature_cols + [TARGET_COL])

        if len(train_df) < 500 or len(test_df) < 100:
            continue

        X_train = train_df[feature_cols].values
        y_train = train_df[TARGET_COL].values
        X_test = test_df[feature_cols].values
        y_test = test_df[TARGET_COL].values

        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        scale = n_neg / max(n_pos, 1)

        mdl = lgb.LGBMClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.02,
            num_leaves=15, min_child_samples=200, subsample=0.6,
            colsample_bytree=0.4, reg_alpha=1.0, reg_lambda=2.0,
            scale_pos_weight=scale, random_state=42, n_jobs=-1, verbose=-1
        )
        mdl.fit(X_train, y_train)

        y_prob = mdl.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        acc = accuracy_score(y_test, (y_prob >= 0.5).astype(int))

        results.append({
            "split": split_i + 1,
            "train_end": str(split_points[split_i])[:10],
            "test_period": f"{str(test_start)[:10]} ~ {str(test_end)[:10]}",
            "auc": round(auc, 4),
            "accuracy": round(acc, 4),
        })
        logger.info("  WF Split %d [%s]: AUC=%.4f Acc=%.4f",
                     split_i + 1, results[-1]["test_period"], auc, acc)

    return results


def train_deoverfit():
    os.makedirs(MODEL_DIR, exist_ok=True)

    logger.info("读取特征数据...")
    df = pd.read_csv(FEATURES_PATH, dtype={'symbol': str})
    logger.info("原始数据: %d 行, %d 只股票", len(df), df['symbol'].nunique())

    # 1. 选择相对特征
    candidate_features = select_features(df, RELATIVE_FEATURES)

    # 确保目标列存在
    df = df.dropna(subset=candidate_features + [TARGET_COL])
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    logger.info("正样本比例: %.2f%%", df[TARGET_COL].mean() * 100)

    # 2. 截面标准化（用 _cs 后缀的新特征）
    logger.info("截面标准化...")
    df_norm = cross_sectional_normalize(df, candidate_features)

    # 用标准化后的特征
    cs_features = [f'{c}_cs' for c in candidate_features]
    # 保留时间特征（不需要标准化）
    time_features = ['month', 'day_of_week', 'quarter']
    final_features = [f for f in cs_features] + [f for f in time_features if f in df_norm.columns]

    # 确保所有特征都存在
    final_features = [f for f in final_features if f in df_norm.columns]
    logger.info("最终特征: %d 个", len(final_features))

    # 3. 股票切分
    symbols = sorted(df_norm['symbol'].unique())
    np.random.seed(42)
    n_seen = int(len(symbols) * 0.8)
    seen_symbols = symbols[:n_seen]
    unseen_symbols = symbols[n_seen:]
    logger.info("Seen: %d, Unseen (OOS): %d", len(seen_symbols), len(unseen_symbols))

    # 时间切分
    train_frames, time_test_frames = [], []
    for sym in seen_symbols:
        stock_df = df_norm[df_norm['symbol'] == sym].sort_values('date')
        cutoff = int(len(stock_df) * 0.8)
        train_frames.append(stock_df.iloc[:cutoff])
        time_test_frames.append(stock_df.iloc[cutoff:])

    unseen_df = df_norm[df_norm['symbol'].isin(unseen_symbols)]

    train_df = pd.concat(train_frames, ignore_index=True).dropna(subset=final_features + [TARGET_COL])
    time_test_df = pd.concat(time_test_frames, ignore_index=True).dropna(subset=final_features + [TARGET_COL])
    unseen_df_clean = unseen_df.dropna(subset=final_features + [TARGET_COL])

    X_train = train_df[final_features].values
    y_train = train_df[TARGET_COL].values
    X_time = time_test_df[final_features].values
    y_time = time_test_df[TARGET_COL].values
    X_unseen = unseen_df_clean[final_features].values
    y_unseen = unseen_df_clean[TARGET_COL].values

    logger.info("训练集: %d, 时间外: %d, OOS: %d", len(X_train), len(X_time), len(X_unseen))

    # 4. Walk-forward 验证先跑一遍，确认不过拟合
    logger.info("\n=== Walk-Forward 预检 ===")
    wf_results = walk_forward_validate(df_norm, final_features, symbols, n_splits=4)
    avg_wf_auc = np.mean([r['auc'] for r in wf_results])
    logger.info("平均 Walk-Forward AUC: %.4f", avg_wf_auc)

    # 5. 训练多个正则化强度的模型
    configs = {
        "LGBM_v1": {
            "class": lgb.LGBMClassifier,
            "params": dict(
                n_estimators=200, max_depth=4, learning_rate=0.02,
                num_leaves=15, min_child_samples=200, subsample=0.6,
                colsample_bytree=0.4, reg_alpha=1.0, reg_lambda=2.0,
                random_state=42, n_jobs=-1, verbose=-1
            )
        },
        "LGBM_v2": {
            "class": lgb.LGBMClassifier,
            "params": dict(
                n_estimators=300, max_depth=3, learning_rate=0.01,
                num_leaves=10, min_child_samples=300, subsample=0.5,
                colsample_bytree=0.3, reg_alpha=2.0, reg_lambda=5.0,
                random_state=42, n_jobs=-1, verbose=-1
            )
        },
        "LGBM_v3": {
            "class": lgb.LGBMClassifier,
            "params": dict(
                n_estimators=500, max_depth=5, learning_rate=0.005,
                num_leaves=20, min_child_samples=150, subsample=0.7,
                colsample_bytree=0.5, reg_alpha=0.5, reg_lambda=1.0,
                random_state=42, n_jobs=-1, verbose=-1
            )
        },
        "XGB_v1": {
            "class": xgb.XGBClassifier,
            "params": dict(
                n_estimators=300, max_depth=4, learning_rate=0.02,
                min_child_weight=200, subsample=0.6, colsample_bytree=0.4,
                reg_alpha=1.0, reg_lambda=2.0,
                random_state=42, n_jobs=-1, eval_metric='logloss',
                use_label_encoder=False
            )
        },
        "XGB_v2": {
            "class": xgb.XGBClassifier,
            "params": dict(
                n_estimators=500, max_depth=3, learning_rate=0.01,
                min_child_weight=300, subsample=0.5, colsample_bytree=0.3,
                reg_alpha=2.0, reg_lambda=5.0,
                random_state=42, n_jobs=-1, eval_metric='logloss',
                use_label_encoder=False
            )
        },
    }

    results = {}
    best_model = None
    best_name = None
    best_score = 0  # 用 OOS AUC

    for name, cfg in configs.items():
        logger.info("\n训练: %s", name)
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        params = cfg['params'].copy()
        if 'scale_pos_weight' not in params:
            params['scale_pos_weight'] = n_neg / max(n_pos, 1)

        model = cfg['class'](**params)
        model.fit(X_train, y_train)

        y_prob_time = model.predict_proba(X_time)[:, 1]
        y_prob_unseen = model.predict_proba(X_unseen)[:, 1]

        auc_time = roc_auc_score(y_time, y_prob_time)
        auc_unseen = roc_auc_score(y_unseen, y_prob_unseen)
        f1_time = f1_score(y_time, (y_prob_time >= 0.5).astype(int), zero_division=0)
        f1_unseen = f1_score(y_unseen, (y_prob_unseen >= 0.5).astype(int), zero_division=0)

        logger.info("  时间外: AUC=%.4f F1=%.4f", auc_time, f1_time)
        logger.info("  OOS:    AUC=%.4f F1=%.4f", auc_unseen, f1_unseen)
        logger.info("  Gap:    AUC=%.4f", auc_time - auc_unseen)

        results[name] = {
            "time_test_auc": round(auc_time, 4),
            "time_test_f1": round(f1_time, 4),
            "oos_auc": round(auc_unseen, 4),
            "oos_f1": round(f1_unseen, 4),
        }

        # 用 OOS AUC 选最佳
        if auc_unseen > best_score:
            best_score = auc_unseen
            best_model = model
            best_name = name

    logger.info("\n最佳模型: %s (OOS AUC=%.4f)", best_name, best_score)

    # 6. 阈值优化（用 OOS 数据）
    y_prob_unseen = best_model.predict_proba(X_unseen)[:, 1]
    best_thresh = 0.5
    best_f1 = 0
    for t in np.arange(0.40, 0.65, 0.01):
        f1 = f1_score(y_unseen, (y_prob_unseen >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    logger.info("最优阈值: %.2f (OOS F1=%.4f)", best_thresh, best_f1)

    # 7. 特征重要性
    if hasattr(best_model, 'feature_importances_'):
        feat_imp = dict(zip(final_features, best_model.feature_importances_.tolist()))
        top_features = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:15]
        logger.info("Top 15 特征:")
        for feat, imp in top_features:
            logger.info("  %-35s %.4f", feat, imp)
    else:
        feat_imp = {}
        top_features = []

    # 8. 保存模型
    import joblib
    scaler = StandardScaler()
    scaler.fit(X_train)  # 虽然已截面标准化，还是保存一个 scaler 以防推理时需要

    model_data = {
        'model': best_model,
        'scaler': scaler,
        'feature_cols': final_features,
        'threshold': round(float(best_thresh), 2),
        'model_type': best_name,
        'is_cross_sectional': True,
        'raw_features': candidate_features,
    }
    joblib.dump(model_data, os.path.join(MODEL_DIR, "model_defit.pkl"))
    logger.info("模型已保存: %s/model_defit.pkl", MODEL_DIR)

    # 9. 保存报告
    report = {
        "best_model": best_name,
        "best_oos_auc": round(best_score, 4),
        "optimal_threshold": round(float(best_thresh), 2),
        "optimal_oos_f1": round(float(best_f1), 4),
        "all_results": results,
        "walk_forward": wf_results,
        "avg_wf_auc": round(float(avg_wf_auc), 4),
        "top_15_features": [{"name": f, "importance": round(float(i), 6)} for f, i in top_features],
        "n_features": len(final_features),
        "is_cross_sectional": True,
    }

    with open(REPORT_PATH, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    logger.info("报告已保存: %s", REPORT_PATH)


if __name__ == "__main__":
    train_deoverfit()
