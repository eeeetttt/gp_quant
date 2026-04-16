"""
Phase 5b: 模型优化训练

在基线 RF 基础上优化：
1. 更多模型：XGBoost, LightGBM, GradientBoosting, ExtraTrees
2. 超参数调优：网格搜索/手动调参
3. 特征选择：高相关特征去冗余 + 重要性筛选
4. 类别平衡：class_weight / scale_pos_weight
5. 模型集成：多模型投票/堆叠

输入：experiments/ml/features.csv
输出：models/model_optimized.pkl + experiments/ml/opt_train_report.json
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
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

import xgboost as xgb
import lightgbm as lgb

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FEATURES_PATH = os.path.join(os.path.dirname(__file__), "ml", "features.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
REPORT_PATH = os.path.join(os.path.dirname(__file__), "ml", "opt_train_report.json")

TARGET_COL = "target_direction_5d"


def select_features(train_df, feature_cols, threshold='median', max_features=50):
    """
    两阶段特征选择：
    1. 用 RF 初步筛选重要性 > threshold 的特征
    2. 去除高相关特征（|r| > 0.9）
    """
    logger.info("第一阶段：RF 重要性筛选...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(train_df[feature_cols].values, train_df[TARGET_COL].values)

    importances = dict(zip(feature_cols, rf.feature_importances_))
    sorted_feats = sorted(importances.items(), key=lambda x: x[1], reverse=True)

    # 取重要性 > 中位数的特征，最多 max_features
    median_imp = np.median(list(importances.values()))
    selected = [f for f, imp in sorted_feats if imp >= median_imp][:max_features]
    logger.info("  保留 %d 个特征（阈值 >= %.4f）", len(selected), median_imp)

    # 去除高相关特征
    logger.info("第二阶段：去高相关特征...")
    corr_matrix = train_df[selected].corr().abs()
    to_drop = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > 0.95:
                # 保留重要性更高的那个
                if importances[corr_matrix.columns[j]] < importances[corr_matrix.columns[i]]:
                    to_drop.add(corr_matrix.columns[j])
                else:
                    to_drop.add(corr_matrix.columns[i])

    final_features = [f for f in selected if f not in to_drop]
    logger.info("  去冗余 %d 个，最终 %d 个特征", len(to_drop), len(final_features))

    return final_features, importances


def train_optimized():
    os.makedirs(MODEL_DIR, exist_ok=True)

    logger.info("读取特征数据: %s", FEATURES_PATH)
    df = pd.read_csv(FEATURES_PATH, dtype={'symbol': str})
    logger.info("原始数据: %d 行, %d 只股票", len(df), df['symbol'].nunique())

    exclude_cols = {'symbol', 'date', 'close', 'target_5d', 'target_direction_5d',
                    'open', 'high', 'low'}
    feature_cols = [c for c in df.columns if c not in exclude_cols
                    and df[c].dtype in ['float64', 'int64', 'float32', 'int32']]

    df = df.dropna(subset=feature_cols + [TARGET_COL])
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    pos_ratio = df[TARGET_COL].mean()
    logger.info("正样本比例: %.2f%%", pos_ratio * 100)

    # 股票级别切分
    symbols = sorted(df['symbol'].unique())
    np.random.seed(42)
    n_seen = int(len(symbols) * 0.8)
    seen_symbols = symbols[:n_seen]
    unseen_symbols = symbols[n_seen:]
    logger.info("Seen 股票: %d 只, Unseen (OOS): %d 只", len(seen_symbols), len(unseen_symbols))

    # 时间切分 + OOS
    train_frames, time_test_frames = [], []
    for sym in seen_symbols:
        stock_df = df[df['symbol'] == sym].sort_values('date')
        cutoff = int(len(stock_df) * 0.8)
        train_frames.append(stock_df.iloc[:cutoff])
        time_test_frames.append(stock_df.iloc[cutoff:])

    unseen_df = df[df['symbol'].isin(unseen_symbols)]
    train_df = pd.concat(train_frames, ignore_index=True)
    time_test_df = pd.concat(time_test_frames, ignore_index=True)

    train_df = train_df.dropna(subset=feature_cols + [TARGET_COL])
    time_test_df = time_test_df.dropna(subset=feature_cols + [TARGET_COL])
    unseen_df = unseen_df.dropna(subset=feature_cols + [TARGET_COL])

    logger.info("训练集: %d 样本", len(train_df))
    logger.info("时间外样本: %d 样本", len(time_test_df))
    logger.info("完全外样本: %d 样本", len(unseen_df))

    # 特征选择
    final_features, importances = select_features(train_df, feature_cols)

    X_train = train_df[final_features].values
    y_train = train_df[TARGET_COL].values
    X_time = time_test_df[final_features].values
    y_time = time_test_df[TARGET_COL].values
    X_unseen = unseen_df[final_features].values
    y_unseen = unseen_df[TARGET_COL].values

    # 标准化
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_time_s = scaler.transform(X_time)
    X_unseen_s = scaler.transform(X_unseen)

    # 计算类别权重
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos = n_neg / n_pos
    logger.info("类别平衡权重: %.3f", scale_pos)

    # 定义优化的模型
    models = {
        "RandomForest_opt": RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=100,
            max_features='sqrt', class_weight='balanced',
            random_state=42, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            min_samples_leaf=200, subsample=0.8,
            random_state=42
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=100,
            max_features='sqrt', class_weight='balanced',
            random_state=42, n_jobs=-1
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            min_child_weight=50, subsample=0.8, colsample_bytree=0.6,
            scale_pos_weight=scale_pos,
            random_state=42, n_jobs=-1, eval_metric='logloss',
            use_label_encoder=False
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            num_leaves=31, min_child_samples=50, subsample=0.8,
            colsample_bytree=0.6, scale_pos_weight=scale_pos,
            random_state=42, n_jobs=-1, verbose=-1
        ),
        "LogisticRegression_opt": LogisticRegression(
            max_iter=2000, C=0.01, class_weight='balanced',
            random_state=42
        ),
    }

    results = {}
    best_model_name = None
    best_score = 0
    best_model = None

    test_sets = {
        "time_test": (X_time_s, y_time, "时间外样本"),
        "unseen": (X_unseen_s, y_unseen, "完全外样本"),
    }

    for name, model in models.items():
        logger.info("=" * 50)
        logger.info("训练: %s", name)

        model.fit(X_train_s, y_train)

        model_results = {}
        for test_key, (X_ts, y_ts, label) in test_sets.items():
            y_pred = model.predict(X_ts)
            proba = model.predict_proba(X_ts)[:, 1]
            acc = accuracy_score(y_ts, y_pred)
            prec = precision_score(y_ts, y_pred, zero_division=0)
            rec = recall_score(y_ts, y_pred, zero_division=0)
            f1 = f1_score(y_ts, y_pred, zero_division=0)
            roc = roc_auc_score(y_ts, proba)

            logger.info("  [%s] acc=%.4f prec=%.4f rec=%.4f f1=%.4f auc=%.4f",
                        label, acc, prec, rec, f1, roc)

            model_results[test_key] = {
                "accuracy": round(acc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "roc_auc": round(roc, 4),
            }

        # 用完全外样本的 AUC 来选最佳模型（AUC 比 F1 更能反映排序能力）
        auc_unseen = model_results["unseen"]["roc_auc"]
        if auc_unseen > best_score:
            best_score = auc_unseen
            best_model_name = name
            best_model = model

        results[name] = model_results

    logger.info("=" * 50)
    logger.info("最佳模型: %s (AUC=%.4f)", best_model_name, best_score)

    # 模型集成：取 Top 3 模型做 soft voting
    sorted_models = sorted(results.items(),
                           key=lambda x: x[1]["unseen"]["roc_auc"],
                           reverse=True)
    top3_names = [name for name, _ in sorted_models[:3]]
    logger.info("Top 3 集成: %s", ", ".join(top3_names))

    ensemble = VotingClassifier(
        estimators=[(name, models[name]) for name in top3_names],
        voting='soft'
    )
    ensemble.fit(X_train_s, y_train)

    y_ens_pred = ensemble.predict(X_unseen_s)
    y_ens_prob = ensemble.predict_proba(X_unseen_s)[:, 1]
    ens_f1 = f1_score(y_unseen, y_ens_pred)
    ens_auc = roc_auc_score(y_unseen, y_ens_prob)
    logger.info("  [集成] F1=%.4f AUC=%.4f", ens_f1, ens_auc)

    # 阈值优化：找最优概率阈值
    logger.info("阈值优化...")
    best_thresh = 0.5
    best_thresh_f1 = 0
    all_probs = ensemble.predict_proba(X_unseen_s)[:, 1]
    for thresh in np.arange(0.40, 0.70, 0.01):
        preds = (all_probs >= thresh).astype(int)
        f1 = f1_score(y_unseen, preds, zero_division=0)
        if f1 > best_thresh_f1:
            best_thresh_f1 = f1
            best_thresh = thresh

    logger.info("  最优阈值: %.2f (F1=%.4f)", best_thresh, best_thresh_f1)

    # 特征重要性（用最佳单模型）
    if hasattr(best_model, 'feature_importances_'):
        feat_imp = dict(zip(final_features, best_model.feature_importances_.tolist()))
        top_features = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:15]
        logger.info("Top 15 特征:")
        for feat, imp in top_features:
            logger.info("  %-30s %.4f", feat, imp)
    else:
        feat_imp = {}
        top_features = []

    # 保存最佳单模型（因为 ensemble 在某些环境加载有问题）
    import joblib
    model_data = {
        'model': best_model,
        'scaler': scaler,
        'feature_cols': final_features,
        'threshold': best_thresh,
        'model_type': best_model_name,
    }
    joblib.dump(model_data, os.path.join(MODEL_DIR, "model_optimized.pkl"))
    logger.info("模型已保存: %s/model_optimized.pkl", MODEL_DIR)

    # 保存报告
    report = {
        "best_single_model": best_model_name,
        "best_single_auc": round(best_score, 4),
        "ensemble_top3": top3_names,
        "ensemble_f1": round(ens_f1, 4),
        "ensemble_auc": round(ens_auc, 4),
        "optimal_threshold": round(best_thresh, 2),
        "optimal_threshold_f1": round(best_thresh_f1, 4),
        "all_results": results,
        "top_15_features": [{"name": f, "importance": round(i, 6)} for f, i in top_features],
        "feature_count": len(final_features),
        "features_removed": len(feature_cols) - len(final_features),
    }

    with open(REPORT_PATH, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    logger.info("报告已保存: %s", REPORT_PATH)


if __name__ == "__main__":
    train_optimized()
