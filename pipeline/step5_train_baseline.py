"""
Phase 5: 模型训练

训练 Random Forest 分类器，预测未来 5 日涨跌方向。
关键：按时间切分训练/测试集（避免未来信息泄漏）

输入：experiments/ml/features.csv
输出：models/model.pkl（模型）+ models/feature_columns.json（特征列）
     experiments/ml/train_report.json（训练报告）
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import logging
import json

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, roc_auc_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FEATURES_PATH = os.path.join(os.path.dirname(__file__), "ml", "features.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
TRAIN_REPORT_PATH = os.path.join(os.path.dirname(__file__), "ml", "train_report.json")

TARGET_COL = "target_direction_5d"


def time_based_split(df: pd.DataFrame, train_ratio: float = 0.8) -> tuple:
    """
    按时间切分（非随机！）
    前 80% 时间 = 训练集，后 20% = 测试集
    避免未来信息泄漏
    """
    df = df.sort_values('date').reset_index(drop=True)
    cutoff_idx = int(len(df) * train_ratio)
    return df.iloc[:cutoff_idx], df.iloc[cutoff_idx:]


def train_models():
    os.makedirs(MODEL_DIR, exist_ok=True)

    logger.info("读取特征数据: %s", FEATURES_PATH)
    df = pd.read_csv(FEATURES_PATH, dtype={'symbol': str})
    logger.info("原始数据: %d 行, %d 只股票", len(df), df['symbol'].nunique())

    # 定义特征列（排除非特征列）
    exclude_cols = {'symbol', 'date', 'close', 'target_5d', 'target_direction_5d',
                    'target_direction_5d', 'open', 'high', 'low'}
    feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['float64', 'int64', 'float32', 'int32']]

    logger.info("特征列: %d 个", len(feature_cols))

    # 目标变量
    df = df.dropna(subset=[TARGET_COL])
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    logger.info("正样本比例: %.2f%%", df[TARGET_COL].mean() * 100)

    # 股票级别切分：80% 股票 = seen，20% = 完全未见（OOS）
    symbols = sorted(df['symbol'].unique())
    np.random.seed(42)
    n_seen = int(len(symbols) * 0.8)
    seen_symbols = symbols[:n_seen]  # 前 73 只
    unseen_symbols = symbols[n_seen:]  # 后 18 只

    logger.info("Seen 股票: %d 只", len(seen_symbols))
    logger.info("Unseen 股票 (OOS): %d 只", len(unseen_symbols))

    # Seen 股票：按时间切分，前 80% 训练，后 20% 时间外样本
    train_frames = []
    time_test_frames = []
    for sym in seen_symbols:
        stock_df = df[df['symbol'] == sym].sort_values('date')
        cutoff = int(len(stock_df) * 0.8)
        train_frames.append(stock_df.iloc[:cutoff])
        time_test_frames.append(stock_df.iloc[cutoff:])

    # Unseen 股票：所有时间都是完全未见
    unseen_df = df[df['symbol'].isin(unseen_symbols)]

    train_df = pd.concat(train_frames, ignore_index=True)
    time_test_df = pd.concat(time_test_frames, ignore_index=True)

    # 丢弃有 NaN 的行
    train_df = train_df.dropna(subset=feature_cols + [TARGET_COL])
    time_test_df = time_test_df.dropna(subset=feature_cols + [TARGET_COL])
    unseen_df = unseen_df.dropna(subset=feature_cols + [TARGET_COL])

    X_train = train_df[feature_cols].values
    y_train = train_df[TARGET_COL].values
    X_time_test = time_test_df[feature_cols].values
    y_time_test = time_test_df[TARGET_COL].values
    X_unseen = unseen_df[feature_cols].values
    y_unseen = unseen_df[TARGET_COL].values

    logger.info("训练集: %d 样本", len(X_train))
    logger.info("时间外样本 (seen 股票后半段): %d 样本", len(X_time_test))
    logger.info("完全外样本 (未见股票): %d 样本", len(X_unseen))

    # 标准化（只在训练集上 fit）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_time_test_scaled = scaler.transform(X_time_test)
    X_unseen_scaled = scaler.transform(X_unseen)

    # 训练多个模型对比
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_leaf=50, random_state=42, n_jobs=-1),
    }

    results = {}
    best_model = None
    best_score = 0

    test_sets = {
        "time_test": (X_time_test_scaled, y_time_test, "时间外样本"),
        "unseen": (X_unseen_scaled, y_unseen, "完全外样本"),
    }

    for name, model in models.items():
        logger.info("=" * 40)
        logger.info("训练: %s", name)

        model.fit(X_train_scaled, y_train)

        model_results = {}
        for test_key, (X_ts, y_ts, label) in test_sets.items():
            y_pred = model.predict(X_ts)
            acc = accuracy_score(y_ts, y_pred)
            prec = precision_score(y_ts, y_pred, zero_division=0)
            rec = recall_score(y_ts, y_pred, zero_division=0)
            f1 = f1_score(y_ts, y_pred, zero_division=0)
            roc = roc_auc_score(y_ts, model.predict_proba(X_ts)[:, 1])

            logger.info("  [%s] acc=%.4f prec=%.4f rec=%.4f f1=%.4f auc=%.4f",
                        label, acc, prec, rec, f1, roc)

            model_results[test_key] = {
                "accuracy": round(acc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "roc_auc": round(roc, 4),
            }

        # 用完全外样本的 F1 来选最佳模型
        f1_unseen = model_results["unseen"]["f1"]
        if f1_unseen > best_score:
            best_score = f1_unseen
            best_model = (name, model)

        results[name] = model_results

    # 特征重要性（用最好的模型）
    name, model = best_model
    logger.info("=" * 40)
    logger.info("最佳模型: %s (F1=%.4f)", name, best_score)

    if hasattr(model, 'feature_importances_'):
        importances = dict(zip(feature_cols, model.feature_importances_.tolist()))
        top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:20]
        logger.info("Top 20 特征:")
        for feat, imp in top_features:
            logger.info("  %-30s %.4f", feat, imp)
    else:
        importances = {}
        top_features = []

    # 保存最佳模型
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_cols': feature_cols,
    }

    import joblib
    joblib.dump(model_data, os.path.join(MODEL_DIR, "model.pkl"))
    logger.info("模型已保存: %s/model.pkl", MODEL_DIR)

    # 保存报告
    report = {
        "best_model": name,
        "best_f1_unseen": round(best_score, 4),
        "all_results": results,
        "top_20_features": [{"name": f, "importance": round(i, 6)} for f, i in top_features],
        "train_samples": len(X_train),
        "time_test_samples": len(X_time_test),
        "unseen_samples": len(X_unseen),
        "positive_ratio_train": round(float(np.mean(y_train)), 4),
        "positive_ratio_time_test": round(float(np.mean(y_time_test)), 4),
        "positive_ratio_unseen": round(float(np.mean(y_unseen)), 4),
    }

    with open(TRAIN_REPORT_PATH, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    logger.info("报告已保存: %s", TRAIN_REPORT_PATH)


if __name__ == "__main__":
    train_models()
