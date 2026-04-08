"""
模型预测模块
"""
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
import pandas as pd
import os

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


@dataclass
class PredictionResult:
    """预测结果"""
    symbol: str
    prediction: Any
    confidence: float
    features_used: Dict[str, float]
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            from datetime import datetime
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "prediction": self.prediction,
            "confidence": self.confidence,
            "features_used": self.features_used,
            "timestamp": self.timestamp
        }


class ModelPredictor:
    """模型预测器"""

    def __init__(self, model_path: Optional[str] = None):
        """
        初始化预测器

        Args:
            model_path: 模型文件路径
        """
        self.model = None
        self.scaler = None
        self.feature_names: List[str] = []
        self.model_type: str = ""
        self.task_type: str = ""

        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str):
        """加载模型"""
        import joblib

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        data = joblib.load(model_path)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.feature_names = data["feature_names"]
        self.model_type = data["config"]["model_type"]
        self.task_type = data["config"]["task_type"]

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        预测

        Args:
            features: 特征数组 (n_samples, n_features)

        Returns:
            预测结果
        """
        if self.model is None:
            raise ValueError("Model not loaded")

        # 特征缩放
        X_scaled = self.scaler.transform(features)

        if self.task_type == "classification":
            predictions = self.model.predict(X_scaled)
            probabilities = self.model.predict_proba(X_scaled) if hasattr(self.model, "predict_proba") else None

            if probabilities is not None:
                # 返回预测类别和置信度
                result = []
                for i, pred in enumerate(predictions):
                    confidence = probabilities[i][pred]
                    result.append({
                        "class": int(pred),
                        "confidence": float(confidence)
                    })
                return np.array(result)
            else:
                return predictions
        else:
            return self.model.predict(X_scaled)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """预测概率"""
        if self.model is None:
            raise ValueError("Model not loaded")

        X_scaled = self.scaler.transform(features)

        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_scaled)
        else:
            raise ValueError("Model does not support predict_proba")

    def predict_single(self, feature_dict: Dict[str, float]) -> PredictionResult:
        """
        单个样本预测

        Args:
            feature_dict: 特征字典

        Returns:
            PredictionResult 对象
        """
        # 转换为数组
        feature_array = np.array([[feature_dict.get(name, 0) for name in self.feature_names]])

        # 预测
        prediction = self.predict(feature_array)

        # 提取结果
        if isinstance(prediction[0], dict):
            pred_class = prediction[0]["class"]
            confidence = prediction[0]["confidence"]
        else:
            pred_class = prediction[0]
            confidence = 1.0

        return PredictionResult(
            symbol="default",
            prediction=pred_class,
            confidence=confidence,
            features_used=feature_dict
        )

    def get_feature_importance(self) -> Dict[str, float]:
        """获取特征重要性"""
        if self.model is None:
            return {}

        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            importances = np.abs(self.model.coef_[0])
        else:
            return {}

        return dict(zip(self.feature_names, importances))

    def explain_prediction(self, feature_dict: Dict[str, float]) -> Dict[str, Any]:
        """
        解释预测结果

        Args:
            feature_dict: 特征字典

        Returns:
            解释字典
        """
        # 获取特征重要性
        importance = self.get_feature_importance()

        # 计算每个特征的贡献
        feature_array = np.array([[feature_dict.get(name, 0) for name in self.feature_names]])
        X_scaled = self.scaler.transform(feature_array)

        if self.task_type == "classification":
            probabilities = self.predict_proba(X_scaled)
            predicted_class = self.predict(feature_array)[0]

            if isinstance(predicted_class, dict):
                predicted_class = predicted_class["class"]

            return {
                "predicted_class": int(predicted_class),
                "probabilities": probabilities[0].tolist(),
                "feature_importance": importance,
                "top_features": sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
            }
        else:
            prediction = self.predict(feature_array)[0]
            return {
                "prediction": prediction,
                "feature_importance": importance,
                "top_features": sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
            }

    def batch_predict(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        批量预测

        Args:
            features_df: 包含特征的 DataFrame

        Returns:
            预测结果数组
        """
        # 确保特征顺序正确
        X = features_df[self.feature_names].values
        return self.predict(X)

    def save_model(self, model_path: str):
        """保存模型"""
        import joblib

        os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else ".", exist_ok=True)

        data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "config": {
                "model_type": self.model_type,
                "task_type": self.task_type
            }
        }

        joblib.dump(data, model_path)
        print(f"Model saved to {model_path}")

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "model_type": self.model_type,
            "task_type": self.task_type,
            "feature_names": self.feature_names,
            "n_features": len(self.feature_names)
        }
