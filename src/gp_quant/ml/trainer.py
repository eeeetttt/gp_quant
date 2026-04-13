"""
模型训练模块
"""
from __future__ import annotations

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
import json
import os

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


@dataclass
class TrainConfig:
    """训练配置"""
    model_type: str = "random_forest"  # "random_forest", "gradient_boosting", "logistic_regression", "neural_network"
    task_type: str = "classification"  # "classification", "regression"
    train_ratio: float = 0.8
    test_ratio: float = 0.2
    val_ratio: float = 0.0
    random_state: int = 42
    n_estimators: int = 100
    max_depth: int = 10
    learning_rate: float = 0.1
    epochs: int = 100
    batch_size: int = 32
    device: str = "cpu"
    save_path: str = "./models"
    use_cross_validation: bool = True
    cv_folds: int = 5


class ModelTrainer:
    """模型训练器"""

    def __init__(self, config: Optional[TrainConfig] = None):
        """初始化训练器"""
        self.config = config or TrainConfig()
        self.model = None
        self.scaler = None
        self.feature_names: List[str] = []
        self.history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
        self._initialize_model()

    def _initialize_model(self):
        """初始化模型"""
        if self.config.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                random_state=self.config.random_state,
                n_jobs=-1
            )
        elif self.config.model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                random_state=self.config.random_state
            )
        elif self.config.model_type == "logistic_regression":
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=self.config.random_state
            )
        elif self.config.model_type == "neural_network" and TORCH_AVAILABLE:
            self.model = self._create_neural_network()
        else:
            self.model = RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                random_state=self.config.random_state
            )

    def _create_neural_network(self) -> nn.Module:
        """创建神经网络"""
        class QuantNeuralNetwork(nn.Module):
            def __init__(self, input_dim: int, hidden_layers: List[int] = None, dropout_rate: float = 0.3):
                super().__init__()
                self.hidden_layers = hidden_layers or [128, 64, 32]
                layers = []
                prev_dim = input_dim

                for hidden_dim in self.hidden_layers:
                    layers.append(nn.Linear(prev_dim, hidden_dim))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(dropout_rate))
                    prev_dim = hidden_dim

                if self.config.task_type == "regression":
                    layers.append(nn.Linear(prev_dim, 1))
                else:
                    layers.append(nn.Linear(prev_dim, 2))

                self.network = nn.Sequential(*layers)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.network(x)

        return QuantNeuralNetwork(0)

    def prepare_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        准备数据

        Args:
            X: 特征数组
            y: 标签数组
            feature_names: 特征名称列表

        Returns:
            (X_train, X_test, y_train, y_test)
        """
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]

        # 数据分割
        if self.config.val_ratio > 0:
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=self.config.test_ratio + self.config.val_ratio,
                random_state=self.config.random_state, stratify=y if self.config.task_type == "classification" else None
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=self.config.val_ratio / (self.config.test_ratio + self.config.val_ratio),
                random_state=self.config.random_state,
                stratify=y_temp if self.config.task_type == "classification" else None
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.config.test_ratio,
                random_state=self.config.random_state,
                stratify=y if self.config.task_type == "classification" else None
            )
            X_val, y_val = X_test, y_test

        return X_train, X_test, y_train, y_test

    def scale_features(self, X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """特征缩放"""
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> "ModelTrainer":
        """
        训练模型

        Args:
            X_train: 训练集特征
            y_train: 训练集标签
            X_val: 验证集特征 (可选)
            y_val: 验证集标签 (可选)

        Returns:
            训练好的模型
        """
        if self.config.model_type == "neural_network" and TORCH_AVAILABLE:
            self._train_neural_network(X_train, y_train, X_val, y_val)
        else:
            self.model.fit(X_train, y_train)

        return self

    def _train_neural_network(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ):
        """训练神经网络"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not installed")

        # 设置输入维度
        if hasattr(self.model, "input_dim"):
            self.model.input_dim = X_train.shape[1]

        # 准备数据
        X_tensor = torch.FloatTensor(X_train)
        y_tensor = torch.FloatTensor(y_train) if self.config.task_type == "regression" else torch.LongTensor(y_train)

        dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        # 损失函数和优化器
        criterion = nn.MSELoss() if self.config.task_type == "regression" else nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

        # 训练循环
        best_val_loss = float("inf")
        patience = 20
        patience_counter = 0

        for epoch in range(self.config.epochs):
            self.model.train()
            total_loss = 0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)

                if self.config.task_type == "regression":
                    loss = criterion(outputs.squeeze(), batch_y)
                else:
                    loss = criterion(outputs, batch_y)

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            self.history["train_loss"].append(avg_loss)

            # 验证
            if X_val is not None:
                val_loss = self._evaluate(X_val, y_val)
                self.history["val_loss"].append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # 保存最佳模型
                    torch.save(self.model.state_dict(), os.path.join(self.config.save_path, "best_model.pt"))
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch + 1}")
                        break
            else:
                self.history["val_loss"].append(avg_loss)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{self.config.epochs}], Loss: {avg_loss:.4f}")

        # 加载最佳模型
        save_path = os.path.join(self.config.save_path, "best_model.pt")
        if os.path.exists(save_path):
            self.model.load_state_dict(torch.load(save_path))

    def _evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """评估模型"""
        if TORCH_AVAILABLE:
            X_tensor = torch.FloatTensor(X)
            outputs = self.model(X_tensor)
            if self.config.task_type == "regression":
                return mean_squared_error(y, outputs.numpy().squeeze())
            else:
                return nn.CrossEntropyLoss()(outputs, torch.LongTensor(y)).item()
        else:
            return self.model.score(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        X_scaled = self.scaler.transform(X) if self.scaler else X

        if self.config.model_type == "neural_network" and TORCH_AVAILABLE:
            self.model.eval()
            X_tensor = torch.FloatTensor(X_scaled)
            outputs = self.model(X_tensor)
            if self.config.task_type == "regression":
                return outputs.numpy().squeeze()
            else:
                _, predicted = torch.max(outputs, 1)
                return predicted.numpy()
        else:
            return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        X_scaled = self.scaler.transform(X) if self.scaler else X

        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_scaled)
        else:
            raise ValueError("Model does not support predict_proba")

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        评估模型

        Args:
            X_test: 测试集特征
            y_test: 测试集标签

        Returns:
            评估指标字典
        """
        y_pred = self.predict(X_test)

        if self.config.task_type == "classification":
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
                "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            }
        else:
            metrics = {
                "mse": mean_squared_error(y_test, y_pred),
                "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                "mae": mean_absolute_error(y_test, y_pred),
                "r2": r2_score(y_test, y_pred),
            }

        return metrics

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv_folds: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """交叉验证"""
        folds = cv_folds or self.config.cv_folds

        if self.config.task_type == "classification":
            scores = cross_val_score(self.model, X, y, cv=folds, scoring="accuracy")
        else:
            scores = cross_val_score(self.model, X, y, cv=folds, scoring="r2")

        return {
            "folds": folds,
            "scores": scores,
            "mean": scores.mean(),
            "std": scores.std(),
            "cv_results": [f"Fold {i+1}: {score:.4f}" for i, score in enumerate(scores)]
        }

    def grid_search(
        self,
        X: np.ndarray,
        y: np.ndarray,
        param_grid: Optional[Dict[str, List]] = None
    ) -> Dict[str, Any]:
        """网格搜索调参"""
        if param_grid is None:
            param_grid = {
                "n_estimators": [50, 100, 200],
                "max_depth": [5, 10, None],
            }

        grid_search = GridSearchCV(
            self.model,
            param_grid,
            cv=5,
            scoring="accuracy" if self.config.task_type == "classification" else "r2",
            n_jobs=-1
        )

        grid_search.fit(X, y)

        return {
            "best_params": grid_search.best_params_,
            "best_score": grid_search.best_score_,
            "cv_results": grid_search.cv_results_
        }

    def get_feature_importance(self) -> Dict[str, float]:
        """获取特征重要性"""
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            importances = np.abs(self.model.coef_[0])
        else:
            return {}

        return dict(zip(self.feature_names, importances))

    def save_model(self, filepath: Optional[str] = None):
        """保存模型"""
        if filepath is None:
            os.makedirs(self.config.save_path, exist_ok=True)
            filepath = os.path.join(self.config.save_path, f"{self.config.model_type}_model.pkl")

        import joblib
        joblib.dump({
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "config": {
                "model_type": self.config.model_type,
                "task_type": self.config.task_type
            }
        }, filepath)

        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """加载模型"""
        import joblib
        data = joblib.load(filepath)

        self.model = data["model"]
        self.scaler = data["scaler"]
        self.feature_names = data["feature_names"]

        print(f"Model loaded from {filepath}")

    def get_history(self) -> Dict[str, List[float]]:
        """获取训练历史"""
        return self.history

    def get_model_summary(self) -> str:
        """获取模型摘要"""
        summary = [
            f"模型类型：{self.config.model_type}",
            f"任务类型：{self.config.task_type}",
            f"训练集大小：{self.config.train_ratio}",
            f"测试集大小：{self.config.test_ratio}",
            f"随机种子：{self.config.random_state}",
        ]

        if hasattr(self.model, "n_estimators"):
            summary.append(f"估计器数量：{self.config.n_estimators}")
        if hasattr(self.model, "max_depth"):
            summary.append(f"最大深度：{self.config.max_depth}")

        return "\n".join(summary)
