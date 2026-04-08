"""
机器学习模型模块
"""
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import warnings

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


@dataclass
class ModelConfig:
    """模型配置"""
    model_type: str = "random_forest"  # "random_forest", "gradient_boosting", "logistic_regression", "neural_network"
    task_type: str = "classification"  # "classification", "regression"
    n_estimators: int = 100
    max_depth: int = 10
    learning_rate: float = 0.1
    hidden_layers: List[int] = None
    dropout_rate: float = 0.3
    epochs: int = 100
    batch_size: int = 32
    device: str = "cpu"

    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [128, 64, 32]


class QuantClassifier(BaseEstimator, ClassifierMixin):
    """量化交易分类器"""

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.model: Optional[Any] = None
        self.scaler = StandardScaler()
        self._initialize_model()

    def _initialize_model(self):
        """初始化模型"""
        if self.config.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced"
            )
        elif self.config.model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                random_state=42
            )
        elif self.config.model_type == "logistic_regression":
            self.model = LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            )
        elif self.config.model_type == "neural_network" and TORCH_AVAILABLE:
            self.model = QuantNeuralNetwork(
                input_dim=0,  # Will be set during fit
                hidden_layers=self.config.hidden_layers,
                dropout_rate=self.config.dropout_rate
            )
        else:
            self.model = RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                random_state=42
            )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QuantClassifier":
        """
        训练模型

        Args:
            X: 特征数组 (n_samples, n_features)
            y: 标签数组 (n_samples,)

        Returns:
            训练好的模型
        """
        # 特征缩放
        X_scaled = self.scaler.fit_transform(X)

        if self.config.model_type == "neural_network" and TORCH_AVAILABLE:
            self.model.input_dim = X.shape[1]
            self._train_neural_network(X_scaled, y)
        else:
            self.model.fit(X_scaled, y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测

        Args:
            X: 特征数组

        Returns:
            预测标签
        """
        X_scaled = self.scaler.transform(X)

        if self.config.model_type == "neural_network" and TORCH_AVAILABLE:
            return self._predict_neural_network(X_scaled)
        else:
            return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测概率

        Args:
            X: 特征数组

        Returns:
            预测概率
        """
        X_scaled = self.scaler.transform(X)

        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_scaled)
        else:
            # Fallback: use decision function
            return self.model.decision_function(X_scaled)

    def get_feature_importance(self) -> Dict[str, float]:
        """获取特征重要性"""
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            return dict(zip(self.feature_names, importances))
        elif hasattr(self.model, "coef_"):
            importances = np.abs(self.model.coef_[0])
            return dict(zip(self.feature_names, importances))
        return {}

    def set_feature_names(self, names: List[str]):
        """设置特征名称"""
        self.feature_names = names


class QuantRegressor(BaseEstimator, RegressorMixin):
    """量化交易回归器"""

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig(task_type="regression")
        self.model: Optional[Any] = None
        self.scaler = StandardScaler()
        self._initialize_model()

    def _initialize_model(self):
        """初始化模型"""
        if self.config.model_type == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                random_state=42,
                n_jobs=-1
            )
        elif self.config.model_type == "gradient_boosting":
            self.model = GradientBoostingRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                random_state=42
            )
        elif self.config.model_type == "neural_network" and TORCH_AVAILABLE:
            self.model = QuantNeuralNetwork(
                input_dim=0,
                hidden_layers=self.config.hidden_layers,
                dropout_rate=self.config.dropout_rate,
                is_regression=True
            )
        else:
            self.model = Ridge(alpha=1.0)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QuantRegressor":
        """训练模型"""
        X_scaled = self.scaler.fit_transform(X)

        if self.config.model_type == "neural_network" and TORCH_AVAILABLE:
            self.model.input_dim = X.shape[1]
            self._train_neural_network(X_scaled, y)
        else:
            self.model.fit(X_scaled, y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        X_scaled = self.scaler.transform(X)

        if self.config.model_type == "neural_network" and TORCH_AVAILABLE:
            return self._predict_neural_network(X_scaled)
        else:
            return self.model.predict(X_scaled)

    def get_feature_importance(self) -> Dict[str, float]:
        """获取特征重要性"""
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            return dict(zip(self.feature_names, importances))
        return {}

    def set_feature_names(self, names: List[str]):
        """设置特征名称"""
        self.feature_names = names

    def _train_neural_network(self, X: np.ndarray, y: np.ndarray):
        """训练神经网络"""
        self.model.input_dim = X.shape[1]
        self.model.fit(X, y, epochs=self.config.epochs, batch_size=self.config.batch_size)

    def _predict_neural_network(self, X: np.ndarray) -> np.ndarray:
        """神经网络预测"""
        return self.model.predict(X)


class QuantNeuralNetwork(nn.Module):
    """神经网络模型"""

    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int] = None,
        dropout_rate: float = 0.3,
        is_regression: bool = False
    ):
        super().__init__()
        self.is_regression = is_regression
        self.hidden_layers = hidden_layers or [128, 64, 32]

        layers = []
        prev_dim = input_dim

        for hidden_dim in self.hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        if is_regression:
            layers.append(nn.Linear(prev_dim, 1))
        else:
            layers.append(nn.Linear(prev_dim, 2))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001
    ):
        """训练模型"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not installed")

        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y) if self.is_regression else torch.LongTensor(y)

        criterion = nn.CrossEntropyLoss() if not self.is_regression else nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self(batch_X)

                if not self.is_regression:
                    loss = criterion(outputs, batch_y)
                else:
                    loss = criterion(outputs.squeeze(), batch_y)

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not installed")

        self.eval()
        X_tensor = torch.FloatTensor(X)

        with torch.no_grad():
            outputs = self(X_tensor)

            if not self.is_regression:
                _, predicted = torch.max(outputs, 1)
                return predicted.numpy()
            else:
                return outputs.numpy().squeeze()
