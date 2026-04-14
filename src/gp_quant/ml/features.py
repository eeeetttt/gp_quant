"""
特征工程模块
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA


@dataclass
class FeatureConfig:
    """特征配置"""
    target_column: str = "close"
    target_horizon: int = 5  # 预测未来 N 日收益率
    train_ratio: float = 0.8
    use_lag_features: bool = True
    use_technical_indicators: bool = True
    use_price_features: bool = True
    use_volume_features: bool = True
    scaling: str = "standard"  # "standard" or "minmax"
    drop_na: bool = True


class FeatureEngineer:
    """特征工程类"""

    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        初始化特征工程器

        Args:
            config: 特征配置
        """
        self.config = config or FeatureConfig()
        self.scaler: Optional[Any] = None
        self.pca: Optional[PCA] = None
        self.feature_columns: List[str] = []

    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        创建特征

        Args:
            data: 包含 OHLCV 的 DataFrame

        Returns:
            包含特征的 DataFrame
        """
        df = data.copy()

        # 技术指标特征 (must come first — price features reference bb_upper/bb_lower)
        if self.config.use_technical_indicators:
            from ..strategy.indicators import TechnicalIndicators
            indicators = TechnicalIndicators(df)
            df = indicators.get_all_indicators()

        # 价格特征
        if self.config.use_price_features:
            df = self._create_price_features(df)

        # 滞后特征
        if self.config.use_lag_features:
            df = self._create_lag_features(df)

        # 目标变量
        df = self._create_target(df)

        # 删除 NA
        if self.config.drop_na:
            df = df.dropna()

        # 提取特征列
        self.feature_columns = [col for col in df.columns if col not in [self.config.target_column, f"target_direction_{self.config.target_horizon}d"]]

        return df

    def _create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建价格相关特征"""
        df = df.copy()
        # 收益率
        df.loc[:, "return_1d"] = df["close"].pct_change(1)
        df.loc[:, "return_5d"] = df["close"].pct_change(5)
        df.loc[:, "return_10d"] = df["close"].pct_change(10)

        # 价格位置
        df.loc[:, "price_ma_ratio"] = df["close"] / df["close"].rolling(20).mean()

        # 布林带位置 (only if bollinger bands already calculated)
        if "bb_upper" in df.columns and "bb_lower" in df.columns:
            df.loc[:, "price_bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

        # 波动率
        df.loc[:, "volatility_5d"] = df["return_1d"].rolling(5).std()
        df.loc[:, "volatility_10d"] = df["return_1d"].rolling(10).std()

        # 价格区间
        df.loc[:, "range_pct"] = (df["high"] - df["low"]) / df["close"]

        # 跳空
        df.loc[:, "gap"] = df["open"] - df["close"].shift(1)
        df.loc[:, "gap_pct"] = df["gap"] / df["close"].shift(1)

        return df

    def _create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建滞后特征"""
        df = df.copy()
        # 价格滞后
        for lag in [1, 2, 3, 5]:
            df.loc[:, f"close_lag_{lag}"] = df["close"].shift(lag)
            df.loc[:, f"return_lag_{lag}"] = df["return_1d"].shift(lag)
            df.loc[:, f"volume_lag_{lag}"] = df["volume"].shift(lag)

        # 指标滞后
        if "rsi" in df.columns:
            df.loc[:, "rsi_lag_1"] = df["rsi"].shift(1)
            df.loc[:, "rsi_lag_5"] = df["rsi"].shift(5)

        if "macd" in df.columns:
            df.loc[:, "macd_lag_1"] = df["macd"].shift(1)
            df.loc[:, "macd_signal_lag_1"] = df["signal"].shift(1)

        if "volatility" in df.columns:
            df.loc[:, "volatility_lag_1"] = df["volatility"].shift(1)

        return df

    def _create_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建目标变量（未来收益率）"""
        df = df.copy()
        horizon = self.config.target_horizon

        # 未来 N 日收益率（向前看，不是向后看！）
        df.loc[:, f"target_{horizon}d"] = (df["close"].shift(-horizon) - df["close"]) / df["close"] * 100

        # 方向标签 (0: 下跌，1: 上涨)
        df.loc[:, f"target_direction_{horizon}d"] = (df[f"target_{horizon}d"] > 0).astype(int)

        return df

    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        准备训练数据

        Args:
            df: 包含特征的 DataFrame

        Returns:
            (X, y) 特征和目标数组
        """
        # Only use numeric columns, exclude target and non-numeric
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude = {self.config.target_column, f"target_direction_{self.config.target_horizon}d"}
        feature_cols = [c for c in numeric_cols if c not in exclude]
        self.feature_columns = feature_cols

        X = df[feature_cols].values
        y = df[self.config.target_column].values

        return X, y

    def scale_features(self, X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        特征缩放

        Args:
            X_train: 训练集特征
            X_test: 测试集特征

        Returns:
            (scaled_X_train, scaled_X_test)
        """
        if self.config.scaling == "minmax":
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler()

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled

    def apply_pca(self, X_train: np.ndarray, X_test: np.ndarray, n_components: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        PCA 降维

        Args:
            X_train: 训练集特征
            X_test: 测试集特征
            n_components: 降维后维度

        Returns:
            (pca_X_train, pca_X_test)
        """
        self.pca = PCA(n_components=min(n_components, X_train.shape[1]))
        X_train_pca = self.pca.fit_transform(X_train)
        X_test_pca = self.pca.transform(X_test)

        return X_train_pca, X_test_pca

    def get_feature_importance(self, model: Any) -> Dict[str, float]:
        """
        获取特征重要性

        Args:
            model: 训练好的模型 (需有 feature_importances_ 或 coef_ 属性)

        Returns:
            特征重要性字典
        """
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_)
        else:
            raise ValueError("Model does not have feature_importances_ or coef_")

        return dict(zip(self.feature_columns, importances))

    def select_top_features(self, importance_dict: Dict[str, float], top_n: int = 30) -> List[str]:
        """
        选择最重要的特征

        Args:
            importance_dict: 特征重要性字典
            top_n: 选择前 N 个特征

        Returns:
            前 N 个特征列表
        """
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        return [feat for feat, _ in sorted_features[:top_n]]

    def create_feature_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        创建特征报告

        Args:
            df: 包含特征的 DataFrame

        Returns:
            特征报告字典
        """
        return {
            "total_features": len(df.columns) - 1,
            "price_features": sum(1 for col in df.columns if "price" in col or "return" in col),
            "technical_features": sum(1 for col in df.columns if any(kw in col for kw in ["rsi", "macd", "bollinger", "atr"])),
            "lag_features": sum(1 for col in df.columns if "lag" in col),
            "target_columns": [col for col in df.columns if col.startswith("target")],
            "missing_values": df.isnull().sum().to_dict(),
            "feature_statistics": df.describe().to_dict(),
        }

    def validate_features(self, df: pd.DataFrame) -> List[str]:
        """
        验证特征质量

        Args:
            df: 包含特征的 DataFrame

        Returns:
            警告信息列表
        """
        warnings = []

        # 检查常数特征（只检查数值列）
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].std() == 0:
                warnings.append(f"Constant feature: {col}")

        # 检查高缺失值
        missing_pct = df.isnull().sum() / len(df)
        high_missing = missing_pct[missing_pct > 0.5]
        if len(high_missing) > 0:
            warnings.append(f"High missing ratio features: {high_missing.to_dict()}")

        # 检查特征相关性
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()
        high_corr = []

        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > 0.95:
                    high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j]))

        if high_corr:
            warnings.append(f"Highly correlated features: {high_corr[:10]}")

        return warnings
