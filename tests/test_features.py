"""Tests for ML features"""
import pytest
import pandas as pd
import numpy as np
from gp_quant.ml.features import FeatureEngineer, FeatureConfig


class TestFeatureEngineer:
    def test_create_features(self, sample_ohlcv):
        config = FeatureConfig(
            use_technical_indicators=True,
            use_lag_features=True,
            use_price_features=True,
        )
        engineer = FeatureEngineer(config)
        df = engineer.create_features(sample_ohlcv)
        assert len(df.columns) > 10
        assert len(df) > 0

    def test_target_creation(self, sample_ohlcv):
        config = FeatureConfig(target_horizon=5)
        engineer = FeatureEngineer(config)
        df = engineer.create_features(sample_ohlcv)
        assert f"target_5d" in df.columns
        assert f"target_direction_5d" in df.columns

    def test_prepare_data(self, sample_ohlcv):
        config = FeatureConfig()
        engineer = FeatureEngineer(config)
        df = engineer.create_features(sample_ohlcv)
        X, y = engineer.prepare_data(df)
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == len(engineer.feature_columns)

    def test_scale_features(self, sample_ohlcv):
        config = FeatureConfig(scaling="standard")
        engineer = FeatureEngineer(config)
        df = engineer.create_features(sample_ohlcv)
        X, y = engineer.prepare_data(df)
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        X_train_s, X_test_s = engineer.scale_features(X_train, X_test)
        assert X_train_s.shape == X_train.shape
        assert abs(X_train_s.mean()) < 1e-10

    def test_validate_features(self, sample_ohlcv):
        config = FeatureConfig()
        engineer = FeatureEngineer(config)
        df = engineer.create_features(sample_ohlcv)
        warnings = engineer.validate_features(df)
        assert isinstance(warnings, list)

    def test_feature_report(self, sample_ohlcv):
        config = FeatureConfig()
        engineer = FeatureEngineer(config)
        df = engineer.create_features(sample_ohlcv)
        report = engineer.create_feature_report(df)
        assert "total_features" in report
        assert report["total_features"] > 0
