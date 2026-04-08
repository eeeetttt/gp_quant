"""测试框架功能"""
import sys
sys.path.insert(0, '/Users/et/program/gp-quant/src')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 创建模拟数据
dates = pd.date_range(start="2023-01-01", end="2024-12-31", freq="D")
np.random.seed(42)

n = len(dates)
df = pd.DataFrame({
    "date": dates,
    "open": 10 + np.random.randn(n).cumsum() + 50,
    "high": 10 + np.random.randn(n).cumsum() + 50 + np.abs(np.random.randn(n)),
    "low": 10 + np.random.randn(n).cumsum() + 50 - np.abs(np.random.randn(n)),
    "close": 10 + np.random.randn(n).cumsum() + 50 + np.random.randn(n) * 0.5,
    "volume": np.random.randint(1000000, 10000000, n)
})

df["high"] = np.maximum(df["open"], df["high"])
df["low"] = np.minimum(df["open"], df["low"])
df["close"] = df["close"].clip(lower=df["low"]*0.9, upper=df["high"]*1.1)

print(f"模拟数据形状：{df.shape}")
print(df.head())

# 1. 测试技术指标
print("\n1. 测试技术指标...")
from gp_quant.strategy.indicators import TechnicalIndicators

indicators = TechnicalIndicators(df)
indicators.add_indicator("rsi")
indicators.add_indicator("macd")
indicators.add_indicator("bollinger")

print(f"   RSI 范围：{df['rsi'].min():.2f} - {df['rsi'].max():.2f}")
print(f"   MACD 信号：{indicators.get_macd_signal()}")
print(f"   布林带信号：{indicators.get_bollinger_signal()}")

# 2. 测试策略
print("\n2. 测试策略...")
from gp_quant.strategy.base import SimpleMovingAverageStrategy, Signal

strategy = SimpleMovingAverageStrategy(fast_window=5, slow_window=20)
signal = strategy.generate_signal(df)
print(f"   策略信号：{signal}")

# 3. 测试特征工程
print("\n3. 测试特征工程...")
from gp_quant.ml.features import FeatureEngineer, FeatureConfig

engineer = FeatureEngineer(FeatureConfig(target_horizon=5))
features_df = engineer.create_features(df.copy())
print(f"   特征数量：{len(features_df.columns) - 1}")
print(f"   前 5 个特征：{features_df.columns.tolist()[:5]}")

# 4. 测试模型训练
print("\n4. 测试模型训练...")
from gp_quant.ml.trainer import ModelTrainer, TrainConfig

X, y = engineer.prepare_data(features_df)
X_train, X_test, y_train, y_test = X[:800], X[800:], y[:800], y[800:]

config = TrainConfig(model_type="random_forest", task_type="classification", random_state=42)
trainer = ModelTrainer(config)
X_train_scaled, X_test_scaled = trainer.scale_features(X_train, X_test)
trainer.train(X_train_scaled, y_train)

metrics = trainer.evaluate(X_test_scaled, y_test)
print(f"   准确率：{metrics['accuracy']:.4f}")
print(f"   特征重要性前 3:")
for feat, imp in sorted(trainer.get_feature_importance().items(), key=lambda x: x[1], reverse=True)[:3]:
    print(f"      {feat}: {imp:.4f}")

# 5. 测试回测
print("\n5. 测试回测...")
from gp_quant.backtest.engine import BacktestEngine

engine = BacktestEngine(initial_capital=100000.0, fee_rate=0.001)
engine.set_market_data(df)

# 生成简单信号
df["signal"] = df["rsi"].apply(lambda x: "BUY" if x < 30 else ("SELL" if x > 70 else "HOLD"))
engine.set_signals(df)
results = engine.run()

print(f"   总交易次数：{results['total_trades']}")
print(f"   胜率：{results['win_rate']:.2f}%")
print(f"   总收益率：{results['total_return']:.2f}%")
print(f"   夏普比率：{results['sharpe_ratio']:.4f}")
print(f"   最大回撤：{results['max_drawdown']:.2f}%")

print("\n" + "=" * 60)
print("所有测试通过！")
print("=" * 60)
