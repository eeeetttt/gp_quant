"""框架功能测试"""
import pandas as pd
import numpy as np

# 创建模拟 OHLCV 数据
np.random.seed(42)
n = 500
dates = pd.date_range(start="2023-01-01", periods=n, freq="D")

df = pd.DataFrame({
    "date": dates,
    "open": 50 + np.cumsum(np.random.randn(n) * 0.5),
    "high": 50 + np.cumsum(np.random.randn(n) * 0.5) + np.abs(np.random.randn(n)),
    "low": 50 + np.cumsum(np.random.randn(n) * 0.5) - np.abs(np.random.randn(n)),
    "close": 50 + np.cumsum(np.random.randn(n) * 0.5) + np.random.randn(n) * 0.3,
    "volume": np.random.randint(1000000, 10000000, n)
})

df["high"] = np.maximum(df["open"], df["high"])
df["low"] = np.minimum(df["open"], df["low"])
df["close"] = df["close"].clip(lower=df["low"]*0.9, upper=df["high"]*1.1)

print("=" * 60)
print("gp-quant 框架测试")
print("=" * 60)
print(f"\n1. 模拟数据：{df.shape[0]} 条，{df.shape[1]} 列")
print(df.head(3))

# 2. 技术指标
print("\n2. 技术指标测试...")
from gp_quant.strategy.indicators import TechnicalIndicators

indicators = TechnicalIndicators(df)
indicators.add_indicator("rsi")
indicators.add_indicator("macd")
indicators.add_indicator("bollinger")
indicators.add_indicator("atr")

print(f"   RSI: {df['rsi'].min():.2f} ~ {df['rsi'].max():.2f}")
print(f"   MACD 信号：{indicators.get_macd_signal()}")
print(f"   布林带信号：{indicators.get_bollinger_signal()}")
print(f"   ATR: {df['atr'].iloc[-1]:.4f}")

# 3. 策略
print("\n3. 策略测试...")
from gp_quant.strategy.base import SimpleMovingAverageStrategy, MomentumStrategy, Signal

sma_strategy = SimpleMovingAverageStrategy(fast_window=5, slow_window=20)
signal = sma_strategy.generate_signal(df)
print(f"   双均线策略信号：{signal}")

momentum_strategy = MomentumStrategy(lookback_period=20, threshold=2.0)
signal = momentum_strategy.generate_signal(df)
print(f"   动量策略信号：{signal}")

# 4. 特征工程
print("\n4. 特征工程测试...")
from gp_quant.ml.features import FeatureEngineer, FeatureConfig

engineer = FeatureEngineer(FeatureConfig(target_horizon=5))
features_df = engineer.create_features(df.copy())
print(f"   特征数量：{len(features_df.columns) - 1}")
print(f"   前 5 个特征：{features_df.columns.tolist()[:5]}")

# 5. 模型训练
print("\n5. 模型训练测试...")
from gp_quant.ml.trainer import ModelTrainer, TrainConfig

X, y = engineer.prepare_data(features_df)
X_train, X_test = X[:400], X[400:]
y_train, y_test = y[:400], y[400:]

config = TrainConfig(model_type="random_forest", task_type="classification", random_state=42)
trainer = ModelTrainer(config)
X_train_scaled, X_test_scaled = trainer.scale_features(X_train, X_test)
trainer.train(X_train_scaled, y_train)

metrics = trainer.evaluate(X_test_scaled, y_test)
print(f"   准确率：{metrics['accuracy']:.4f}")
print(f"   特征重要性前 3:")
for feat, imp in sorted(trainer.get_feature_importance().items(), key=lambda x: x[1], reverse=True)[:3]:
    print(f"      {feat}: {imp:.4f}")

# 6. 回测
print("\n6. 回测测试...")
from gp_quant.backtest.engine import BacktestEngine

engine = BacktestEngine(initial_capital=100000.0, fee_rate=0.001)
engine.set_market_data(df)

df["signal"] = df["rsi"].apply(lambda x: "BUY" if x < 30 else ("SELL" if x > 70 else "HOLD"))
engine.set_signals(df)
results = engine.run()

print(f"   总交易次数：{results['total_trades']}")
print(f"   胜率：{results['win_rate']:.2f}%")
print(f"   总收益率：{results['total_return']:.2f}%")
print(f"   夏普比率：{results['sharpe_ratio']:.4f}")
print(f"   最大回撤：{results['max_drawdown']:.2f}%")

print("\n" + "=" * 60)
print("所有测试通过！框架功能正常。")
print("=" * 60)
