# gp-quant 股票量化交易框架

A 股量化交易框架，支持技术指标分析、策略回测、机器学习预测。

## 功能特性

### 数据层
- **数据获取**: AkShare A 股数据
- **数据处理**: 数据清洗、标准化、特征工程
- **数据存储**: CSV 文件存储

### 策略层
- **技术指标**: RSI、MACD、布林带、ATR、ADX、KDJ 等 15+ 指标
- **策略模板**: 双均线策略、动量策略

### 回测引擎
- **交易记录**: 完整的订单/平仓记录
- **性能指标**: 夏普比率、最大回撤、胜率、盈亏比

### 机器学习
- **特征工程**: 自动生成技术指标 + 滞后特征
- **模型**: 随机森林、梯度提升、神经网络
- **训练/预测**: 完整的训练流程和预测接口

## 安装

```bash
# 创建 conda 环境
conda create -n gp-quant python=3.12 -y
conda activate gp-quant

# 安装框架
pip install -e .

# 或使用 requirements.txt
pip install -r requirements.txt
```

## 快速开始

### 获取数据

```bash
# 获取平安银行数据
gp-quant fetch 000001.SZ --start-date 2023-01-01 --end-date 2024-12-31

# 保存到文件
gp-quant fetch 000001.SZ --output data/000001.csv
```

### 技术指标分析

```bash
# 计算 RSI 和 MACD
gp-quant analyze 000001.SZ --indicator rsi --indicator macd

# 输出到文件
gp-quant analyze 000001.SZ --output analysis.csv
```

### 回测

```bash
# 运行回测
gp-quant backtest 000001.SZ --initial-capital 100000

# 保存结果
gp-quant backtest 000001.SZ --output results.json
```

### 机器学习

```bash
# 训练模型
gp-quant train --model-type random_forest

# 预测
gp-quant predict --model-path ./models/best_model.pkl
```

## Python 使用示例

```python
from gp_quant import create_fetcher, TechnicalIndicators, BacktestEngine

# 1. 获取数据
fetcher = create_fetcher(source="akshare")
df = fetcher.fetch("000001.SZ", "2023-01-01", "2024-12-31")

# 2. 计算技术指标
indicators = TechnicalIndicators(df)
indicators.add_indicator("rsi")
indicators.add_indicator("macd")
indicators.add_indicator("bollinger")

# 3. 回测
engine = BacktestEngine(initial_capital=100000.0)
engine.set_market_data(df)
df["signal"] = df["rsi"].apply(lambda x: "BUY" if x < 30 else "HOLD")
engine.set_signals(df)
results = engine.run()

print(f"总收益率：{results['total_return']:.2f}%")
print(f"夏普比率：{results['sharpe_ratio']:.4f}")
```

## 项目结构

```
gp-quant/
├── src/gp_quant/
│   ├── data/           # 数据层
│   │   ├── fetcher.py  # 数据获取 (AkShare)
│   │   ├── processor.py # 数据处理
│   │   └── storage.py  # 数据存储
│   ├── strategy/       # 策略层
│   │   ├── base.py     # 策略基类
│   │   └── indicators.py # 技术指标
│   ├── backtest/       # 回测层
│   │   └── engine.py   # 回测引擎
│   └── ml/            # 机器学习层
│       ├── features.py # 特征工程
│       ├── model.py    # 模型类
│       ├── trainer.py  # 训练器
│       └── predictor.py # 预测器
├── temp/              # 测试脚本
└── requirements.txt   # 依赖列表
```

## 依赖

- Python >= 3.12
- pandas >= 2.0.3
- numpy >= 1.24.3
- akshare >= 1.14.0 (A 股数据)
- scikit-learn >= 1.3.0 (机器学习)
- torch (可选，神经网络)

## 许可证

MIT License
