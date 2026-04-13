---
name: gp-quant
description: A股量化交易框架 — 数据获取、技术指标、策略回测、机器学习、交易调度
---

## gp-quant 实战流程

### 1. 数据获取
```bash
python experiments/fetch_pool.py
```
- 从 `experiments/data/stock_pool.txt` 读取股票代码
- 使用 baostock 拉取前复权日线（OHLCV）
- 输出：`experiments/data/stock_pool_1y.csv`

### 2. 数据清洗
```bash
python experiments/clean_data.py
```
- 按 symbol 分组清洗：去空值/去重/OHLC 一致性检查/停牌剔除/异常值过滤
- 计算衍生列：涨跌幅、多周期收益率
- 输出：`experiments/data/cleaned/<symbol>.csv`（独立）+ `cleaned_pool.csv`（合并）

### 3. 技术指标
```bash
python experiments/calc_indicators.py
```
- 计算 24 个指标：MA/MACD/RSI/布林带/ATR/ADX/Stochastic/CCI/WR/OBV/MFI/成交量比率/波动率
- 输出：`experiments/data/indicators/<symbol>.csv` + `indicators_pool.csv`

### 4. 策略 + 回测
- 基于指标信号构建策略（MACD 金叉/RSI 超卖等）
- 回测引擎输出：胜率/夏普/最大回撤/盈亏比

### 5. 特征工程 + 模型训练
- 自动生成价格特征、滞后特征、技术指标特征
- scikit-learn 分类（涨跌方向）或回归（收益率）

### 6. 交易调度
- 风控配置 / 仓位 sizing / 模拟执行

## 项目结构
```
src/gp_quant/
├── data/           # 数据层 (fetcher/processor/storage)
├── strategy/       # 策略层 (base/indicators)
├── backtest/       # 回测引擎
├── ml/             # 机器学习 (features/trainer/model/predictor)
└── harness/        # 交易调度 (risk/sizing/execution)

experiments/        # 实战脚本
tests/              # 单元测试
```

详细流程见 [WORKFLOW.md](WORKFLOW.md)。
