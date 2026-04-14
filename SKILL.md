---
name: gp-quant
description: A股量化交易框架 — 获取最新行情，计算技术指标，用训练好的模型预测涨跌方向，输出买卖建议
---

## 调用方式（推理模式）

给定股票代码，获取最新数据，用训练好的模型预测涨跌方向。

```bash
# 单只股票预测
python experiments/predict_single.py 600519

# 批量预测（从股票池）
python experiments/predict_batch.py
```

输出：JSON，包含预测方向、置信度、技术指标信号、买卖建议。

## 训练流程（开发者用）

```bash
# 0. 选股 → experiments/data/stock_pool.txt
python pipeline/step0_screen.py

# 1. 数据获取 — baostock 拉取前复权日线
python pipeline/step1_fetch.py

# 2. 数据清洗 — 去空值/停牌剔除/OHLC 一致性/多周期收益率
python pipeline/step2_clean.py

# 3. 技术指标 — MA/MACD/RSI/布林带/ATR/ADX 等 24 个指标
python pipeline/step3_indicators.py

# 4. 特征工程 — 价格特征/滞后特征/大盘基准/相对强弱/资金流/时间/动量
python pipeline/step4_features.py

# 5. 模型训练 — scikit-learn 分类（涨跌方向，股票级别 OOS 切分）
python pipeline/step5_train_baseline.py

# 6. 回测验证 — 用历史数据验证模型预测准确率、夏普、回撤
python pipeline/step6_backtest.py
```

训练产出：`models/model.pkl`（供推理使用）+ 回测报告

## 项目结构
```
src/gp_quant/          # 框架核心代码
  ├── data/            # 数据层 (获取/处理/存储)
  ├── strategy/        # 技术指标
  ├── backtest/        # 回测引擎
  ├── ml/              # 特征工程/训练/预测
  └── harness/         # 交易调度 (风控/仓位/执行)

pipeline/              # 训练流程脚本（step0 → step6）
scripts/               # 调试/检查脚本
models/                # 训练好的模型文件
tests/                 # 单元测试（conftest + test_*.py）
experiments/           # 数据/中间结果（.gitignore 忽略）
```

详细流程见 [WORKFLOW.md](WORKFLOW.md)。
