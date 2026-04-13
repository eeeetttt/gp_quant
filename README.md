# gp-quant A 股量化交易框架

A 股量化交易框架，支持技术指标分析、策略回测、机器学习预测和交易调度。

## 快速开始

### 安装

```bash
conda create -n gp-quant python=3.12 -y
conda activate gp-quant
pip install -e .
```

### 实战流程

```bash
# 1. 获取数据（baostock，免费无需注册）
python experiments/fetch_pool.py

# 2. 数据清洗
python experiments/clean_data.py

# 3. 计算技术指标
python experiments/calc_indicators.py

# 4. 运行单元测试
pytest tests/
```

详细流程见 [WORKFLOW.md](WORKFLOW.md)

## 项目结构

```
gp-quant/
├── src/gp_quant/
│   ├── data/           # 数据层 (获取/处理/存储)
│   ├── strategy/       # 策略层 (技术指标/策略模板)
│   ├── backtest/       # 回测引擎
│   ├── ml/             # 机器学习 (特征/训练/预测)
│   └── harness/        # 交易调度 (风控/仓位/执行)
├── experiments/        # 实战脚本 (数据获取/清洗/指标/选股)
├── tests/              # 单元测试
├── requirements.txt
└── WORKFLOW.md
```

## 依赖

- Python >= 3.12
- pandas >= 2.0.3
- numpy >= 1.24.3
- baostock >= 0.8.8 (A 股数据，免费)
- akshare >= 1.14.0 (A 股数据，辅助)
- scikit-learn >= 1.3.0 (机器学习)
- torch (可选，神经网络)

## 许可证

MIT License
