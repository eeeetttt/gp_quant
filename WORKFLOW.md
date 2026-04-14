# gp-quant 实战流程

## 训练流程

### Phase 0: 选股
- 筛选沪深 300 成分股 → `experiments/data/stock_pool.txt`

### Phase 1: 数据获取
- 拉取历史日线(OHLCV) → `experiments/data/stock_pool_1y.csv`
- 数据源：baostock（前复权日线），10 年跨度
- 产出：~20 万行，91 只 × ~2500 天

### Phase 2: 数据清洗
- 输入：`stock_pool_1y.csv`（raw OHLCV）
- **按 symbol 分组清洗**，每只独立处理
- 步骤：类型转换 → 去空值 → 去重 → 排序 → 衍生列 → 停牌剔除 → 异常值过滤 → OHLC 一致性检查 → 多周期收益率
- 输出：`experiments/data/cleaned/<symbol>.csv`（独立）+ `cleaned_pool.csv`（合并）

### Phase 3: 技术指标
- 输入：清洗后数据
- 计算 24 个指标：MA5/10/20/60, MACD, RSI, 布林带, ATR, ADX, Stochastic, CCI, WR, OBV, MFI, 成交量比率, 波动率
- 输出：`experiments/data/indicators/<symbol>.csv` + `indicators_pool.csv`

### Phase 4: 特征工程
- 输入：指标数据 + 上证指数（大盘基准）
- 生成：
  - 价格特征、滞后特征（价格/收益率/指标滞后）
  - **大盘特征**：上证指数日收益率、5日收益率、量变率
  - **相对强弱**：个股 vs 大盘超额收益（alpha_1d/5d）、滚动 beta_20d
  - **资金流**：量价配合（放量上涨/缩量下跌）、资金流 5日/20日累积、换手率
  - **时间特征**：月份、星期、月初/月末、季度
  - **价格动量**：20日/60日价格位置、趋势强度、线性回归斜率
- 目标变量：未来 N 日收益率+方向
- 输出：`experiments/ml/features.csv` + `feature_report.json`

### Phase 5: 模型训练
- 输入：特征数据
- 任务：分类（预测未来涨跌方向）
- 模型：Random Forest, Gradient Boosting 等
- 评估：准确率、F1、特征重要性
- 输出：`models/model.pkl` + 评估报告

### Phase 6: 回测验证
- 输入：训练好的模型 + 历史数据
- 方式：用模型预测生成买卖信号，在历史数据上模拟交易
- 指标：总收益率、胜率、夏普比率、最大回撤、盈亏比
- 输出：`experiments/results/backtest_report.json`

### Phase 7: 模型导出
- 导出训练好的模型文件，供推理使用
- 同时导出特征工程 pipeline（标准化器、特征列等）

---

## 推理流程（openclaw 调用）

### 调用方式
1. 给定股票代码
2. 获取最新行情数据（baostock）
3. 数据清洗 + 计算技术指标
4. 构建特征（复用训练时的特征工程 pipeline）
5. 加载 `models/model.pkl` 预测
6. 输出预测结果：涨跌方向 + 置信度 + 技术指标信号 + 买卖建议

### 输出格式
```json
{
  "symbol": "600519",
  "prediction": "BUY",
  "probability": 0.73,
  "indicators": {
    "rsi": 45.2,
    "macd_signal": "HOLD",
    "bb_signal": "BUY"
  }
}
```

---

## 当前进度
- [x] Phase 1: 数据获取 — stock_pool_1y.csv（197,387 行）
- [x] Phase 2: 数据清洗 — cleaned_pool.csv（197,273 行）
- [x] Phase 3: 技术指标 — indicators_pool.csv（194,427 行，40 列）
- [ ] Phase 4: 特征工程
- [ ] Phase 5: 模型训练
- [ ] Phase 6: 回测验证
- [ ] Phase 7: 模型导出
- [ ] 推理脚本

---

## 目录映射

| Phase | 描述 | 脚本 |
|-------|------|------|
| 0 | 选股 | `pipeline/step0_screen.py` |
| 1 | 数据获取 | `pipeline/step1_fetch.py` |
| 2 | 数据清洗 | `pipeline/step2_clean.py` |
| 3 | 技术指标 | `pipeline/step3_indicators.py` |
| 4 | 特征工程 | `pipeline/step4_features.py` |
| 5 | 模型训练 | `pipeline/step5_train_baseline.py` |
| 6 | 回测验证 | `pipeline/step6_backtest.py` |
| 7 | 模型导出 | `pipeline/step7_export.py` |

调试脚本：`scripts/`
