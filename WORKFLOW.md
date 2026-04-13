# gp-quant 实战流程

按框架 6 大模块顺序，每一步产出中间文件。

## Phase 1: 数据获取 (data/fetcher)
- 选股 → `experiments/data/stock_pool.txt`
- 拉取历史日线(OHLCV) → `experiments/data/stock_pool_1y.csv`
- 数据源：baostock（前复权日线，adjustflag="2"）
- 日期范围：10年（2016-01-01 ~ 至今）
- 产出：约 20 万行，91 只 × ~2500 天

## Phase 2: 数据清洗 (data/processor)
- 输入：`stock_pool_1y.csv`（raw OHLCV，带 symbol）
- **按 symbol 分组清洗**，每只独立处理，输出 `cleaned/<symbol>.csv`
- 步骤（按顺序）：
  1. **类型转换** — OHLCV 列转数值，symbol 保留 6 位字符串（zfill）
  2. **去空值** — dropna(subset=['close'])
  3. **去重** — 同日期只留最后一条
  4. **排序** — 按 date 升序
  5. **衍生列** — change（涨跌幅%）、change_abs、range、range_pct
  6. **停牌剔除** — volume 为 NaN 的行（baostock 返回占位价格，无真实成交）
  7. **异常值** — range_pct > 50% 排除
  8. **OHLC 一致性** — 确保 high >= low, high >= open, high >= close, low <= open, low <= close
  9. **多周期收益率** — return_1d, return_5d, return_10d, return_20d
- 同时输出合并版：`experiments/data/cleaned_pool.csv`
- **不做标准化**（留给 Phase 5 ML 特征工程）

## Phase 3: 技术指标 (strategy/indicators)
- 输入：清洗后数据
- 计算：MA5/10/20/60, MACD, RSI, 布林带, ATR, ADX, Stochastic, CCI, WR, OBV, MFI, 成交量比率, 波动率
- 输出：`experiments/data/indicators_pool.csv`（独立文件在 `experiments/data/indicators/<symbol>.csv`）

## Phase 4: 策略开发 + 回测 (backtest/engine)
- 基于指标信号构建策略（MACD金叉/RSI超卖/综合评分等）
- 回测引擎：初始资金/费率/仓位管理/胜率/夏普/最大回撤
- 输出：`experiments/results/backtest_report.json`

## Phase 5: 特征工程 (ml/features)
- 价格特征/滞后特征/技术指标特征
- 目标变量：未来N日收益率+方向
- 特征缩放/PCA降维
- 输出：`experiments/ml/features.csv`

## Phase 6: 模型训练 (ml/trainer)
- 分类（涨跌方向）或回归（收益率）
- scikit-learn 模型
- 输出：`experiments/ml/model.pkl` + 评估报告

## Phase 7: 交易调度 (harness/engine)
- 风控配置/仓位 sizing/模拟执行
- 实盘/模拟盘调度

---

## 当前进度
- [x] Phase 1: stock_pool_1y.csv（91只×~2500天，197,387行）
- [x] Phase 2: cleaned_pool.csv（197,273行，91只，16列，含OHLC一致性检查+多周期收益率）
- [x] Phase 3: indicators_pool.csv（194,427行，91只，40列，24个技术指标）
- [ ] Phase 4: 策略开发 + 回测
- [ ] Phase 5: 特征工程
- [ ] Phase 6: 模型训练
- [ ] Phase 7: 交易调度
