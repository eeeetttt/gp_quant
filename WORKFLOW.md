# gp-quant 实战流程

## 训练流程（当前版本 v5）

### Phase 0: 选股
- 基于 baostock `query_stock_industry()` 获取全部 A 股行业分类
- 覆盖 5515 只股票，83 个证监会行业分类
- 输出：`pipeline/data/stock_pool.txt`
- 训练时通过 `--pool 500` 控制子集大小

### Phase 1: 数据获取
- 拉取历史日线(OHLCV) → `pipeline/data/raw_pool.csv`
- 数据源：baostock（前复权日线），10 年跨度（2016-2026）
- 产出：~960 万行，5176 只股票 × ~2500 天

### Phase 2: 数据清洗
- 输入：`raw_pool.csv`
- 按 symbol 分组清洗，每只独立处理
- 步骤：类型转换 → 去空值 → 去重 → 排序 → 衍生列 → 停牌剔除 → 异常值过滤 → OHLC 一致性检查 → 多周期收益率(return_1d/5d/10d/20d)
- 输出：`pipeline/data/cleaned_pool.csv`（~2.2GB）

### Phase 3: 技术指标
- 输入：清洗后数据
- 计算 40+ 指标：MA, MACD, RSI, 布林带, ATR, ADX, Stochastic(K/D), CCI, WR, OBV, MFI, 成交量比率, 波动率等
- 输出：`pipeline/data/indicators_pool.csv`（~6.3GB）

### Phase 4a: 特征工程
- 输入：指标数据 + 上证指数（大盘基准）
- 生成特征：
  - 价格特征、滞后特征（价格/收益率/指标滞后 1d/5d）
  - 大盘特征：上证指数日收益率、5日收益率、量变率
  - 相对强弱：个股 vs 大盘超额收益（alpha_1d/5d）、滚动 beta_20d
  - 资金流：量价配合、资金流累积、换手率
  - 时间特征：月份、星期、月初/月末
  - 价格动量：价格位置、趋势强度、线性回归斜率
- 去冗余：自动删除 corr=1.0 重复特征（wr, volatility_annual, market_return_1d/5d）
- 输出：`pipeline/ml/features.csv`（~13.5GB，9.6M 行，66 特征）

### Phase 4b: 板块特征（Sector Features）
- 输入：`indicators_pool.csv` + baostock 行业分类
- 计算 8 个板块特征：
  - `industry`：所属行业（83 个证监会行业）
  - `sector_return_1d`：行业当日等权平均收益率
  - `sector_return_5d`：行业 5 日收益率
  - `sector_momentum_20d`：行业 20 日动量
  - `sector_volatility_20d`：行业 20 日波动率
  - `sector_alpha_1d`：个股收益 - 行业收益
  - `sector_rank_in_industry`：个股在行业内当日收益率排名百分位
  - `sector_breadth_5d`：行业近 5 日上涨股票比例
- 输出：`pipeline/ml/sector_features.csv`（~1.7GB）

### Phase 5: 模型训练（step5e_all.py）

#### 目标定义
- **买入阈值**：未来 N 日收益率 ≥ 4%
- **卖出阈值**：未来 N 日收益率 ≤ -5%
- 三分类：1（买入）、0（卖出）、-1（中性，训练时排除）

#### 多周期模型
| 模型 | 周期 | 角色 | 样本量 | AUC |
|------|------|------|--------|-----|
| mid_10d | 10d | 主力 | 4.6M | 0.5851 |
| mid_20d | 20d | 主力 | 5.8M | 0.5429 |
| short_5d | 5d | 辅助 | 3.3M | 0.5948 |

#### 训练配置
- **Walk-forward 验证**：4 splits，expanding window
- **超参搜索**：Optuna（TPE sampler），首 split 上搜索
  - mid_10d/mid_20d: 20 trials
  - short_5d: 100 trials
- **配置文件**：`pipeline/config.yaml`（超参空间、目标阈值、回测参数均可在此修改）
- **超参搜索空间**：

| 参数 | 类型 | 范围 | 说明 |
|------|------|------|------|
| `n_estimators` | int | 200–600 (step=50) | 树数量 |
| `max_depth` | int | 3–10 | 最大深度 |
| `learning_rate` | float | 0.005–0.1 (log) | 学习率 |
| `num_leaves` | int | 8–100 | 叶子数（LightGBM） |
| `min_child_samples` | int | 50–500 | 最小叶子样本数 |
| `subsample` | float | 0.4–1.0 | 行采样比例 |
| `colsample_bytree` | float | 0.3–1.0 | 列采样比例 |
| `reg_alpha` | float | 0.0–5.0 | L1 正则 |
| `reg_lambda` | float | 0.0–10.0 | L2 正则 |

- **目标阈值**：买入 ≥4%，卖出 ≤-5%（config.yaml 中可调）
- **模型**：LightGBM (CPU, OpenMP 18 核) + XGBoost (CPU, tree_method=hist)
  - 超参在 LightGBM 上搜索，XGBoost 共享同一组最优参数
  - 最终集成：LGB + XGB 各占 50% 权重
- **Hybrid 集成**：按 AUC 加权（10d: 0.52, 20d: 0.48）
- **时间泄露防护**：最终模型训练排除所有 WF 测试日期

#### 运行命令
```bash
# 编辑配置：修改超参空间、目标阈值、回测参数
# vim pipeline/config.yaml

# 完整训练（3 个模型 + Hybrid）
python pipeline/step5e_all.py --pool 500 --n-trials 20

# 只跑 short_5d（100 trials，由 config.yaml 控制）
python pipeline/step5e_all.py --pool 500 --n-trials 20
```

#### 输出
- 模型：`pipeline/models/model_all.pkl`
- 报告：`pipeline/ml/all_report.json`
- 回测：`pipeline/results/backtest_report.json`

### Phase 5b: 独立训练（已废弃，集成到 step5e_all.py）
- `step5b_train_optimized.py` — 优化版训练
- `step5c_deoverfit.py` — 过拟合检查
- `step5d_walkforward.py` — 独立 Walk-forward 验证

### Phase 6: 回测验证
- 基于模型预测概率生成买卖信号
- 止损：-5%，止盈：4%，最大持有 20 天
- 信号确认：双模型同时看多才买入
- 输出：
  - 单模型：1176 笔，胜率 61.5%，收益 134.6%，回撤 -42.9%
  - Hybrid：1093 笔，胜率 63.0%，收益 91.6%，回撤 -44.9%

---

## 目录映射

| 脚本 | 描述 |
|------|------|
| `pipeline/step0_screen.py` | 选股 |
| `pipeline/step1_fetch.py` | 数据获取 |
| `pipeline/step2_clean.py` | 数据清洗 |
| `pipeline/step3_indicators.py` | 技术指标 |
| `pipeline/step4_features.py` | 特征工程 |
| `pipeline/step4b_sector.py` | 板块特征 |
| `pipeline/step5e_all.py` | 多周期模型训练（主力） |
| `pipeline/step6b_backtest_optimized.py` | 回测 |
| `pipeline/step6c_overfit_check.py` | 过拟合检查 |

## 硬件环境

| 组件 | 规格 | 用途 |
|------|------|------|
| CPU | Apple M5 Max 18 核 (6P+12E) | LightGBM + XGBoost 训练 |
| GPU | Apple M5 Max 40 核 Metal | 暂未使用（见下方） |
| 内存 | 大内存 | 9.6M 行特征数据处理 |

### GPU 说明
LightGBM 和 XGBoost 官方均不支持 Apple Silicon Metal GPU：
- LightGBM `device='mps'` → 崩溃
- XGBoost `device='gpu'` → 静默降级 CPU
- MLX-Boosting → 可用但比 CPU 慢 7.8x
- 当前方案：18 核 CPU + OpenMP 多线程，训练 ~13 分钟

## 性能基线

| 版本 | 股票数 | 特征数 | 目标 | AUC |
|------|--------|--------|------|-----|
| v1 baseline | 91 | ~20 | direction_0pct | 0.5433 |
| v4 优化 | 500 | 63 | quantile_top30 | 0.5808 |
| **v5 当前** | **5176** | **66** | **4%/-5% 三分类** | **0.5851** (mid_10d) |
