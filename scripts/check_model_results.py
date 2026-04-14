"""详细检查训练结果"""
import json
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

with open('experiments/ml/train_report.json') as f:
    report = json.load(f)

print("=== 模型结果 ===")
for name, metrics in report['all_results'].items():
    print(f"\n{name}:")
    print(f"  时间外样本: acc={metrics['time_test']['accuracy']} f1={metrics['time_test']['f1']}")
    print(f"  完全外样本: acc={metrics['unseen']['accuracy']} f1={metrics['unseen']['f1']}")

print("\n=== 训练集特征重要性 Top 10 ===")
for f in report['top_20_features'][:10]:
    print(f"  {f['name']:30s} {f['importance']:.4f}")

# 检查特征和目标变量的关系
df = pd.read_csv('experiments/ml/features.csv', dtype={'symbol': str})
print("\n=== 特征与 target_direction_5d 的相关性 ===")
for col in ['return_5d', 'bb_position', 'k', 'wr', 'return_10d', 'cci',
            'market_return_5d', 'alpha_5d', 'rsi', 'macd']:
    if col in df.columns:
        corr = df[col].corr(df['target_direction_5d'])
        print(f"  {col:30s} corr={corr:.4f}")

# 看混淆矩阵
import joblib
model_data = joblib.load('experiments/models/model.pkl')
model = model_data['model']

feature_cols = model_data['feature_cols']
unseen = df[~df['symbol'].isin(report.get('unseen_stocks', []))].dropna(subset=feature_cols + ['target_direction_5d'])

# 手动计算几个股票的混淆矩阵
symbols = sorted(df['symbol'].unique())
test_syms = symbols[-5:]  # 最后5只股票
test_df = df[df['symbol'].isin(test_syms)].dropna(subset=feature_cols + ['target_direction_5d'])
X = test_df[feature_cols].values
y = test_df['target_direction_5d'].values
X_scaled = model_data['scaler'].transform(X)
y_pred = model.predict(X_scaled)

cm = confusion_matrix(y, y_pred)
print(f"\n=== 混淆矩阵（最后5只未完全见股票）===")
print(f"  预测/实际  涨(0)  跌(1)")
print(f"  涨(0)       {cm[0,0]:5d}  {cm[0,1]:5d}")
print(f"  跌(1)       {cm[1,0]:5d}  {cm[1,1]:5d}")
print(f"  准确率: {(cm[0,0]+cm[1,1])/cm.sum():.4f}")

# 看错分类的样本
wrong = test_df.iloc[y != y_pred]
print(f"\n错分类样本: {len(wrong)} / {len(test_df)}")
if len(wrong) > 0:
    print(wrong[['date', 'symbol', 'return_5d', 'target_direction_5d']].head(10).to_string())
