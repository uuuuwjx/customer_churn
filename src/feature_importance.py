"""
特征重要性分析：多模型对比
"""
import pandas as pd
import numpy as np
import os
import sys
import joblib
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import preprocess
from sklearn.model_selection import train_test_split

# ── 路径设置 ────────────────────────────────────────────
script_dir = Path(__file__).resolve().parent
data_path = script_dir.parent / "data" / "processed" / "telco_churn_featured.parquet"
results_dir = script_dir.parent / "results"
reports_dir = script_dir.parent / "reports"
reports_dir.mkdir(parents=True, exist_ok=True)

# ── 加载数据 ────────────────────────────────────────────
df = pd.read_parquet(data_path)
train_df, _ = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Churn'])

# ── 加载最优模型并计算特征重要性 ───────────────────────
importance_records = []

# --- Logistic Regression ---
print(">>> Logistic Regression 特征重要性（|coef|）...")
X_lr, y_lr, _ = preprocess(train_df, encoding='onehot', fit=True)
lr = joblib.load(results_dir / 'lr_best_model.joblib')
lr_imp = pd.DataFrame({
    'feature': X_lr.columns,
    'importance': np.abs(lr.coef_[0]),
    'model': 'Logistic Regression'
}).sort_values('importance', ascending=False)
importance_records.append(lr_imp)
print(f"  Top 5: {list(lr_imp['feature'].head(5))}")

# --- Random Forest ---
print(">>> Random Forest 特征重要性...")
X_rf, y_rf, _ = preprocess(train_df, encoding='ordinal', fit=True)
rf = joblib.load(results_dir / 'rf_best_model.joblib')
rf_imp = pd.DataFrame({
    'feature': X_rf.columns,
    'importance': rf.feature_importances_,
    'model': 'Random Forest'
}).sort_values('importance', ascending=False)
importance_records.append(rf_imp)
print(f"  Top 5: {list(rf_imp['feature'].head(5))}")

# --- XGBoost ---
print(">>> XGBoost 特征重要性...")
xgb_model = joblib.load(results_dir / 'xgb_best_model.joblib')
xgb_imp = pd.DataFrame({
    'feature': X_rf.columns,  # XGBoost uses same ordinal encoding as RF
    'importance': xgb_model.feature_importances_,
    'model': 'XGBoost'
}).sort_values('importance', ascending=False)
importance_records.append(xgb_imp)
print(f"  Top 5: {list(xgb_imp['feature'].head(5))}")

# --- LightGBM ---
print(">>> LightGBM 特征重要性...")
X_lgb, y_lgb, _ = preprocess(train_df, encoding='none', fit=True)
lgb_model = joblib.load(results_dir / 'lgb_best_model.joblib')
# LightGBM 用 split 次数作为重要性（更稳定）
lgb_imp = pd.DataFrame({
    'feature': X_lgb.columns,
    'importance': lgb_model.feature_importances_,
    'model': 'LightGBM'
}).sort_values('importance', ascending=False)
importance_records.append(lgb_imp)
print(f"  Top 5: {list(lgb_imp['feature'].head(5))}")

# ═══════════════════════════════════════════════════════════
# 汇总：每个特征取四模型平均排名
# ═══════════════════════════════════════════════════════════
all_imp = pd.concat(importance_records, ignore_index=True)

# 对每个模型内部排名
all_imp['rank'] = all_imp.groupby('model')['importance'].rank(ascending=False, method='min')

# 每个特征的平均排名（四模型）
avg_rank = all_imp.groupby('feature')['rank'].mean().sort_values()
avg_importance = all_imp.groupby('feature')['importance'].mean()

# 综合排名表
summary = pd.DataFrame({
    'feature': avg_rank.index,
    'avg_rank': avg_rank.values,
    'avg_importance': avg_importance[avg_rank.index].values
}).sort_values('avg_importance', ascending=False)

# ═══════════════════════════════════════════════════════════
# 打印 Top 15
# ═══════════════════════════════════════════════════════════
top_n = 15
print("\n" + "=" * 70)
print(f"  综合特征重要性 Top {top_n}（四模型平均排名）")
print("=" * 70)
print(f"{'Rank':<5} {'Feature':<25} {'Avg Importance':<16} {'Avg Rank':<10}")
print("-" * 56)
for i, (_, row) in enumerate(summary.head(top_n).iterrows(), 1):
    print(f"{i:<5} {row['feature']:<25} {row['avg_importance']:<16.4f} {row['avg_rank']:<10.1f}")



# ═══════════════════════════════════════════════════════════
# 可视化：3 张图
# ═══════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(22, 6))

# 1) 综合重要性 Top 10 柱状图
ax1 = axes[0]
top_10 = summary.head(10)
colors1 = plt.cm.Reds_r(np.linspace(0.35, 0.9, 10))
ax1.barh(range(9, -1, -1), top_10['avg_importance'].values[::-1],
         color=colors1[::-1], edgecolor='white', height=0.7)
ax1.set_yticks(range(9, -1, -1))
ax1.set_yticklabels(top_10['feature'].values[::-1], fontsize=10)
ax1.set_xlabel('Average Importance')
ax1.set_title('Top 10 Features (4-Model Average)', fontsize=14, fontweight='bold')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# 2) 各模型 Top 10 热力图
ax2 = axes[1]
top_10_features = summary.head(10)['feature'].tolist()
heatmap_data = []
for model_name in ['Logistic Regression', 'Random Forest', 'XGBoost', 'LightGBM']:
    model_imp = all_imp[all_imp['model'] == model_name].set_index('feature')
    row = []
    for f in top_10_features:
        row.append(model_imp.loc[f, 'importance'] if f in model_imp.index else 0)
    row = np.array(row)
    row = row / row.max() if row.max() > 0 else row
    heatmap_data.append(row)

im = ax2.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
ax2.set_xticks(range(len(top_10_features)))
ax2.set_xticklabels(top_10_features, rotation=45, ha='right', fontsize=9)
ax2.set_yticks(range(4))
ax2.set_yticklabels(['LR', 'RF', 'XGBoost', 'LightGBM'], fontsize=10)
ax2.set_title('Feature Importance Heatmap (Normalized per Model)', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax2, shrink=0.85)

# 3) 四模型 Top 8 分组柱状图
ax3 = axes[2]
top_8 = summary.head(8)['feature'].tolist()
x = np.arange(len(top_8))
bar_width = 0.2
model_colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']
for i, model_name in enumerate(['Logistic Regression', 'Random Forest', 'XGBoost', 'LightGBM']):
    model_imp = all_imp[all_imp['model'] == model_name].set_index('feature')
    values = [model_imp.loc[f, 'importance'] if f in model_imp.index else 0 for f in top_8]
    # 组内归一化
    max_v = max(values) if max(values) > 0 else 1
    values = [v / max_v for v in values]
    ax3.bar(x + i * bar_width, values, bar_width,
            label=model_name, color=model_colors[i], alpha=0.85)

ax3.set_xticks(x + bar_width * 1.5)
ax3.set_xticklabels(top_8, rotation=45, ha='right', fontsize=9)
ax3.set_ylabel('Normalized Importance')
ax3.set_title('Top 8 Features by Model', fontsize=14, fontweight='bold')
ax3.legend(fontsize=8, loc='upper right')
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(reports_dir / 'feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"\n[OK] 特征重要性图已保存到 {reports_dir / 'feature_importance.png'}")

# ── 保存重要性表 ────────────────────────────────────────
summary.to_csv(results_dir / 'feature_importance.csv', index=False)
print(f"[OK] 特征重要性表已保存到 {results_dir / 'feature_importance.csv'}")
