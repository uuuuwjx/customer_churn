"""
最终评估：加载各模型最优参数，阈值优化，集成对比
"""
import pandas as pd
import numpy as np
import os
import sys
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import preprocess
from evaluate import (
    evaluate_fold, cross_validate, print_cv_summary,
    find_optimal_threshold, business_gain, ks_stat
)

# ── 路径设置 ────────────────────────────────────────────
script_dir = Path(__file__).resolve().parent
data_path = script_dir.parent / "data" / "processed" / "telco_churn_featured.parquet"
results_dir = script_dir.parent / "results"
reports_dir = script_dir.parent / "reports"
reports_dir.mkdir(parents=True, exist_ok=True)

# ── 加载数据 ────────────────────────────────────────────
df = pd.read_parquet(data_path)
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df['Churn']
)

# ── 辅助：加载最优参数或使用默认 ──────────────────────
def load_best_params(model_name, defaults):
    param_file = results_dir / f'{model_name}_best_params.json'
    if param_file.exists():
        with open(param_file) as f:
            return json.load(f)
    else:
        print(f"  [WARN]{param_file.name} 不存在，使用默认参数")
        return defaults

# ═══════════════════════════════════════════════════════════
# 1. 加载模型 & 预处理
# ═══════════════════════════════════════════════════════════
scale_pos_weight = (train_df['Churn'] == 0).sum() / (train_df['Churn'] == 1).sum()

models = {}
results_final = []

# ── Logistic Regression ──────────────────────────────────
print(">>> 加载 Logistic Regression ...")
X_train_lr, y_train_lr, scaler_lr = preprocess(train_df, encoding='onehot', fit=True)
X_test_lr, y_test_lr, _ = preprocess(test_df, encoding='onehot', scaler=scaler_lr, fit=False)

lr_defaults = {'C': 1.0, 'penalty': 'l2', 'solver': 'lbfgs', 'class_weight': 'balanced'}
lr_params = load_best_params('lr', lr_defaults)
lr = LogisticRegression(**lr_params, max_iter=2000, random_state=42)
lr.fit(X_train_lr, y_train_lr)
models['Logistic Regression'] = {
    'model': lr, 'X_test': X_test_lr, 'y_test': y_test_lr,
    'X_train': X_train_lr, 'y_train': y_train_lr
}

# ── Random Forest ────────────────────────────────────────
print(">>> 加载 Random Forest ...")
X_train_rf, y_train_rf, scaler_rf = preprocess(train_df, encoding='ordinal', fit=True)
X_test_rf, y_test_rf, _ = preprocess(test_df, encoding='ordinal', scaler=scaler_rf, fit=False)

rf_defaults = {'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 5,
               'min_samples_leaf': 2, 'max_features': 'sqrt', 'class_weight': 'balanced'}
rf_params = load_best_params('rf', rf_defaults)
rf = RandomForestClassifier(**rf_params, random_state=42, n_jobs=-1)
rf.fit(X_train_rf, y_train_rf)
models['Random Forest'] = {
    'model': rf, 'X_test': X_test_rf, 'y_test': y_test_rf,
    'X_train': X_train_rf, 'y_train': y_train_rf
}

# ── XGBoost ──────────────────────────────────────────────
print(">>> 加载 XGBoost ...")
xgb_defaults = {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.1,
                'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 1.0}
xgb_params = load_best_params('xgb', xgb_defaults)
xgb_model = xgb.XGBClassifier(
    **xgb_params, scale_pos_weight=scale_pos_weight,
    eval_metric='logloss', random_state=42, n_jobs=-1
)
xgb_model.fit(X_train_rf, y_train_rf)
models['XGBoost'] = {
    'model': xgb_model, 'X_test': X_test_rf, 'y_test': y_test_rf,
    'X_train': X_train_rf, 'y_train': y_train_rf
}

# ── LightGBM ─────────────────────────────────────────────
print(">>> 加载 LightGBM ...")
X_train_lgb, y_train_lgb, scaler_lgb = preprocess(train_df, encoding='none', fit=True)
X_test_lgb, y_test_lgb, _ = preprocess(test_df, encoding='none', scaler=scaler_lgb, fit=False)

lgb_defaults = {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.1,
                'num_leaves': 31, 'subsample': 0.8, 'colsample_bytree': 0.8,
                'reg_alpha': 0.1, 'reg_lambda': 1.0}
lgb_params = load_best_params('lgb', lgb_defaults)
lgb_cat_names = [
    'Contract', 'InternetService', 'MultipleLines', 'PaymentMethod', 'TenureGroup',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies'
]
lgb_cat_indices = [X_train_lgb.columns.get_loc(c) for c in lgb_cat_names if c in X_train_lgb.columns]

lgb_model = lgb.LGBMClassifier(
    **lgb_params,
    categorical_feature=lgb_cat_indices,
    random_state=42, n_jobs=-1, verbose=-1
)
lgb_model.fit(X_train_lgb, y_train_lgb)
models['LightGBM'] = {
    'model': lgb_model, 'X_test': X_test_lgb, 'y_test': y_test_lgb,
    'X_train': X_train_lgb, 'y_train': y_train_lgb
}

# ═══════════════════════════════════════════════════════════
# 2. 阈值优化 & 最终指标
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  阈值优化 & 最终评估")
print("=" * 70)

threshold_results = {}

for name, m in models.items():
    print(f"\n--- {name} ---")
    best_t, best_gain, tdf = find_optimal_threshold(
        m['model'], m['X_test'], m['y_test']
    )
    threshold_results[name] = {
        'best_threshold': best_t,
        'best_gain': best_gain,
        'df': tdf
    }
    print(f"  最优阈值: {best_t:.2f}  |  最大业务收益: ${best_gain:,.0f}")

    # 默认阈值 (0.5) 评估
    y_proba = m['model'].predict_proba(m['X_test'])[:, 1]
    y_pred_default = (y_proba >= 0.5).astype(int)
    gain_default, _ = business_gain(m['y_test'], y_proba, 0.5)
    print(f"  默认阈值 0.5 业务收益: ${gain_default:,.0f}  (提升 ${best_gain - gain_default:,.0f})")

# ═══════════════════════════════════════════════════════════
# 3. 最终指标汇总（各模型用最优阈值）
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  最终模型对比（Test Set，最优阈值）")
print("=" * 90)

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, auc, confusion_matrix
)

final_records = []
for name, m in models.items():
    best_t = threshold_results[name]['best_threshold']
    y_proba = m['model'].predict_proba(m['X_test'])[:, 1]
    y_pred = (y_proba >= best_t).astype(int)

    final_records.append({
        'Model': name,
        'Threshold': f"{best_t:.2f}",
        'Accuracy': accuracy_score(m['y_test'], y_pred),
        'Precision': precision_score(m['y_test'], y_pred),
        'Recall': recall_score(m['y_test'], y_pred),
        'F1': f1_score(m['y_test'], y_pred),
        'ROC-AUC': roc_auc_score(m['y_test'], y_proba),
        'KS': ks_stat(m['y_test'], y_proba),
        'Business Gain': threshold_results[name]['best_gain'],
        'y_proba': y_proba,
        'y_pred': y_pred,
    })

final_df = pd.DataFrame(final_records)

# 打印格式化表格
print_df = final_df[['Model', 'Threshold', 'Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC', 'KS', 'Business Gain']].copy()
for col in ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC', 'KS']:
    print_df[col] = print_df[col].apply(lambda x: f"{x:.4f}")
print_df['Business Gain'] = print_df['Business Gain'].apply(lambda x: f"${x:,.0f}")
print(print_df.to_string(index=False))

# ═══════════════════════════════════════════════════════════
# 4. 集成：简单加权平均
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  简单集成（加权平均概率）")
print("=" * 70)

# 收集所有模型的概率（对齐到 LR 测试集的索引）
proba_dict = {}
for name, m in models.items():
    # 各模型概率用各自测试集的索引对齐
    proba_dict[name] = pd.Series(
        m['model'].predict_proba(m['X_test'])[:, 1],
        index=m['y_test'].index
    )

proba_df = pd.DataFrame(proba_dict)

# 等权集成
proba_df['Ensemble_Equal'] = proba_df.mean(axis=1)

# 加权集成（按各模型 ROC-AUC 加权）
weights = {row['Model']: row['ROC-AUC'] for row in final_records}
total_w = sum(weights.values())
proba_df['Ensemble_Weighted'] = sum(
    proba_df[name] * weights[name] / total_w for name in weights
)

# 评估集成模型
y_true = y_test_lr  # 所有 y_test 相同
for ens_name in ['Ensemble_Equal', 'Ensemble_Weighted']:
    y_proba_ens = proba_df[ens_name].values
    y_pred_ens = (y_proba_ens >= 0.5).astype(int)

    print(f"\n  {ens_name}:")
    print(f"    Accuracy:  {accuracy_score(y_true, y_pred_ens):.4f}")
    print(f"    Precision: {precision_score(y_true, y_pred_ens):.4f}")
    print(f"    Recall:    {recall_score(y_true, y_pred_ens):.4f}")
    print(f"    F1:        {f1_score(y_true, y_pred_ens):.4f}")
    print(f"    ROC-AUC:   {roc_auc_score(y_true, y_proba_ens):.4f}")
    print(f"    KS:        {ks_stat(y_true, y_proba_ens):.4f}")

# ═══════════════════════════════════════════════════════════
# 5. 可视化
# ═══════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# 5.1 指标雷达对比
ax1 = axes[0, 0]
metrics_plot = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC', 'KS']
x = np.arange(len(metrics_plot))
bar_width = 0.2
colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']

for i, (_, row) in enumerate(final_df.iterrows()):
    values = [row[m] for m in metrics_plot]
    ax1.bar(x + i * bar_width, values, bar_width,
            label=row['Model'], color=colors[i], alpha=0.85)

ax1.set_xticks(x + bar_width * 1.5)
ax1.set_xticklabels(metrics_plot, fontsize=9)
ax1.set_ylim(0, 1)
ax1.set_title('Model Metrics Comparison (Optimal Threshold)', fontsize=13, fontweight='bold')
ax1.legend(loc='lower right', fontsize=8)
ax1.grid(axis='y', alpha=0.3)

# 5.2 业务收益对比
ax2 = axes[0, 1]
gains = [row['Business Gain'] for _, row in final_df.iterrows()]
names = [row['Model'] for _, row in final_df.iterrows()]
bars = ax2.bar(names, gains, color=colors, alpha=0.85, edgecolor='white')
ax2.set_title('Business Gain by Model', fontsize=13, fontweight='bold')
ax2.set_ylabel('Business Gain ($)')
ax2.tick_params(axis='x', rotation=15)
for bar, gain in zip(bars, gains):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
             f'${gain:,.0f}', ha='center', fontsize=10, fontweight='bold')

# 5.3 阈值-收益曲线
ax3 = axes[1, 0]
for name, m in models.items():
    tdf = threshold_results[name]['df']
    ax3.plot(tdf['threshold'], tdf['business_gain'], marker='.', label=name)
    best_t = threshold_results[name]['best_threshold']
    ax3.axvline(x=best_t, linestyle='--', alpha=0.3,
                color=colors[list(models.keys()).index(name)])
ax3.set_xlabel('Threshold')
ax3.set_ylabel('Business Gain ($)')
ax3.set_title('Threshold vs Business Gain', fontsize=13, fontweight='bold')
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3)

# 5.4 详细分类表格
ax4 = axes[1, 1]
ax4.axis('off')
table_data = []
for _, row in print_df.iterrows():
    table_data.append(row.tolist())
table = ax4.table(
    cellText=table_data,
    colLabels=print_df.columns.tolist(),
    cellLoc='center',
    loc='center',
)
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.2, 1.6)
ax4.set_title('Detailed Metrics Table', fontsize=13, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig(reports_dir / 'model_comparison_final.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n[OK] 最终对比图已保存到 {reports_dir / 'model_comparison_final.png'}")

# ═══════════════════════════════════════════════════════════
# 6. ROC 曲线（四模型叠加）
# ═══════════════════════════════════════════════════════════
fig_roc, ax_roc = plt.subplots(figsize=(8, 7))

for i, (name, m) in enumerate(models.items()):
    y_proba = m['model'].predict_proba(m['X_test'])[:, 1]
    fpr, tpr, _ = roc_curve(m['y_test'], y_proba)
    roc_auc = auc(fpr, tpr)
    ax_roc.plot(fpr, tpr, color=colors[i], lw=2,
                label=f'{name} (AUC = {roc_auc:.4f})')

ax_roc.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random (AUC = 0.50)')
ax_roc.set_xlim([0.0, 1.0])
ax_roc.set_ylim([0.0, 1.05])
ax_roc.set_xlabel('False Positive Rate', fontsize=12)
ax_roc.set_ylabel('True Positive Rate', fontsize=12)
ax_roc.set_title('ROC Curves - All Models', fontsize=14, fontweight='bold')
ax_roc.legend(loc='lower right', fontsize=10)
ax_roc.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(reports_dir / 'roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"[OK] ROC 曲线已保存到 {reports_dir / 'roc_curves.png'}")

# ═══════════════════════════════════════════════════════════
# 7. 混淆矩阵（每模型一个子图）
# ═══════════════════════════════════════════════════════════
fig_cm, axes_cm = plt.subplots(2, 2, figsize=(10, 9))
axes_cm = axes_cm.flatten()

for i, (name, m) in enumerate(models.items()):
    best_t = threshold_results[name]['best_threshold']
    y_proba = m['model'].predict_proba(m['X_test'])[:, 1]
    y_pred = (y_proba >= best_t).astype(int)
    cm = confusion_matrix(m['y_test'], y_pred)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'],
                ax=axes_cm[i], cbar=False)
    axes_cm[i].set_title(f'{name} (Threshold={best_t:.2f})', fontsize=12, fontweight='bold')
    axes_cm[i].set_ylabel('Actual', fontsize=10)
    axes_cm[i].set_xlabel('Predicted', fontsize=10)

plt.tight_layout()
plt.savefig(reports_dir / 'confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"[OK] 混淆矩阵已保存到 {reports_dir / 'confusion_matrices.png'}")

# ── 保存最终结果表 ──────────────────────────────────────
final_df.drop(columns=['y_proba', 'y_pred']).to_csv(
    results_dir / 'final_model_comparison.csv', index=False
)
print(f"[OK] 最终结果表已保存到 {results_dir / 'final_model_comparison.csv'}")
