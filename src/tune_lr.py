"""
Logistic Regression 超参数调优（GridSearchCV，小空间穷举即可）
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
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import preprocess
from evaluate import cross_validate, print_cv_summary

# ── 路径设置 ────────────────────────────────────────────
script_dir = Path(__file__).resolve().parent
data_path = script_dir.parent / "data" / "processed" / "telco_churn_featured.parquet"
results_dir = script_dir.parent / "results"
results_dir.mkdir(parents=True, exist_ok=True)

# ── 加载数据 ────────────────────────────────────────────
df = pd.read_parquet(data_path)
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df['Churn']
)

X_train, y_train, scaler = preprocess(train_df, encoding='onehot', fit=True)
X_test, y_test, _ = preprocess(test_df, encoding='onehot', scaler=scaler, fit=False)

# ── 基线评估 ────────────────────────────────────────────
def build_lr(**kwargs):
    defaults = dict(
        C=1.0, penalty='l2', solver='lbfgs',
        max_iter=1000, class_weight='balanced', random_state=42
    )
    defaults.update(kwargs)
    return LogisticRegression(**defaults)

print("\n>>> 基线模型 CV 评估 ...")
_, baseline_summary = cross_validate(build_lr, X_train, y_train, n_splits=5)
print_cv_summary(baseline_summary, "Logistic Regression Baseline")
baseline_auc = baseline_summary[baseline_summary['fold'] == 'mean']['roc_auc'].values[0]
baseline_recall = baseline_summary[baseline_summary['fold'] == 'mean']['recall'].values[0]
print(f"\n  基线 ROC-AUC: {baseline_auc:.4f}  |  基线 Recall: {baseline_recall:.4f}")

# ═══════════════════════════════════════════════════════════
# GridSearchCV（参数空间小，直接穷举）
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  Logistic Regression — 网格搜索")
print("=" * 60)

# 分两组搜索：l1 和 l2 的 solver 不兼容
param_grid = [
    {  # l2 + lbfgs（支持大数据集）
        'C': [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100],
        'penalty': ['l2'],
        'solver': ['lbfgs'],
        'class_weight': ['balanced', None],
    },
    # {  # l1 + saga
    #     'C': [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100],
    #     'penalty': ['l1'],
    #     'solver': ['saga'],
    #     'class_weight': ['balanced', None],
    # },
    # {  # elasticnet + saga
    #     'C': [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100],
    #     'penalty': ['elasticnet'],
    #     'l1_ratio': [0.25, 0.5, 0.75],
    #     'solver': ['saga'],
    #     'class_weight': ['balanced', None],
    #     'max_iter': [5000],
    # },
]

search = GridSearchCV(
    LogisticRegression(max_iter=2000, random_state=42),
    param_grid=param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1,
)
search.fit(X_train, y_train)

print(f"\n最佳 ROC-AUC: {search.best_score_:.4f}")
print(f"最佳参数: {json.dumps(search.best_params_, indent=2)}")

# ── 最优模型 ───────────────────────────────────────────
best_params = search.best_params_
lr_best = LogisticRegression(**best_params, max_iter=2000, random_state=42)
lr_best.fit(X_train, y_train)

# ── CV 评估 ────────────────────────────────────────────
def build_lr_best():
    return LogisticRegression(**best_params, max_iter=2000, random_state=42)

print("\n>>> 最优模型 CV 评估 ...")
_, tuned_summary = cross_validate(build_lr_best, X_train, y_train, n_splits=5)
print_cv_summary(tuned_summary, "Logistic Regression Tuned")

tuned_auc = tuned_summary[tuned_summary['fold'] == 'mean']['roc_auc'].values[0]
tuned_recall = tuned_summary[tuned_summary['fold'] == 'mean']['recall'].values[0]

print(f"\n  ROC-AUC: {baseline_auc:.4f} → {tuned_auc:.4f}  ({tuned_auc - baseline_auc:+.4f})")
print(f"  Recall:   {baseline_recall:.4f} → {tuned_recall:.4f}  ({tuned_recall - baseline_recall:+.4f})")

# ── 保存 ────────────────────────────────────────────────
joblib.dump(lr_best, results_dir / 'lr_best_model.joblib')
joblib.dump(scaler, results_dir / 'lr_best_scaler.joblib')

with open(results_dir / 'lr_best_params.json', 'w') as f:
    json.dump(best_params, f, indent=2)

print(f"\n[OK] 最优模型和参数已保存到 {results_dir}/")
print(f"   最优 ROC-AUC (CV): {tuned_auc:.4f}")
