"""
Random Forest 超参数调优（一轮 RandomizedSearchCV）
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
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

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

X_train, y_train, scaler = preprocess(train_df, encoding='ordinal', fit=True)
X_test, y_test, _ = preprocess(test_df, encoding='ordinal', scaler=scaler, fit=False)

# ── 基线评估 ────────────────────────────────────────────
def build_rf(**kwargs):
    defaults = dict(
        n_estimators=200, max_depth=10, min_samples_split=5,
        min_samples_leaf=2, max_features='sqrt',
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    defaults.update(kwargs)
    return RandomForestClassifier(**defaults)

print("\n>>> 基线模型 CV 评估 ...")
_, baseline_summary = cross_validate(build_rf, X_train, y_train, n_splits=5)
print_cv_summary(baseline_summary, "Random Forest Baseline")
baseline_auc = baseline_summary[baseline_summary['fold'] == 'mean']['roc_auc'].values[0]
baseline_recall = baseline_summary[baseline_summary['fold'] == 'mean']['recall'].values[0]
print(f"\n  基线 ROC-AUC: {baseline_auc:.4f}  |  基线 Recall: {baseline_recall:.4f}")

# ═══════════════════════════════════════════════════════════
# RandomizedSearchCV
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  Random Forest — 随机搜索")
print("=" * 60)

param_dist = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [5, 8, 10, 12, 15, 18, 20, None],
    'min_samples_split': [2, 5, 10, 15, 20],
    'min_samples_leaf': [1, 2, 4, 6, 8, 10],
    'max_features': ['sqrt', 'log2', None],
    'class_weight': ['balanced', 'balanced_subsample', None],
}

search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_distributions=param_dist,
    n_iter=30,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='roc_auc',
    n_jobs=-1,
    random_state=42,
    verbose=1,
)
search.fit(X_train, y_train)

print(f"\n最佳 ROC-AUC: {search.best_score_:.4f}")
print(f"最佳参数: {json.dumps(search.best_params_, indent=2)}")

# ── 最优模型 ───────────────────────────────────────────
best_params = search.best_params_
rf_best = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
rf_best.fit(X_train, y_train)

# ── CV 评估 ────────────────────────────────────────────
def build_rf_best():
    return RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)

print("\n>>> 最优模型 CV 评估 ...")
_, tuned_summary = cross_validate(build_rf_best, X_train, y_train, n_splits=5)
print_cv_summary(tuned_summary, "Random Forest Tuned")

tuned_auc = tuned_summary[tuned_summary['fold'] == 'mean']['roc_auc'].values[0]
tuned_recall = tuned_summary[tuned_summary['fold'] == 'mean']['recall'].values[0]

print(f"\n  ROC-AUC: {baseline_auc:.4f} → {tuned_auc:.4f}  ({tuned_auc - baseline_auc:+.4f})")
print(f"  Recall:   {baseline_recall:.4f} → {tuned_recall:.4f}  ({tuned_recall - baseline_recall:+.4f})")

# ── 保存 ────────────────────────────────────────────────
joblib.dump(rf_best, results_dir / 'rf_best_model.joblib')
joblib.dump(scaler, results_dir / 'rf_best_scaler.joblib')

with open(results_dir / 'rf_best_params.json', 'w') as f:
    json.dump(best_params, f, indent=2)

print(f"\n[OK] 最优模型和参数已保存到 {results_dir}/")
print(f"   最优 ROC-AUC (CV): {tuned_auc:.4f}")
