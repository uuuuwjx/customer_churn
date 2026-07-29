"""
XGBoost 超参数调优（Windows 安全版）
"""
import pandas as pd
import numpy as np
import os
import sys
import joblib
import json
import warnings
import platform

warnings.filterwarnings('ignore')

from pathlib import Path
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
import xgboost as xgb

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

# XGBoost 使用 ordinal 编码（XGBoost 不支持原生类别特征，必须用数字编码）
X_train, y_train, scaler = preprocess(train_df, encoding='ordinal', fit=True)
X_test, y_test, _ = preprocess(test_df, encoding='ordinal', scaler=scaler, fit=False)

# 计算类别不平衡比例，让模型更重视"流失"这个少数类
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"scale_pos_weight: {scale_pos_weight:.2f}")

# Windows 安全设置（防止和 LightGBM 一样崩溃）
IS_WINDOWS = platform.system() == 'Windows'
SEARCH_N_JOBS = 1 if IS_WINDOWS else -1
print(f"[系统检测] {platform.system()} — RandomizedSearchCV n_jobs={SEARCH_N_JOBS}")

# ── 基线评估 ────────────────────────────────────────────
def build_xgb(**kwargs):
    defaults = dict(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        eval_metric='auc',              # 和搜索目标一致
        scale_pos_weight=scale_pos_weight,  # 修复：基线也要处理类别不平衡
        random_state=42, n_jobs=-1
    )
    defaults.update(kwargs)
    return xgb.XGBClassifier(**defaults)

print("\n>>> 基线模型 CV 评估 ...")
_, baseline_summary = cross_validate(build_xgb, X_train, y_train, n_splits=5)
print_cv_summary(baseline_summary, "XGBoost Baseline")
baseline_auc = baseline_summary[baseline_summary['fold'] == 'mean']['roc_auc'].values[0]
baseline_recall = baseline_summary[baseline_summary['fold'] == 'mean']['recall'].values[0]
print(f"\n  基线 ROC-AUC: {baseline_auc:.4f}  |  基线 Recall: {baseline_recall:.4f}")

# ═══════════════════════════════════════════════════════════
# Round 1: 粗搜
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  XGBoost Round 1 — 随机搜索（粗搜）")
print("=" * 60)

param_dist_1 = {
    'n_estimators': [100, 200, 300, 400, 500, 600],
    'max_depth': [3, 4, 5, 6, 7, 8, 10],
    'learning_rate': [0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'min_child_weight': [1, 3, 5, 7, 10],
    'gamma': [0, 0.05, 0.1, 0.2, 0.3, 0.5],
    'reg_alpha': [0, 0.01, 0.1, 0.5, 1.0],
    'reg_lambda': [0, 0.01, 0.1, 0.5, 1.0],
}

search_1 = RandomizedSearchCV(
    xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric='auc',
        random_state=42, n_jobs=-1
    ),
    param_distributions=param_dist_1,
    n_iter=40,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='roc_auc',
    n_jobs=SEARCH_N_JOBS,       # 修复：Windows 下用 1，其他系统用 -1
    random_state=42,
    verbose=1,
)
search_1.fit(X_train, y_train)

print(f"\nRound 1 最佳 ROC-AUC: {search_1.best_score_:.4f}")
print(f"Round 1 最佳参数: {json.dumps(search_1.best_params_, indent=2)}")

# ═══════════════════════════════════════════════════════════
# Round 2: 精搜
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  XGBoost Round 2 — 随机搜索（精搜）")
print("=" * 60)

best_1 = search_1.best_params_

param_dist_2 = {
    'n_estimators': [best_1.get('n_estimators', 300)],
    'max_depth': list(range(
        max(3, best_1['max_depth'] - 2),
        best_1['max_depth'] + 3
    )),
    'learning_rate': sorted(set(
        [v for v in [0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.12, 0.15]
         if abs(v - best_1['learning_rate']) <= 0.08]
    )),
    'subsample': [v/100 for v in range(
        max(50, int(best_1['subsample'] * 100) - 10),
        min(101, int(best_1['subsample'] * 100) + 11), 5
    )],
    'colsample_bytree': [v/100 for v in range(
        max(50, int(best_1['colsample_bytree'] * 100) - 10),
        min(101, int(best_1['colsample_bytree'] * 100) + 11), 5
    )],
    'min_child_weight': list(range(
        max(1, best_1['min_child_weight'] - 2),
        best_1['min_child_weight'] + 3
    )),
    'gamma': [v/100 for v in range(
        max(0, int(best_1['gamma'] * 100) - 10),
        int(best_1['gamma'] * 100) + 11, 5
    )],
    'reg_alpha': [v/100 for v in range(
        max(0, int(best_1['reg_alpha'] * 100) - 20),
        min(201, int(best_1['reg_alpha'] * 100) + 21), 10
    )],
    'reg_lambda': [v/100 for v in range(
        max(0, int(best_1['reg_lambda'] * 100) - 20),
        min(201, int(best_1['reg_lambda'] * 100) + 21), 10
    )],
}

search_2 = RandomizedSearchCV(
    xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric='auc',
        random_state=42, n_jobs=-1
    ),
    param_distributions=param_dist_2,
    n_iter=25,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='roc_auc',
    n_jobs=SEARCH_N_JOBS,       # 修复：Windows 下用 1
    random_state=42,
    verbose=1,
)
search_2.fit(X_train, y_train)

print(f"\nRound 2 最佳 ROC-AUC: {search_2.best_score_:.4f}")
print(f"Round 2 最佳参数: {json.dumps(search_2.best_params_, indent=2)}")

# ═══════════════════════════════════════════════════════════
# 使用 Round 2 最优参数训练最终模型（当前环境不支持早停）
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  XGBoost — 使用 Round 2 最优参数训练最终模型")
print("=" * 60)

best_params = search_2.best_params_.copy()

# 直接用 Round 2 搜出的 n_estimators 训练（通常已经足够好）
xgb_best = xgb.XGBClassifier(
    **best_params,
    scale_pos_weight=scale_pos_weight,
    eval_metric='auc',
    random_state=42, n_jobs=-1
)
xgb_best.fit(X_train, y_train)

print(f"\n最终模型 n_estimators: {best_params.get('n_estimators', 300)}")
# ── 调参后 CV 评估 ──────────────────────────────────────
def build_xgb_best():
    return xgb.XGBClassifier(
        **best_params,
        scale_pos_weight=scale_pos_weight,
        eval_metric='auc',
        random_state=42, n_jobs=-1
    )

print("\n>>> 最优模型 CV 评估 ...")
_, tuned_summary = cross_validate(build_xgb_best, X_train, y_train, n_splits=5)
print_cv_summary(tuned_summary, "XGBoost Tuned")

tuned_auc = tuned_summary[tuned_summary['fold'] == 'mean']['roc_auc'].values[0]
tuned_recall = tuned_summary[tuned_summary['fold'] == 'mean']['recall'].values[0]

print(f"\n  ROC-AUC: {baseline_auc:.4f} → {tuned_auc:.4f}  ({tuned_auc - baseline_auc:+.4f})")
print(f"  Recall:   {baseline_recall:.4f} → {tuned_recall:.4f}  ({tuned_recall - baseline_recall:+.4f})")

# ── 保存 ────────────────────────────────────────────────
joblib.dump(xgb_best, results_dir / 'xgb_best_model.joblib')
joblib.dump(scaler, results_dir / 'xgb_best_scaler.joblib')

with open(results_dir / 'xgb_best_params.json', 'w') as f:
    json.dump(best_params, f, indent=2)

print(f"\n[OK] 最优模型和参数已保存到 {results_dir}/")
print(f"   最优 ROC-AUC (CV): {tuned_auc:.4f}")