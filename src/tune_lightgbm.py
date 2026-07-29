"""
LightGBM 超参数调优（Windows 安全版）
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
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

# ═══════════════════════════════════════════════════════════════
# Windows 兼容性：强制使用行优先模式，减少内存冲突
# ═══════════════════════════════════════════════════════════════
if platform.system() == 'Windows':
    os.environ.setdefault('LIGHTGBM_FORCE_ROW_WISE', '1')

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

# LightGBM 使用 encoding='none'（原生 categorical）
X_train, y_train, scaler = preprocess(train_df, encoding='none', fit=True)
X_test, y_test, _ = preprocess(test_df, encoding='none', scaler=scaler, fit=False)

# LightGBM 类别特征（用列索引）
CAT_COL_NAMES = [
    'Contract', 'InternetService', 'MultipleLines', 'PaymentMethod', 'TenureGroup',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies'
]
CAT_FEATURES = [X_train.columns.get_loc(c) for c in CAT_COL_NAMES if c in X_train.columns]
print(f"类别特征索引 ({len(CAT_FEATURES)}): {CAT_FEATURES}")

# ── 基线评估 ────────────────────────────────────────────
def build_lgb(**kwargs):
    defaults = dict(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        num_leaves=31, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1, verbose=-1,
        force_row_wise=True,  # 减少内存冲突
    )
    defaults.update(kwargs)
    return lgb.LGBMClassifier(**defaults)

print("\n>>> 基线模型 CV 评估 ...")
_, baseline_summary = cross_validate(build_lgb, X_train, y_train, n_splits=5,fit_params={'categorical_feature': CAT_FEATURES})
print_cv_summary(baseline_summary, "LightGBM Baseline")
baseline_auc = baseline_summary[baseline_summary['fold'] == 'mean']['roc_auc'].values[0]
baseline_recall = baseline_summary[baseline_summary['fold'] == 'mean']['recall'].values[0]
print(f"\n  基线 ROC-AUC: {baseline_auc:.4f}  |  基线 Recall: {baseline_recall:.4f}")

# ═══════════════════════════════════════════════════════════
# 关键：Windows 下禁用 RandomizedSearchCV 的多进程
# ═══════════════════════════════════════════════════════════
# Windows spawn 模式 + loky + LightGBM C++ = 高概率 access violation
# 策略：搜索过程串行 (n_jobs=1)，让 LightGBM 内部多线程 (n_jobs=-1)
# Mac/Linux 可以继续用多进程搜索
# ═══════════════════════════════════════════════════════════
IS_WINDOWS = platform.system() == 'Windows'
SEARCH_N_JOBS = 1 if IS_WINDOWS else -1
print(f"\n[系统检测] {platform.system()} — RandomizedSearchCV n_jobs={SEARCH_N_JOBS}")

# ═══════════════════════════════════════════════════════════
# Round 1: 粗搜（宽范围）
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  LightGBM Round 1 — 随机搜索（粗搜）")
print("=" * 60)

param_dist_1 = {
    'n_estimators': [100, 200, 300, 400, 500, 600],
    'max_depth': [3, 4, 5, 6, 7, 8, 10, 12],
    'num_leaves': [15, 31, 47, 63, 95, 127],
    'learning_rate': [0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'min_child_samples': [10, 20, 30, 50, 100],
    'reg_alpha': [0, 0.01, 0.1, 0.5, 1.0],
    'reg_lambda': [0, 0.01, 0.1, 0.5, 1.0],
}

search_1 = RandomizedSearchCV(
    lgb.LGBMClassifier(
        class_weight=None,
        random_state=42,
        n_jobs=-1,              # LightGBM 内部多线程（快且稳）
        verbose=-1,
        force_row_wise=True,    # Windows 安全
    ),
    param_distributions=param_dist_1,
    n_iter=50,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='roc_auc',
    n_jobs=SEARCH_N_JOBS,       # ⚠️ Windows 下必须为 1
    random_state=42,
    verbose=1,
    pre_dispatch='2*n_jobs',
)

# categorical_feature 必须通过 fit_params 传递，不能放构造函数！
search_1.fit(X_train, y_train, categorical_feature=CAT_FEATURES)

print(f"\nRound 1 最佳 ROC-AUC: {search_1.best_score_:.4f}")
print(f"Round 1 最佳参数: {json.dumps(search_1.best_params_, indent=2)}")

# ═══════════════════════════════════════════════════════════
# Round 2: 精搜（收窄范围）
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  LightGBM Round 2 — 随机搜索（精搜）")
print("=" * 60)

best_1 = search_1.best_params_

def narrow(val, options, ratio=0.5):
    if isinstance(options, list):
        return options
    idx = options.index(val) if val in options else len(options) // 2
    lo = max(0, int(idx - len(options) * ratio))
    hi = min(len(options), int(idx + len(options) * ratio) + 1)
    return options[lo:hi]

param_dist_2 = {
    'n_estimators': [best_1.get('n_estimators', 200)],
    'max_depth': list(range(
        max(3, best_1['max_depth'] - 2),
        best_1['max_depth'] + 3
    )),
    'num_leaves': list(range(
        max(7, best_1['num_leaves'] - 20),
        best_1['num_leaves'] + 21,
        4
    )),
    'learning_rate': [v for v in [0.01, 0.03, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2]
                      if abs(v - best_1['learning_rate']) <= 0.05],
    'subsample': [v/100 for v in range(
        max(50, int(best_1['subsample'] * 100) - 15),
        min(101, int(best_1['subsample'] * 100) + 16), 5
    )],
    'colsample_bytree': [v/100 for v in range(
        max(50, int(best_1['colsample_bytree'] * 100) - 15),
        min(101, int(best_1['colsample_bytree'] * 100) + 16), 5
    )],
    'min_child_samples': list(range(
        max(5, best_1['min_child_samples'] - 15),
        best_1['min_child_samples'] + 16, 5
    )),
    'reg_alpha': [0] + [v/100 for v in range(
        max(1, int(best_1['reg_alpha'] * 100) - 30),
        min(301, int(best_1['reg_alpha'] * 100) + 31), 10
    )],
    'reg_lambda': [0] + [v/100 for v in range(
        max(1, int(best_1['reg_lambda'] * 100) - 30),
        min(301, int(best_1['reg_lambda'] * 100) + 31), 10
    )],
}

search_2 = RandomizedSearchCV(
    lgb.LGBMClassifier(
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        force_row_wise=True,
    ),
    param_distributions=param_dist_2,
    n_iter=30,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='roc_auc',
    n_jobs=SEARCH_N_JOBS,
    random_state=42,
    verbose=1,
    pre_dispatch='2*n_jobs',
)
search_2.fit(X_train, y_train, categorical_feature=CAT_FEATURES)

print(f"\nRound 2 最佳 ROC-AUC: {search_2.best_score_:.4f}")
print(f"Round 2 最佳参数: {json.dumps(search_2.best_params_, indent=2)}")

# ═══════════════════════════════════════════════════════════
# 最优参数 + 早停精调 n_estimators
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  LightGBM — 早停精调 n_estimators")
print("=" * 60)

best_params = search_2.best_params_.copy()
n_estimators_fixed = best_params.pop('n_estimators', 200)

lgb_final = lgb.LGBMClassifier(
    **best_params,
    n_estimators=1000,
    random_state=42, n_jobs=-1, verbose=-1,
    force_row_wise=True,
)
lgb_final.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric='auc',
    categorical_feature=CAT_FEATURES,
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
)

actual_trees = lgb_final.best_iteration_
print(f"\n早停后最优迭代次数: {actual_trees}")

best_params['n_estimators'] = actual_trees
lgb_best = lgb.LGBMClassifier(
    **best_params,
    random_state=42, n_jobs=-1, verbose=-1,
    force_row_wise=True,
)
lgb_best.fit(X_train, y_train, categorical_feature=CAT_FEATURES)

# ── 调参后 CV 评估 ──────────────────────────────────────
def build_lgb_best():
    return lgb.LGBMClassifier(
        **best_params,
        random_state=42, n_jobs=-1, verbose=-1,
        force_row_wise=True,
    )

print("\n>>> 最优模型 CV 评估 ...")
_, tuned_summary = cross_validate(build_lgb_best, X_train, y_train, n_splits=5,
                                  fit_params={'categorical_feature': CAT_FEATURES})
print_cv_summary(tuned_summary, "LightGBM Tuned")

tuned_auc = tuned_summary[tuned_summary['fold'] == 'mean']['roc_auc'].values[0]
tuned_recall = tuned_summary[tuned_summary['fold'] == 'mean']['recall'].values[0]

print(f"\n  ROC-AUC: {baseline_auc:.4f} → {tuned_auc:.4f}  ({tuned_auc - baseline_auc:+.4f})")
print(f"  Recall:   {baseline_recall:.4f} → {tuned_recall:.4f}  ({tuned_recall - baseline_recall:+.4f})")

# ── 保存 ────────────────────────────────────────────────
joblib.dump(lgb_best, results_dir / 'lgb_best_model.joblib')
joblib.dump(scaler, results_dir / 'lgb_best_scaler.joblib')

with open(results_dir / 'lgb_best_params.json', 'w') as f:
    json.dump(best_params, f, indent=2)

print(f"\n[OK] 最优模型和参数已保存到 {results_dir}/")
print(f"   最优 ROC-AUC (CV): {tuned_auc:.4f}")