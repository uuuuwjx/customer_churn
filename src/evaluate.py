"""
评估框架：交叉验证、业务收益计算、阈值优化
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from scipy import stats


def ks_stat(y_true, y_proba):
    """KS 统计量"""
    pos = y_proba[y_true == 1]
    neg = y_proba[y_true == 0]
    return stats.ks_2samp(pos, neg).statistic


def business_gain(y_true, y_proba, threshold=0.5, gain_save=500, cost_intervene=100):
    """
    业务收益计算

    假设：
    - 成功挽留一个即将流失的客户，收益 gain_save 元（默认 500）
    - 错误干预一个不会流失的客户，成本 cost_intervene 元（默认 100）

    Returns:
        total_gain: 总收益
        confusion: {'TP':, 'FP':, 'TN':, 'FN':}
    """
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total_gain = tp * gain_save - fp * cost_intervene
    return total_gain, {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn}


def evaluate_fold(model, X_train, y_train, X_val, y_val, threshold=0.5,fit_params=None):
    """单折评估，返回全部指标"""
    fit_params = fit_params or {}   # 保险：如果是 None，变成空字典
    model.fit(X_train, y_train,**fit_params)
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    gain, _ = business_gain(y_val, y_proba, threshold)

    return {
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred),
        'recall': recall_score(y_val, y_pred),
        'f1': f1_score(y_val, y_pred),
        'roc_auc': roc_auc_score(y_val, y_proba),
        'ks': ks_stat(y_val, y_proba),
        'business_gain': gain,
    }


def cross_validate(model_builder, X, y, n_splits=5, threshold=0.5,fit_params=None):
    """
    分层 K 折交叉验证

    Parameters:
        model_builder: callable, 每次调用返回一个未训练的新模型实例（避免数据泄漏）
        X, y: 特征和标签
        n_splits: 折数
        threshold: 分类阈值

    Returns:
        metrics_df: 每折的指标明细
        summary_df: 均值和标准差汇总
    """
    fit_params = fit_params or {} 
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    records = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train_fold = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx]
        X_val_fold = X.iloc[val_idx] if hasattr(X, 'iloc') else X[val_idx]
        y_train_fold = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
        y_val_fold = y.iloc[val_idx] if hasattr(y, 'iloc') else y[val_idx]

        model = model_builder()
        metrics = evaluate_fold(model, X_train_fold, y_train_fold, X_val_fold, y_val_fold, threshold,fit_params)
        metrics['fold'] = fold
        records.append(metrics)

    metrics_df = pd.DataFrame(records)

    # 汇总
    mean_row = {col: metrics_df[col].mean() for col in metrics_df.columns if col != 'fold'}
    std_row = {col: metrics_df[col].std() for col in metrics_df.columns if col != 'fold'}

    summary_df = pd.DataFrame([
        {**mean_row, 'fold': 'mean'},
        {**std_row, 'fold': 'std'},
    ])

    return metrics_df, summary_df


def find_optimal_threshold(model, X_val, y_val, gain_save=500, cost_intervene=100):
    """
    搜索最优分类阈值，最大化业务收益

    Returns:
        best_threshold: 最优阈值
        best_gain: 最大业务收益
        results_df: 各阈值下的指标明细
    """
    y_proba = model.predict_proba(X_val)[:, 1]

    records = []
    thresholds = np.arange(0.1, 0.91, 0.02)

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        gain, conf = business_gain(y_val, y_proba, t, gain_save, cost_intervene)
        records.append({
            'threshold': round(t, 2),
            'business_gain': gain,
            'recall': recall_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred),
            'f1': f1_score(y_val, y_pred),
            **conf
        })

    results_df = pd.DataFrame(records)
    best_idx = results_df['business_gain'].idxmax()
    best = results_df.iloc[best_idx]

    return best['threshold'], best['business_gain'], results_df


def print_cv_summary(summary_df, model_name, n_splits=5):
    """格式化打印 CV 结果"""
    mean = summary_df[summary_df['fold'] == 'mean'].iloc[0]
    std = summary_df[summary_df['fold'] == 'std'].iloc[0]

    print(f"\n{'='*60}")
    print(f"  {model_name} -- {n_splits}-Fold CV Results")
    print(f"{'='*60}")
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'ks', 'business_gain']:
        print(f"  {metric:<18}: {mean[metric]:.4f}  (+/-{std[metric]:.4f})")
