"""
预测脚本：支持单个预测和批量预测，可选择任意模型
"""
import pandas as pd
import os
import sys
import joblib
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import preprocess

# ── 模型编码映射 ────────────────────────────────────────
MODEL_CONFIG = {
    'lr':  {'name': 'Logistic Regression', 'encoding': 'onehot',  'model_file': 'lr_best_model.joblib',  'scaler_file': 'lr_best_scaler.joblib'},
    'rf':  {'name': 'Random Forest',       'encoding': 'ordinal', 'model_file': 'rf_best_model.joblib',  'scaler_file': 'rf_best_scaler.joblib'},
    'xgb': {'name': 'XGBoost',             'encoding': 'ordinal', 'model_file': 'xgb_best_model.joblib', 'scaler_file': 'xgb_best_scaler.joblib'},
    'lgb': {'name': 'LightGBM',            'encoding': 'none',    'model_file': 'lgb_best_model.joblib', 'scaler_file': 'lgb_best_scaler.joblib'},
}

DEFAULT_MODEL = 'rf'


def load_model(model_key='rf'):
    """加载指定模型和 scaler"""
    results_dir = Path(__file__).resolve().parent.parent / "results"
    config = MODEL_CONFIG.get(model_key, MODEL_CONFIG[DEFAULT_MODEL])

    model = joblib.load(results_dir / config['model_file'])
    scaler = joblib.load(results_dir / config['scaler_file'])

    # 尝试加载阈值
    threshold = 0.5
    comparison_file = results_dir / 'final_model_comparison.csv'
    if comparison_file.exists():
        df_comp = pd.read_csv(comparison_file)
        match = df_comp[df_comp['Model'] == config['name']]
        if not match.empty:
            threshold = float(match['Threshold'].values[0])

    return model, scaler, config['encoding'], threshold


def predict_single(customer_dict, model_key='rf'):
    """
    单条预测

    Parameters:
        customer_dict: dict, 客户特征（需包含所有特征列）
        model_key: 'lr' | 'rf' | 'xgb' | 'lgb'

    Returns:
        dict: {'churn_probability': float, 'prediction': int, 'model': str}
    """
    model, scaler, encoding, threshold = load_model(model_key)

    df = pd.DataFrame([customer_dict])
    X, _, _ = preprocess(df, encoding=encoding, scaler=scaler, fit=False)

    proba = model.predict_proba(X)[:, 1][0]
    pred = int(proba >= threshold)

    return {
        'churn_probability': round(proba, 4),
        'prediction': pred,
        'prediction_label': 'Churn' if pred == 1 else 'No Churn',
        'threshold': threshold,
        'model': MODEL_CONFIG[model_key]['name']
    }


def predict_batch(input_path, output_path=None, model_key='rf', threshold=None):
    """
    批量预测：读取 CSV/Parquet，输出带预测结果的 CSV

    Parameters:
        input_path: str, 输入文件路径
        output_path: str or None, 输出文件路径（None 则自动生成）
        model_key: 'lr' | 'rf' | 'xgb' | 'lgb'
        threshold: float or None, 自定义阈值（None 则使用训练时最优阈值）

    Returns:
        pd.DataFrame: 包含原始数据 + churn_probability + prediction 的结果
    """
    model, scaler, encoding, best_threshold = load_model(model_key)
    threshold = threshold if threshold is not None else best_threshold

    # 读取数据
    if str(input_path).endswith('.parquet'):
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)

    X, _, _ = preprocess(df, encoding=encoding, scaler=scaler, fit=False)
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)

    result = df.copy()
    result['churn_probability'] = proba.round(4)
    result['prediction'] = pred
    result['prediction_label'] = pred.map({1: 'Churn', 0: 'No Churn'})

    # 输出统计
    n_churn = pred.sum()
    n_total = len(pred)
    print(f"\n{'='*50}")
    print(f"  预测结果摘要 ({MODEL_CONFIG[model_key]['name']})")
    print(f"{'='*50}")
    print(f"  总样本数:      {n_total}")
    print(f"  预测流失:      {n_churn} ({n_churn/n_total*100:.1f}%)")
    print(f"  预测未流失:    {n_total - n_churn} ({(n_total-n_churn)/n_total*100:.1f}%)")
    print(f"  使用阈值:      {threshold:.2f}")

    # 保存
    if output_path is None:
        input_name = Path(input_path).stem
        output_path = Path(input_path).parent / f"{input_name}_predictions.csv"
    result.to_csv(output_path, index=False)
    print(f"\n[OK] 预测结果已保存到 {output_path}")

    return result


# ═══════════════════════════════════════════════════════════
# 命令行入口
# ═══════════════════════════════════════════════════════════
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Customer Churn Prediction')
    parser.add_argument('--mode', choices=['single', 'batch'], default='batch',
                        help='single: 单条示例; batch: 批量预测')
    parser.add_argument('--model', choices=['lr', 'rf', 'xgb', 'lgb'], default=DEFAULT_MODEL,
                        help='模型选择 (default: rf)')
    parser.add_argument('--input', type=str, help='输入 CSV 或 Parquet 文件路径')
    parser.add_argument('--output', type=str, default=None, help='输出文件路径')
    parser.add_argument('--threshold', type=float, default=None, help='自定义阈值')

    args = parser.parse_args()

    if args.mode == 'batch':
        if args.input is None:
            # 默认：对训练数据做预测
            data_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "telco_churn_featured.parquet"
            print(f"未指定 --input，使用默认数据: {data_path}")
            args.input = data_path

        predict_batch(args.input, args.output, args.model, args.threshold)

    elif args.mode == 'single':
        # 示例：用数据集中第一条做演示
        data_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "telco_churn_featured.parquet"
        df = pd.read_parquet(data_path)
        sample = df.drop(columns=['Churn']).iloc[0].to_dict()

        print("示例客户特征:")
        for k, v in sample.items():
            print(f"  {k}: {v}")

        result = predict_single(sample, args.model)
        print(f"\n预测结果:")
        print(f"  流失概率: {result['churn_probability']}")
        print(f"  预测结果: {result['prediction_label']}")
        print(f"  使用模型: {result['model']}")
        print(f"  使用阈值: {result['threshold']}")
