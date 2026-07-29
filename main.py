"""
================================================================================
  Telco Customer Churn Prediction System
  客户流失预测系统 — 主入口
================================================================================

Usage:
  python main.py                          # 交互式菜单
  python main.py predict --model rf       # 批量预测（默认数据）
  python main.py predict --input new.csv --model lgb   # 对新数据预测
  python main.py importance               # 特征重要性分析
  python main.py report                   # 查看模型对比结果
================================================================================
"""
import sys
import os
import argparse

# 确保 src 目录在 path 中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def menu():
    """交互式菜单"""
    while True:
        print("\n" + "=" * 50)
        print("  客户流失预测系统")
        print("=" * 50)
        print("  1. 查看模型对比结果")
        print("  2. 特征重要性分析")
        print("  3. 单条预测示例")
        print("  4. 批量预测")
        print("  5. 退出")
        print("=" * 50)

        choice = input("请输入选项 [1-5]: ").strip()

        if choice == '1':
            show_report()
        elif choice == '2':
            run_importance()
        elif choice == '3':
            run_single_demo()
        elif choice == '4':
            run_batch_predict()
        elif choice == '5':
            print("再见!")
            break
        else:
            print("无效选项，请重新输入。")


def show_report():
    """显示最终模型对比结果"""
    from pathlib import Path
    import pandas as pd

    results_dir = Path(__file__).resolve().parent / "results"
    csv_path = results_dir / "final_model_comparison.csv"

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        cols = ['Model', 'Threshold', 'Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC', 'KS', 'Business Gain']
        print_df = df[cols].copy()
        for col in ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC', 'KS']:
            print_df[col] = print_df[col].apply(lambda x: f"{x:.4f}")
        print_df['Business Gain'] = print_df['Business Gain'].apply(lambda x: f"${x:,.0f}")
        print("\n" + "=" * 90)
        print("  最终模型对比结果（最优阈值）")
        print("=" * 90)
        print(print_df.to_string(index=False))
    else:
        print("\n[WARN] 未找到 final_model_comparison.csv，请先运行 train_final.py")


def run_importance():
    """运行特征重要性分析"""
    print("\n正在运行特征重要性分析...")
    import subprocess
    subprocess.run([sys.executable, 'src/feature_importance.py'])


def run_single_demo():
    """单条预测演示"""
    from predict import predict_single
    from pathlib import Path
    import pandas as pd

    data_path = Path(__file__).resolve().parent / "data" / "processed" / "telco_churn_featured.parquet"
    df = pd.read_parquet(data_path)
    sample = df.drop(columns=['Churn']).iloc[0].to_dict()

    print("\n示例客户特征:")
    for k, v in sample.items():
        print(f"  {k}: {v}")

    model_key = input("\n选择模型 [lr/rf/xgb/lgb] (默认 rf): ").strip() or 'rf'

    result = predict_single(sample, model_key)
    print(f"\n预测结果:")
    print(f"  模型:       {result['model']}")
    print(f"  流失概率:   {result['churn_probability']:.2%}")
    print(f"  预测结果:   {result['prediction_label']}")
    print(f"  使用阈值:   {result['threshold']:.2f}")


def run_batch_predict():
    """批量预测"""
    from predict import predict_batch
    from pathlib import Path

    input_path = input("\n输入文件路径 (默认: 使用内置测试数据): ").strip()
    model_key = input("选择模型 [lr/rf/xgb/lgb] (默认 rf): ").strip() or 'rf'

    if not input_path:
        input_path = Path(__file__).resolve().parent / "data" / "processed" / "telco_churn_featured.parquet"

    predict_batch(input_path, model_key=model_key)


# ═══════════════════════════════════════════════════════════
# 命令行参数入口
# ═══════════════════════════════════════════════════════════
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Telco Customer Churn Prediction System'
    )
    parser.add_argument('command', nargs='?', default='menu',
                        choices=['menu', 'predict', 'importance', 'report'],
                        help='操作: menu | predict | importance | report')
    parser.add_argument('--model', default='rf', choices=['lr', 'rf', 'xgb', 'lgb'],
                        help='模型选择 (default: rf)')
    parser.add_argument('--input', type=str, help='输入文件路径')
    parser.add_argument('--output', type=str, help='输出文件路径')

    args = parser.parse_args()

    if args.command == 'menu':
        menu()
    elif args.command == 'report':
        show_report()
    elif args.command == 'importance':
        run_importance()
    elif args.command == 'predict':
        from predict import predict_batch
        from pathlib import Path

        if args.input is None:
            args.input = Path(__file__).resolve().parent / "data" / "processed" / "telco_churn_featured.parquet"
            print(f"未指定 --input，使用默认数据")

        predict_batch(args.input, args.output, args.model)
