import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess(df, encoding='onehot', scaler=None, fit=True):
    """
    特征预处理：根据模型类型选择编码策略
    
    Parameters
    ----------
    df : DataFrame,processed/telco_churn_featured
        包含原始特征 + 'Churn' 的数据
    encoding : str
        'onehot'  -> 线性模型、神经网络
        'ordinal' -> 树模型（RF/XGBoost）
        'none'    -> LightGBM / CatBoost（原生支持类别特征）
    scaler : StandardScaler or None
        训练时传 None，测试时传入训练阶段 fit 好的 scaler
    fit : bool
        True 表示训练阶段，False 表示推理阶段
    
    Returns
    -------
    X : DataFrame
        处理后的特征矩阵
    y : Series or None
        目标变量（如果输入包含 Churn）
    scaler : StandardScaler
        拟合好的 scaler，用于后续推理
    """
    df = df.copy()

    # ---- 多重共线性处理 ----
    drop_cols = ['TotalCharges', 'AvgMonthlyCharge']
    if encoding == 'onehot':
        drop_cols.append('tenure')  # 线性模型用 TenureGroup 替代
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # ---- 分离目标变量 ----
    if 'Churn' in df.columns:
        y = df['Churn'].copy()
        X = df.drop(columns=['Churn'])
    else:
        y = None
        X = df.copy()
    
    # =====================
    # 1. 二值变量统一映射
    # =====================
    binary_cols = [
        'gender', 'Partner', 'Dependents', 'PhoneService',
        'PaperlessBilling'
    ]
    binary_map = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0}
    
    for col in binary_cols:
        if col in X.columns:
            X[col] = X[col].map(binary_map)
    
    # =====================
    # 2. 多类别变量编码
    # =====================
    multicol_cols = ['Contract', 'InternetService', 'MultipleLines', 
                     'PaymentMethod', 'TenureGroup','OnlineSecurity','OnlineBackup',
                     'DeviceProtection','TechSupport','StreamingTV', 'StreamingMovies']
    
    if encoding == 'onehot':
        X = pd.get_dummies(X, columns=multicol_cols, drop_first=False)
        
    elif encoding == 'ordinal':
        ordinal_maps = {
            'Contract': {'Month-to-month': 0, 'One year': 1, 'Two year': 2},
            'InternetService': {'No': 0, 'DSL': 1, 'Fiber optic': 2},
            'MultipleLines': {'No phone service': 0, 'No': 1, 'Yes': 2},
            'PaymentMethod': {
                'Electronic check': 0,
                'Mailed check': 1,
                'Bank transfer (automatic)': 2,
                'Credit card (automatic)': 3
            },
            'TenureGroup': {'0-12': 0, '13-24': 1, '25-48': 2, '49-72': 3},
            'OnlineSecurity':{'No internet service':0,'No':1,'Yes':2},
            'OnlineBackup':{'No internet service':0,'No':1,'Yes':2},
            'DeviceProtection':{'No internet service':0,'No':1,'Yes':2},
            'TechSupport':{'No internet service':0,'No':1,'Yes':2},
            'StreamingTV':{'No internet service':0,'No':1,'Yes':2},
            'StreamingMovies': {'No internet service': 0, 'No': 1, 'Yes': 2}
        }
        for col, mapping in ordinal_maps.items():
            if col in X.columns:
                X[col] = X[col].map(mapping)
                
    elif encoding == 'none':
        # LightGBM/CatBoost 原生类别编码
        # 使用与 ordinal 相同的整数映射，避免 category dtype 的 Windows 兼容问题
        # 配合 categorical_feature 参数，LightGBM 会将其视为类别特征
        ordinal_maps = {
            'Contract': {'Month-to-month': 0, 'One year': 1, 'Two year': 2},
            'InternetService': {'No': 0, 'DSL': 1, 'Fiber optic': 2},
            'MultipleLines': {'No phone service': 0, 'No': 1, 'Yes': 2},
            'PaymentMethod': {
                'Electronic check': 0,
                'Mailed check': 1,
                'Bank transfer (automatic)': 2,
                'Credit card (automatic)': 3
            },
            'TenureGroup': {'0-12': 0, '13-24': 1, '25-48': 2, '49-72': 3},
            'OnlineSecurity': {'No internet service': 0, 'No': 1, 'Yes': 2},
            'OnlineBackup': {'No internet service': 0, 'No': 1, 'Yes': 2},
            'DeviceProtection': {'No internet service': 0, 'No': 1, 'Yes': 2},
            'TechSupport': {'No internet service': 0, 'No': 1, 'Yes': 2},
            'StreamingTV': {'No internet service': 0, 'No': 1, 'Yes': 2},
            'StreamingMovies': {'No internet service': 0, 'No': 1, 'Yes': 2}
        }
        for col, mapping in ordinal_maps.items():
            if col in X.columns:
                X[col] = X[col].map(mapping)
    else:
        raise ValueError("encoding must be 'onehot', 'ordinal' or 'none'")
    
    # =====================
    # 3. 数值标准化
    # =====================
    num_cols = [
        'tenure', 'MonthlyCharges',
        'ChargeDiff', 'TotalServices'
    ]
    num_cols = [c for c in num_cols if c in X.columns]
    
    if num_cols:
        if fit:
            scaler = StandardScaler()
            X[num_cols] = scaler.fit_transform(X[num_cols])
        else:
            if scaler is None:
                raise ValueError("推理阶段必须传入训练好的 scaler")
            X[num_cols] = scaler.transform(X[num_cols])
    
    return X, y, scaler
