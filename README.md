# Telco Customer Churn Prediction

## 项目概述

基于 IBM Watson Analytics 的电信客户流失数据集，构建机器学习模型预测客户流失风险，并制定数据驱动的客户挽留策略。

**核心结论**：

- 最佳模型：**Logistic Regression**（业务收益 $129,800，可解释性最强）
- 最强召回：**Random Forest**（Recall 92.25%）
- 最高 AUC：**LightGBM**（ROC-AUC 0.8455）
- Top 5 流失驱动因素：在网时长、合同类型、月费、互联网服务类型、在线安全

## 数据背景说明

本数据集来源于 **IBM Watson Analytics** 社区公开的经典案例，模拟了一家为美国加州地区提供**家庭电话和互联网服务**的电信公司。

* **公司痛点**：电信市场竞争激烈，**争取一位新客户的成本是留住一位老客户的 5 倍以上**。近期该公司的月度用户流失率持续攀升，严重影响了营收和客户生命周期价值（LTV）。
* **项目目标**：通过分析客户的人口属性、订阅服务和消费行为，**建立预测模型识别高流失风险客户**，从而支持运营团队提前进行干预（如发送优惠券、提供专属客服）。

在本数据集中，**`Churn`（流失）** 具有明确的时间界定：

* 数据采集周期覆盖了某一连续时间段（例如 2020 年 Q1 至 Q3）。
* 若客户在**该时间段的最后一个月（观测窗口末期）** 主动取消了合同、终止了所有付费服务，则标记为 `Churn = Yes`；若在末期仍保持活跃付费状态，则标记为 `Churn = No`。

原始数据共包含 **7,043 条客户记录**，涵盖了 **21 个字段。**


| 字段名             | 中文翻译                  | 业务含义                                                                                                                                                                   |
| -------------------- | --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `customerID`       | 客户唯一标识号            | 每个客户的唯一 ID，用于区分不同个体，**无预测能力**，建模时需剔除。                                                                                                        |
| `gender`           | 性别                      | 客户性别，取值`Male`（男）或`Female`（女）。                                                                                                                               |
| `SeniorCitizen`    | 是否老年人                | 取值`1` 表示年龄 >= 65 岁（老年人），`0` 表示非老年人。                                                                                                                    |
| `Partner`          | 是否有伴侣                | 客户是否有伴侣，取值`Yes`（有）或`No`（无）。                                                                                                                              |
| `Dependents`       | 是否有家属（子女/老人等） | 客户是否有家属（如子女、父母等）需要照顾，取值`Yes`或`No`。                                                                                                                |
| `tenure`           | 在网时长（月数）          | 客户在该电信公司连续订阅服务的月数。范围 0~72 个月。                                                                                                                       |
| `PhoneService`     | 是否订阅电话服务          | 客户是否订购了家庭电话服务，取值`Yes`或`No`。                                                                                                                              |
| `MultipleLines`    | 是否有多条电话线          | 是否有多条电话线，取值`Yes`、`No`或`No phone service`（无电话服务）。                                                                                                      |
| `InternetService`  | 互联网服务类型            | 客户订购的互联网服务类型，取值`DSL`（数字用户线路）、`Fiber optic`（光纤）或`No`（无互联网）。                                                                             |
| `OnlineSecurity`   | 是否订阅在线安全服务      | 是否订阅了在线安全/杀毒服务，取值`Yes`、`No`或`No internet service`（无互联网服务）。                                                                                      |
| `OnlineBackup`     | 是否订阅在线备份服务      | 是否订阅了在线数据备份服务，取值同上。                                                                                                                                     |
| `DeviceProtection` | 是否订阅设备保护服务      | 是否订阅了设备（如路由器/电脑）的保修/保护服务，取值同上。                                                                                                                 |
| `TechSupport`      | 是否订阅技术支持服务      | 是否订阅了专业技术支持服务，取值同上。                                                                                                                                     |
| `StreamingTV`      | 是否订阅流媒体电视服务    | 是否通过该电信公司订阅了流媒体电视（如 IPTV），取值同上。                                                                                                                  |
| `StreamingMovies`  | 是否订阅流媒体电影服务    | 是否通过该电信公司订阅了流媒体电影（如 Netflix 等合作），取值同上。                                                                                                        |
| `Contract`         | 合同类型                  | 客户与公司签订的合同期限，取值`Month-to-month`（按月签约）、`One year`（一年期）或`Two year`（两年期）。                                                                   |
| `PaperlessBilling` | 是否使用电子账单          | 客户是否采用电子账单（无纸化），取值`Yes`或`No`。                                                                                                                          |
| `PaymentMethod`    | 支付方式                  | 客户的付费方式，取值`Electronic check`（电子支票）、`Mailed check`（邮寄支票）、`Bank transfer (automatic)`（银行自动转账）或`Credit card (automatic)`（信用卡自动扣款）。 |
| `MonthlyCharges`   | 月均费用（美元）          | 客户每个月需要支付的费用总额（基于当前套餐）。范围约 18~120 美元。                                                                                                         |
| `TotalCharges`     | 历史总费用（美元）        | 客户从入网至今累计支付给该公司的总费用。                                                                                                                                   |
| `Churn`            | 是否流失（目标变量）      | 取值`Yes`表示客户在观测期末已离网（流失），`No`表示客户仍为活跃用户。**本项目的预测目标。                                                                                  |

---

## 项目结构

```
customer_churn(re)/
├── main.py                          # 主入口（交互式菜单 / 命令行）
├── README.md                        # 项目文档
├── requirements.txt                 # 依赖列表
├── .gitignore
├── data/
│   ├── raw/
│   │   └── telco_customer_churn.csv     # 原始数据
│   └── processed/
│       ├── telco_churn_cleaned.parquet  # 清洗后数据
│       └── telco_churn_featured.parquet # 特征工程后数据（建模输入）
├── notebooks/
│   ├── 01_EDA.ipynb                     # 探索性数据分析
|   └── 02_feature_engineering.ipynb     # 特征工程（创建新特征）
├── src/
│   ├── preprocess.py                   # 数据预处理（编码 + 标准化）
│   ├── evaluate.py                     # 评估框架（CV + 业务收益 + 阈值优化）
│   ├── train_final.py                  # 最终训练 & 评估（加载最优参数）
│   ├── tune_lightgbm.py                # LightGBM 调参
│   ├── tune_xgboost.py                 # XGBoost 调参
│   ├── tune_rf.py                      # Random Forest 调参
│   ├── tune_lr.py                      # Logistic Regression 调参
│   ├── feature_importance.py           # 特征重要性分析
│   └── predict.py                      # 预测脚本（单条 / 批量）
├── results/                            # 最优模型 + 评估结果
│   ├── *_best_model.joblib             # 各模型最优参数版本
│   ├── *_best_params.json              # 各模型最优参数
│   ├── *_best_scaler.joblib            # 各模型标准化器
│   ├── final_model_comparison.csv      # 最终模型对比表
│   └── feature_importance.csv          # 特征重要性表
└── reports/                            # 可视化图表
    ├── model_comparison_final.png      # 最终模型对比
    ├── feature_importance.png          # 特征重要性
    ├── confusion_matrics.png           # 混淆矩阵图
    └── roc_curves.png                  # roc曲线图
```

---

## 环境配置

```bash
# 1. 创建虚拟环境（推荐）
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Mac/Linux

# 2. 安装依赖
pip install -r requirements.txt
```

**依赖包**：`pandas`, `numpy`, `scikit-learn`, `scipy`, `xgboost`, `lightgbm`, `joblib`, `matplotlib`, `seaborn`, `pyarrow`

---

## 流程

```
原始 CSV (7043 x 21)
    │
    ├── EDA & 清洗 (notebooks/01_EDA.ipynb)
    │   - TotalCharges 类型转换
    │   - 空白值填充 (tenure=0)
    │   - 去除空格
    │
    ├── 特征工程 (notebooks/02_feature_engineering.ipynb)
    │   - Churn: Yes/No → 1/0
    │   - AvgMonthlyCharge = TotalCharges / tenure
    │   - ChargeDiff = MonthlyCharges - AvgMonthlyCharge
    │   - TotalServices = 统计开通服务数
    │   - TenureGroup = 在网时长分组 (0-12/13-24/25-48/49-72)
    │
    ├── 预处理 (src/preprocess.py)
    │   - 二值变量映射 (Yes/No → 1/0)
    │   - 类别编码 (OneHot / Ordinal / None)
    │   - 数值标准化 (StandardScaler)
    │
    ├── 训练 + 调参 (src/tune_*.py)
    │   - 5-Fold CV 基线评估
    │   - RandomizedSearchCV/GridSearchCV 调参
    │   - 早停确定最优 n_estimators
    │
    └── 最终评估 (src/train_final.py)
        - 阈值优化（最大化业务收益）
        - 集成尝试
        - 可视化输出
```

---

## 快速开始

```bash
# 方式 1：交互式菜单
python main.py

# 方式 2：命令行
python main.py report                       # 查看模型对比结果
python main.py importance                   # 特征重要性分析
python main.py predict --model rf           # 用 RF 模型批量预测
python main.py predict --input new.csv --model lgb  # 对新数据预测

# 方式 3：直接用 predict.py
python src/predict.py --mode batch --model rf
python src/predict.py --mode single --model lgb

# 方式 4：从头训练
python src/train_final.py                   # 训练 + 评估（用最优参数）
```
### 注意
| 数据文件                      | 获取方式                                        |
| ------------------------- | ------------------------------------------- |
| `data/customer_churn.csv` | 从 [Kaggle Telco Churn](链接) 下载，放入 `data/` 目录 |
| `data/processed/`         | 运行 `02_feature_engineering.ipynb` 自动生成              |
| `results/`                | 运行 `src/` 目录下文件生成                            |

---

## 模型对比结果


| 模型                | 阈值 | Accuracy | Precision | Recall     | F1         | ROC-AUC    | KS         | 业务收益     |
| --------------------- | ------ | ---------- | ----------- | ------------ | ------------ | ------------ | ------------ | -------------- |
| Logistic Regression | 0.16 | 0.6820   | 0.4513    | 0.9171     | 0.6049     | 0.8371     | 0.5227     | **$129,800** |
| Random Forest       | 0.26 | 0.6444   | 0.4223    | **0.9225** | 0.5793     | 0.8434     | 0.5457     | $125,300     |
| XGBoost             | 0.32 | 0.6615   | 0.4349    | 0.9198     | 0.5906     | 0.8453     | 0.5330     | $127,300     |
| LightGBM            | 0.16 | 0.6686   | 0.4398    | 0.9091     | **0.5929** | **0.8455** | **0.5422** | $126,700     |

> **业务收益假设**：成功挽留一位流失客户收益 $500，错误干预一位非流失客户成本 $100。所有模型通过阈值优化后，Recall 均达到 85%+，确保不遗漏高风险客户。

### ROC-AUC 水平评估

对于该 Telco 公开数据集，ROC-AUC 0.84-0.85 属于**正常优秀水平**（业界 benchmark 通常 0.83-0.87）。四个模型间差距仅 0.008，说明特征工程质量比模型选择更重要。

---

## 模型选型结论


| 场景                     | 推荐模型            | 理由                                 |
| -------------------------- | --------------------- | -------------------------------------- |
| 追求可解释性             | Logistic Regression | 每个特征有权重系数，可直接解读       |
| 追求最高召回（不漏流失） | **Random Forest**   | Recall 92.25%，综合指标优秀          |
| 追求最大业务收益         | Logistic Regression | $129,800                             |
| 生产环境上线（速度优先） | LightGBM            | 模型仅 731KB，推理极快，ROC-AUC 最高 |

**综合推荐：Random Forest**，结合特征重要性可制定清晰的业务策略。

---

## 特征重要性 Top 10


| Rank | 特征             | 业务含义                                   |
| ------ | ------------------ | -------------------------------------------- |
| 1    | tenure           | 在网时长（新客户流失率最高）               |
| 2    | Contract         | 合同类型（月付流失率最高）                 |
| 3    | MonthlyCharges   | 月费（高月费→价格敏感）                   |
| 4    | InternetService  | 互联网服务类型（光纤客户流失高）           |
| 5    | OnlineSecurity   | 在线安全服务（没有→流失率高）             |
| 6    | ChargeDiff       | 费用偏差（当前月费高于历史均价→涨价冲击） |
| 7    | MultipleLines    | 多条电话线                                 |
| 8    | PaymentMethod    | 支付方式（电子支票→流失率最高）           |
| 9    | TechSupport      | 技术支持（没有→问题无法解决→流失）       |
| 10   | PaperlessBilling | 电子账单（可能被忽略→欠费→流失）         |

---

## 主要发现

- 月付客户流失率是年付客户的 5-10 倍
- 0-12 个月客户流失率最高
- Fiber optic 客户流失率显著偏高
- 没有在线安全/技术支持的客户流失率更高
- Electronic check 流失率最高；自动扣款流失率最低

后续可以提出针对性的策略。

---

## 编码策略说明


| 模型                | 编码方式                      | 说明                                                   |
| --------------------- | ------------------------------- | -------------------------------------------------------- |
| Logistic Regression | OneHot                        | 线性模型需要，避免虚假序关系                           |
| Random Forest       | Ordinal (整数)                | 树模型用 ordinal 即可，onehot 反而分散信息             |
| XGBoost             | Ordinal (整数)                | 同 RF，树模型通过阈值分裂，序关系影响有限              |
| LightGBM            | Ordinal + categorical_feature | 整数编码但告知 LightGBM 为类别特征，利用其最优分组算法 |

## 多重共线性处理

为解决特征间的共线性问题，在预处理阶段按模型类型差异化处理：


| 删除特征           | 适用模型               | 原因                                                      |
| -------------------- | ------------------------ | ----------------------------------------------------------- |
| `TotalCharges`     | 全部                   | ≈ MonthlyCharges × tenure，高度冗余                     |
| `AvgMonthlyCharge` | 全部                   | = TotalCharges / tenure，保留 MonthlyCharges 直接价格信号 |
| `tenure`           | 仅 Logistic Regression | 与 TenureGroup 完全共线；线性模型用分箱变量捕捉非线性关系 |

树模型保留 `tenure`（能自动找到最优切分点），也保留 `TenureGroup`（作为类别特征辅助分裂）。
