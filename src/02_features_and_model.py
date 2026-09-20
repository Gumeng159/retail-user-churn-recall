import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, roc_curve
import warnings
warnings.filterwarnings('ignore')

# 1. 动态获取项目根目录和 results 文件夹路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# 2. 读取清洗后的数据（从 results 里读）
data_path = os.path.join(RESULTS_DIR, 'online_retail_clean.csv')
print(f"正在读取数据: {data_path}")
df = pd.read_csv(data_path, parse_dates=['InvoiceDate'])
df['CustomerID'] = df['CustomerID'].astype(int)

# ==================== 3. 流失标签 ====================
cutoff = pd.Timestamp('2011-09-09 23:59:59')
perf_end = cutoff + pd.Timedelta(days=90)

obs = df[df['InvoiceDate'] <= cutoff].copy()
perf = df[(df['InvoiceDate'] > cutoff) & (df['InvoiceDate'] <= perf_end)].copy()

obs_users = set(obs['CustomerID'].unique())
perf_users = set(perf['CustomerID'].unique())

labels = pd.DataFrame({'CustomerID': list(obs_users)})
labels['churn'] = labels['CustomerID'].apply(lambda x: 1 if x not in perf_users else 0)

print("观察期用户数：", len(labels))
print("流失率：", round(labels['churn'].mean(), 4))

# ==================== 4. 特征工程 ====================
obs['TotalPrice'] = obs['Quantity'] * obs['Price']
snapshot_date = cutoff + pd.Timedelta(days=1)

# 4.1 RFM
rfm = obs.groupby('CustomerID').agg(
    recency=('InvoiceDate', lambda x: (snapshot_date - x.max()).days),
    frequency=('Invoice', 'nunique'),
    monetary=('TotalPrice', 'sum')
).reset_index()

# 4.2 平均购买间隔（按发票取唯一日期）
invoice_dates = obs.groupby(['CustomerID', 'Invoice'])['InvoiceDate'].min().reset_index()

def avg_interval_by_invoice(dates):
    dates = sorted(dates)
    if len(dates) < 2:
        return np.nan
    intervals = [(dates[i] - dates[i-1]).days for i in range(1, len(dates))]
    return np.mean(intervals)

interval_df = invoice_dates.groupby('CustomerID')['InvoiceDate'].apply(avg_interval_by_invoice).reset_index()
interval_df.columns = ['CustomerID', 'avg_interval']

# 4.3 最近30天订单数
last_30 = obs[obs['InvoiceDate'] > (cutoff - pd.Timedelta(days=30))]
recent_orders = last_30.groupby('CustomerID')['Invoice'].nunique().reset_index()
recent_orders.columns = ['CustomerID', 'recent_30_orders']

# 4.4 购买品类数
category_count = obs.groupby('CustomerID')['StockCode'].nunique().reset_index()
category_count.columns = ['CustomerID', 'category_count']

# 4.5 客单价
aov = obs.groupby('CustomerID').apply(
    lambda x: x['TotalPrice'].sum() / x['Invoice'].nunique()
).reset_index()
aov.columns = ['CustomerID', 'avg_order_value']

# 4.6 购买跨度
span = obs.groupby('CustomerID')['InvoiceDate'].apply(
    lambda x: (x.max() - x.min()).days
).reset_index()
span.columns = ['CustomerID', 'purchase_span']

# ==================== 5. 合并特征与标签 ====================
features = rfm.merge(interval_df, on='CustomerID', how='left')
features = features.merge(recent_orders, on='CustomerID', how='left')
features = features.merge(category_count, on='CustomerID', how='left')
features = features.merge(aov, on='CustomerID', how='left')
features = features.merge(span, on='CustomerID', how='left')
features = features.merge(labels, on='CustomerID', how='left')

features['avg_interval'] = features['avg_interval'].fillna(features['avg_interval'].median())
features['recent_30_orders'] = features['recent_30_orders'].fillna(0)

print("特征表形状：", features.shape)

# ==================== 6. 建模与评估 ====================
feature_cols = ['recency', 'frequency', 'monetary', 'avg_interval',
                'recent_30_orders', 'category_count', 'avg_order_value', 'purchase_span']
X = features[feature_cols]
y = features['churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 6.1 逻辑回归
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])
print("逻辑回归 AUC：", round(lr_auc, 4))

# 6.2 随机森林
rf = RandomForestClassifier(
    n_estimators=200, max_depth=6, min_samples_leaf=10,
    random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)
rf_pred = rf.predict_proba(X_test)[:, 1]
rf_auc = roc_auc_score(y_test, rf_pred)
print("随机森林 AUC：", round(rf_auc, 4))

# 6.3 KS
fpr, tpr, _ = roc_curve(y_test, rf_pred)
ks = max(tpr - fpr)
print("KS：", round(ks, 4))

# 6.4 Top 10% Lift
n_top = int(len(y_test) * 0.1)
top_idx = np.argsort(rf_pred)[::-1][:n_top]
lift = y_test.iloc[top_idx].mean() / y_test.mean()
print("Top 10% Lift：", round(lift, 2))

# ==================== 7. 保存结果到 results 文件夹 ====================
features_path = os.path.join(RESULTS_DIR, 'features_v3.csv')
features.to_csv(features_path, index=False, encoding='utf-8-sig')

features['churn_prob'] = rf.predict_proba(X)[:, 1]
pred_path = os.path.join(RESULTS_DIR, 'churn_predictions_v3.csv')
features[['CustomerID', 'churn', 'churn_prob']].to_csv(pred_path, index=False, encoding='utf-8-sig')

print(f"\n已保存特征表: {features_path}")
print(f"已保存预测结果: {pred_path}")