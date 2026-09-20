import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# 1. 动态获取项目根目录和 results 文件夹
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# 2. 从 results 读取特征表和预测结果
features_path = os.path.join(RESULTS_DIR, 'features_v3.csv')
pred_path = os.path.join(RESULTS_DIR, 'churn_predictions_v3.csv')

features = pd.read_csv(features_path)
pred = pd.read_csv(pred_path)

features['CustomerID'] = features['CustomerID'].astype(int)
pred['CustomerID'] = pred['CustomerID'].astype(int)

df = features.merge(pred[['CustomerID', 'churn_prob']], on='CustomerID', how='left')

# ==================== 3. 价值与风险分层 ====================
df['value_level'] = pd.qcut(df['monetary'], q=3, labels=['低价值', '中价值', '高价值'])
df['risk_level'] = pd.cut(df['churn_prob'], bins=[0, 0.4, 0.7, 1.0],
                          labels=['低风险', '中风险', '高风险'])

# ==================== 4. 策略映射 ====================
def assign_strategy(row):
    v = row['value_level']
    r = row['risk_level']
    if v == '高价值' and r == '高风险':
        return 'A_专属优惠券+短信+忠诚计划'
    elif v == '高价值' and r == '中风险':
        return 'B_品类推荐+邮件触达'
    elif v == '高价值' and r == '低风险':
        return 'C_维护型触达，不投优惠'
    elif v == '中价值' and r == '高风险':
        return 'D_满减券+短信召回'
    elif v == '中价值' and r == '中风险':
        return 'E_关联推荐+邮件'
    elif v == '中价值' and r == '低风险':
        return 'F_低成本触达'
    elif v == '低价值' and r == '高风险':
        return 'G_低成本邮件，不投优惠'
    elif v == '低价值' and r == '中风险':
        return 'H_低成本站内信，不投优惠'
    else:
        return 'I_不主动触达'

df['strategy'] = df.apply(assign_strategy, axis=1)

print("分层分布（人数）：")
print(df.pivot_table(index='value_level', columns='risk_level',
                     values='CustomerID', aggfunc='count', fill_value=0))

print("\\n各分层流失率：")
print(df.pivot_table(index='value_level', columns='risk_level',
                     values='churn', aggfunc='mean', fill_value=0).round(3))

print("\\n策略分布：")
print(df['strategy'].value_counts().sort_index())

# ==================== 5. 保存到 results 文件夹 ====================
save_path = os.path.join(RESULTS_DIR, 'recall_strategies_v3.csv')
df.to_csv(save_path, index=False, encoding='utf-8-sig')
print(f"\\n已保存到: {save_path}")