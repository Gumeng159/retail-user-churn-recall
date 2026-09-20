import os
import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

# 1. 动态获取项目根目录和 results 文件夹路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# ==================== 2. 读取数据 ====================
strategy_path = os.path.join(RESULTS_DIR, 'recall_strategies_v3.csv')
df = pd.read_csv(strategy_path)
target = df[df['strategy'].str.startswith('A_')].copy()

n_users = len(target)
avg_aov = target['avg_order_value'].mean()

print(f"策略A用户数：{n_users}")
print(f"平均客单价：{avg_aov:.2f}")

# 关键修复：读取最原始的 API 生成文件（而不是被清洗过的 clean 文件）
ai_path = os.path.join(RESULTS_DIR, 'ai_recall_strategies_v3.csv')
ai_df = pd.read_csv(ai_path)

# 从原始 'offer' 列中提取优惠金额（比如从“满2000减300”里提取300）
def parse_discount(text):
    if pd.isna(text):
        return 200  # 兜底：确保不为 0
    text = str(text)
    # 优先匹配 "减数字"
    match = re.search(r'减\\s*(\\d+)', text)
    if match:
        return int(match.group(1))
    # 次选匹配 "数字元"
    match = re.search(r'(\\d+)元', text)
    if match:
        return int(match.group(1))
    # 针对免运费的情况，按 50 元成本估算
    if '免运费' in text or '包邮' in text:
        return 50
    # 如果包含券但没提到具体金额，默认按 150 元估算
    if '券' in text:
        return 150
    return 200

ai_df['discount_amount'] = ai_df['offer'].apply(parse_discount)
avg_discount = ai_df['discount_amount'].mean()

print(f"平均优惠金额：{avg_discount:.2f}")

# ==================== 3. 固定成本假设下的模拟收益 ====================
total_cost = n_users * avg_discount
print("-" * 60)
print(f"总触达用户：{n_users}")
print(f"总优惠成本（固定）：{total_cost:.2f}")

conversion_rates = [0.03, 0.05, 0.08, 0.10, 0.15]

results = []
for cr in conversion_rates:
    converted_users = n_users * cr
    gmv = converted_users * avg_aov
    profit = gmv - total_cost
    roi = profit / total_cost if total_cost > 0 else 0
    results.append({
        '转化率': f"{cr:.0%}",
        '转化用户数': round(converted_users, 1),
        '预估GMV': round(gmv, 2),
        '优惠成本': round(total_cost, 2),
        '利润': round(profit, 2),
        'ROI': round(roi, 2)
    })

result_df = pd.DataFrame(results)
print("-" * 60)
print("模拟收益（固定成本假设）")
print("-" * 60)
print(result_df.to_string(index=False))

# ==================== 4. 三种优惠方案敏感性对比 ====================
print("-" * 60)
print("三种优惠方案对比")
print("-" * 60)

scenarios = {
    '当前方案（248元券）': 248,
    '保守方案（150元券）': 150,
    '轻度方案（100元券）': 100
}

for name, discount in scenarios.items():
    total_cost_scenario = n_users * discount
    print(f"\\n{name}，总成本：{total_cost_scenario:.0f}")
    for cr in conversion_rates:
        gmv = n_users * cr * avg_aov
        profit = gmv - total_cost_scenario
        roi = profit / total_cost_scenario
        print(f"  转化率{cr:.0%}：GMV={gmv:.0f}, 利润={profit:.0f}, ROI={roi:.2f}")

# ==================== 5. 保存到 results 文件夹 ====================
save_path = os.path.join(RESULTS_DIR, 'ab_test_simulation_v2.csv')
result_df.to_csv(save_path, index=False, encoding='utf-8-sig')
print(f"\\n已保存模拟收益结果到: {save_path}")