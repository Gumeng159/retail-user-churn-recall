import os
import pandas as pd
import numpy as np
from statsmodels.stats.proportion import proportion_effectsize
from statsmodels.stats.power import NormalIndPower
import warnings
warnings.filterwarnings('ignore')

# 1. 动态获取项目根目录和 results 文件夹路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# ==================== 2. A/B 实验样本量计算 ====================
baseline_rate = 0.08
mde_relative = 0.25
alpha = 0.05
power = 0.80

p1 = baseline_rate
p2 = baseline_rate * (1 + mde_relative)

effect_size = proportion_effectsize(p1, p2)
analysis = NormalIndPower()
n_per_group = analysis.solve_power(
    effect_size=effect_size, alpha=alpha, power=power,
    ratio=1, alternative='two-sided'
)

print("="*60)
print("A/B 实验样本量计算")
print("="*60)
print(f"基线复购率：{p1:.2%}")
print(f"目标复购率：{p2:.2%}")
print(f"每组所需样本量：{int(np.ceil(n_per_group))}")
print(f"总样本量：{int(np.ceil(n_per_group)) * 2}")

# ==================== 3. 实验设计文档 ====================
experiment_design = {
    "实验名称": "AI个性化召回文案 vs 标准模板文案",
    "实验目标": "验证AI个性化召回文案能否提升14天复购率",
    "实验假设": "实验组14天复购率显著高于对照组",
    "分流单位": "user_id",
    "分流比例": "50% 实验组 / 50% 对照组",
    "实验组": "AI个性化文案 + 满减券",
    "对照组": "标准模板文案 + 满减券",
    "主指标": "14天复购率",
    "次指标": "GMV、客单价、优惠券核销率",
    "护栏指标": "退订率、投诉率",
    "实验周期": "14天",
    "分析方法": "卡方检验 + 置信区间",
    "样本量": f"每组{int(np.ceil(n_per_group))}人，总计{int(np.ceil(n_per_group))*2}人",
    "当前可用样本": "策略A 66人，不足，仅做方案设计"
}

print("\\n" + "="*60)
print("实验设计文档")
print("="*60)
for k, v in experiment_design.items():
    print(f"{k}：{v}")

# ==================== 4. 保存实验设计到 results 文件夹 ====================
design_df = pd.DataFrame([experiment_design]).T
design_df.columns = ['内容']
design_path = os.path.join(RESULTS_DIR, 'ab_test_design.csv')
design_df.to_csv(design_path, encoding='utf-8-sig')
print(f"\\n实验设计文档已保存到: {design_path}")