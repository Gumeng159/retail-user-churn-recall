import os
import pandas as pd
import re
import warnings
warnings.filterwarnings('ignore')

# 1. 动态获取项目根目录和 results 文件夹路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# 2. 读取你粘贴进来的 AI 策略文件（自动从 results 读取）
input_path = os.path.join(RESULTS_DIR, 'ai_recall_strategies_v3.csv')
print(f"正在读取 AI 策略文件: {input_path}")

if not os.path.exists(input_path):
    raise FileNotFoundError(f"找不到文件，请确认 {input_path} 是否已放入 results 文件夹！")

ai_df = pd.read_csv(input_path)

# ==================== 3. 质量检查 ====================
print("\\n" + "="*60)
print("质量检查")
print("="*60)
print("总条数：", len(ai_df))
print("唯一文案数：", ai_df['message'].nunique())

bad_phrases = ['好久不见', '好久没见', '期待您回来']
for phrase in bad_phrases:
    count = ai_df['message'].str.contains(phrase, na=False).sum()
    print(f"  '{phrase}': {count} 条")

# ==================== 4. offer 清洗 ====================
def clean_offer(offer):
    if pd.isna(offer):
        return offer
    offer = str(offer)
    # 去掉中文括号和解释性文字
    offer = re.split(r'[（(]', offer)[0]
    offer = re.split(r'，门槛为', offer)[0]
    offer = re.split(r'，符合', offer)[0]
    offer = re.split(r'，不超过', offer)[0]
    offer = re.split(r'，门槛', offer)[0]
    return offer.strip()

ai_df['offer_clean'] = ai_df['offer'].apply(clean_offer)

# ==================== 5. offer 类型归并 ====================
def simplify_offer(offer):
    if pd.isna(offer):
        return offer
    offer = str(offer)
    if '免运费' in offer or '包邮' in offer:
        return '免运费'
    if re.search(r'满\\d+元?减\\d+元?', offer):
        return '满减券'
    if '折' in offer:
        return '折扣券'
    if '券' in offer:
        return '优惠券'
    return '其他'

ai_df['offer_type'] = ai_df['offer_clean'].apply(simplify_offer)

# ==================== 6. 渠道归并 ====================
def simplify_channel(ch):
    if pd.isna(ch):
        return ch
    ch = ch.replace(' ', '')
    channels = []
    if '短信' in ch:
        channels.append('短信')
    if 'APP推送' in ch or 'App推送' in ch or 'APP弹窗' in ch or '会员中心弹窗' in ch:
        channels.append('APP推送')
    if '企微' in ch or '企业微信' in ch:
        channels.append('企微')
    if '外呼' in ch or '电话' in ch:
        channels.append('外呼')
    if '邮件' in ch:
        channels.append('邮件')
    if '微信服务号' in ch:
        channels.append('微信服务号')
    return '+'.join(channels) if channels else ch

ai_df['channel_clean'] = ai_df['channel'].apply(simplify_channel)

# ==================== 7. 打印清洗结果 ====================
print("\\n最终优惠类型分布：")
print(ai_df['offer_type'].value_counts())

print("\\n最终渠道分布：")
print(ai_df['channel_clean'].value_counts())

# ==================== 8. 保存清洗后结果到 results ====================
save_path = os.path.join(RESULTS_DIR, 'ai_recall_strategies_v3_clean.csv')
ai_df.to_csv(save_path, index=False, encoding='utf-8-sig')
print(f"\\n已保存清洗后的文件到: {save_path}")