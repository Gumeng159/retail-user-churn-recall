import os
import pandas as pd
import json
import time
import random
import openai
import warnings

warnings.filterwarnings('ignore')

# 1. 动态获取项目根目录和 results 文件夹
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# 2. 配置 API
API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-key")
if API_KEY == "sk-key":
    print("⚠️ 警告: 检测到未配置 DEEPSEEK_API_KEY，请检查 PyCharm 运行配置的环境变量！")

openai.api_key = API_KEY
openai.api_base = "https://api.deepseek.com"

# 3. 读取分层结果
df = pd.read_csv(os.path.join(RESULTS_DIR, 'recall_strategies_v3.csv'))
df['CustomerID'] = df['CustomerID'].astype(int)

target = df[df['strategy'].str.startswith('A_')].copy()
print(f"策略A用户数：{len(target)}")

# 4. 多样性控制池
ANGLES = [
    "从新品上市角度切入", "从会员专属权益角度切入", "从限时优惠角度切入",
    "从用户偏好品类的新到货角度切入", "从专属客服服务角度切入",
    "从季节/节日角度切入", "从用户历史购买的高频品类角度切入", "从免运费/包邮角度切入"
]
OPENINGS = ["直接说优惠", "以关心语气开头", "以新品推荐开头", "以专属感开头", "以限时提醒开头"]


# 5. 构建 Prompt
def build_prompt(row, angle, opening):
    return f"""
你是一个电商用户运营分析师。根据以下用户画像，生成召回策略。

用户画像：
- 价值分层：{row['value_level']}
- 风险分层：{row['risk_level']}
- 流失概率：{row['churn_prob']:.2f}
- 最近购买间隔天数：{row['recency']}
- 历史订单数：{row['frequency']}
- 总消费金额：{row['monetary']:.2f}
- 客单价：{row['avg_order_value']:.2f}
- 购买品类数：{row['category_count']}
- 平均购买间隔：{row['avg_interval']:.1f}天
- 购买跨度：{row['purchase_span']}天

要求：
1. 判断用户价值与流失风险等级。
2. 给出召回渠道、优惠力度、触达时机、推荐品类方向。
3. 生成一条不超过80字的召回文案。
   - 本次文案请{angle}。
   - 开头方式：{opening}。
   - 禁止使用以下套话：“好久不见”“期待您回来”“好久没见您了”。
   - 语气根据用户购买品类数和流失时长调整。
   - 不要夸张，不要编造。
4. 优惠力度必须与用户历史客单价匹配，满减门槛不超过客单价的1.5倍。
5. reason 中引用的数字必须来自上述画像，不得编造。
6. 必须输出纯JSON格式，字段如下（不要有多余文字）：
   - segment, channel, offer, timing, category_direction, message, reason
"""


# 6. 调用 API
def generate_strategy(row, retry=3):
    angle = random.choice(ANGLES)
    opening = random.choice(OPENINGS)
    prompt = build_prompt(row, angle, opening)
    for i in range(retry):
        try:
            # 0.8.0 版本使用 openai.ChatCompletion.create
            response = openai.ChatCompletion.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8
            )
            content = response['choices'][0]['message']['content']
            content = content.strip().strip('`').replace('json\\n', '', 1).strip()
            result = json.loads(content)

            result['angle'] = angle
            result['opening'] = opening
            return result
        except Exception as e:
            print(f"重试 {i + 1}/{retry}，错误：{e}")
            time.sleep(2)
    return None


# 7. 批量生成
results = []
failed = []

for idx, row in target.iterrows():
    print(f"生成中：CustomerID={row['CustomerID']}")
    result = generate_strategy(row)
    if result:
        result['CustomerID'] = int(row['CustomerID'])
        result['churn_prob'] = round(row['churn_prob'], 4)
        result['monetary'] = round(row['monetary'], 2)
        result['recency'] = int(row['recency'])
        result['avg_order_value'] = round(row['avg_order_value'], 2)
        results.append(result)
    else:
        failed.append(int(row['CustomerID']))
    time.sleep(1)

# 8. 保存结果
ai_df = pd.DataFrame(results)
print(f"\\n生成成功：{len(ai_df)} 条")
print(f"失败：{len(failed)} 条")

if len(ai_df) > 0:
    save_path = os.path.join(RESULTS_DIR, 'ai_recall_strategies_v3.csv')
    ai_df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"已保存到: {save_path}")
