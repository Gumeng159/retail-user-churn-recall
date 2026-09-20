# 用户流失预警与 AI 辅助召回策略设计

基于 UCI Online Retail II 公开数据集，完成从数据清洗、流失定义、特征工程、建模评估、用户分层、AI 召回文案生成到 A/B 实验设计的完整分析流程。

## 项目背景
电商企业面临用户流失问题，高价值用户流失对 GMV 影响显著。本项目通过历史交易数据识别流失风险，结合用户价值分层设计召回策略，并接入 DeepSeek API 生成个性化召回文案。

## 数据来源
UCI Online Retail II: https://archive.ics.uci.edu/dataset/502/online+retail+ii
- 时间范围：2009-12-01 到 2011-12-09
- 原始交易行数：约 106.7 万，清洗后保留 77.6 万有效交易，覆盖 5,852 名用户。

## 技术栈
Python (pandas, scikit-learn)、SQL、DeepSeek API、statsmodels

## 项目流程
1. 数据清洗：剔除匿名交易、取消单、负值、非商品项
2. 流失定义：90 天未复购为流失
3. 特征工程：R/F/M、平均购买间隔、品类数、客单价、购买跨度
4. 建模评估：逻辑回归 vs 随机森林（AUC 0.805，KS 0.494）
5. 用户分层：价值×风险九宫格，9 类策略
6. AI 文案：DeepSeek API 生成 66 条文案（无重复、无套话）
7. A/B 实验设计：样本量 6410，因样本不足仅做方案
8. 离线回测：模拟 GMV 与 ROI（盈亏平衡约 10% 转化率）

## 边界说明
- 公开数据集个人项目，非企业真实项目。
- A/B 实验仅完成方案设计，未执行线上实验。
- 收益为离线模拟，非真实提升。
- LLM 输出经过人工审核和数字校验。

## 运行说明
1. 下载数据放到根目录，命名为 `online_retail_II.xlsx`
2. `pip install -r requirements.txt`
3. 按 `src/01` 到 `src/07` 顺序运行脚本，产出会自动保存在 `results/`

*注：由于本地开发环境为 Python 3.7，`04_ai_recall_generation.py` 使用了旧版 `openai` 库 API 写法，生成结果的 CSV 已存放在 `results` 目录中。*