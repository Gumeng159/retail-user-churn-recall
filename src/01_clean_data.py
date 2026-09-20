import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# 1. 获取项目根目录（scr 的上一级）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 2. 获取 results 文件夹路径，如果不存在会自动创建
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# 3. 读取原始 Excel（放在根目录，与 scr 和 results 同级）
path = os.path.join(BASE_DIR, 'online_retail_II.xlsx')

df1 = pd.read_excel(path, sheet_name='Year 2009-2010', engine='openpyxl')
df2 = pd.read_excel(path, sheet_name='Year 2010-2011', engine='openpyxl')

print("df1 shape:", df1.shape)
print("df2 shape:", df2.shape)

# ==================== 4. 列名统一 ====================
def clean_columns(df):
    df.columns = [str(c).strip() for c in df.columns]
    df.rename(columns={'Customer ID': 'CustomerID'}, inplace=True)
    return df

df1 = clean_columns(df1)
df2 = clean_columns(df2)

df1['InvoiceDate'] = pd.to_datetime(df1['InvoiceDate'], errors='coerce')
df2['InvoiceDate'] = pd.to_datetime(df2['InvoiceDate'], errors='coerce')

df1['source'] = '2009-2010'
df2['source'] = '2010-2011'

# ==================== 5. 合并 ====================
df = pd.concat([df1, df2], ignore_index=True)
print("\n合并后行数：", len(df))

# ==================== 6. 清洗 ====================
df = df[df['CustomerID'].notna()].copy()
print("剔除空 CustomerID 后行数：", len(df))

df['Invoice'] = df['Invoice'].astype(str)
df = df[~df['Invoice'].str.startswith('C')]
print("剔除取消单后行数：", len(df))

df = df[df['Quantity'] > 0]
df = df[df['Price'] > 0]
print("剔除负值后行数：", len(df))

non_product = ['POST','D','M','BANK CHARGES','DOT','C2','PADS',
               'AMAZONFEE','S','ADJUST','CRUK','B','TEST001',
               'TEST002','SP1002','GIFT']
df['StockCode'] = df['StockCode'].astype(str)
df = df[~df['StockCode'].isin(non_product)]
df = df[df['StockCode'].str.match(r'^\d{5}[A-Za-z]*$', na=False)]
print("剔除非商品后行数：", len(df))

key = ['Invoice','StockCode','Quantity','InvoiceDate','Price','CustomerID']
df = df.drop_duplicates(subset=key, keep='first')
print("去重后行数：", len(df))
print("唯一用户数：", df['CustomerID'].nunique())
print("日期范围：", df['InvoiceDate'].min(), "到", df['InvoiceDate'].max())

# ==================== 7. 保存到 results 文件夹 ====================
df.to_csv(os.path.join(RESULTS_DIR, 'online_retail_clean.csv'), index=False, encoding='utf-8-sig')
print("\n已保存 results/online_retail_clean.csv")