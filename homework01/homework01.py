import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 1. 加载数据
df = pd.read_csv('Carseats.csv')

# 2. 数据预处理：将 ShelveLoc 转换为哑变量 (Dummy Variables)
# drop_first=True 表示将第一个类别（Bad）作为基准组
df_encoded = pd.get_dummies(df, columns=['ShelveLoc'], drop_first=True, dtype=int)

# 3. 定义自变量和因变量
X = df_encoded[['Price', 'Income', 'Advertising', 'ShelveLoc_Good', 'ShelveLoc_Medium']]
y = df_encoded['Sales']
X = sm.add_constant(X) # 添加截距项

# 4. 建立多元线性回归模型
model = sm.OLS(y, X).fit()

# 5. 提取模型拟合报告
print(model.summary())

# 6. 计算 VIF (方差膨胀因子)
vif_data = pd.DataFrame()
vif_data["变量"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print("\n--- VIF 计算结果 ---")
print(vif_data)