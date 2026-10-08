"""
Elastic Net 分析 Netflix Movies and TV Shows 数据集
任务：二分类预测 content type (Movie vs TV Show)
Elastic Net 用于分类时等价于带 L1+L2 正则化的逻辑回归
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_auc_score, roc_curve)
from sklearn.feature_extraction.text import TfidfVectorizer

import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 加载数据
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, 'netflix_titles.csv')

df = pd.read_csv(csv_path)
print(f"数据规模: {df.shape[0]} 行 × {df.shape[1]} 列")
print(f"\n前 5 行预览:\n{df.head()}")
print(f"\n数据类型:\n{df.dtypes}")
print(f"\n缺失值统计:\n{df.isnull().sum()}")

# ============================================================
# 2. 数据清洗
# ============================================================
df['director'] = df['director'].fillna('Unknown')
df['cast'] = df['cast'].fillna('Not Available')
df['country'] = df['country'].fillna('Unknown')
df['rating'] = df['rating'].fillna('Not Rated')
df['listed_in'] = df['listed_in'].fillna('Unknown')
df['duration'] = df['duration'].fillna('Unknown')

df = df.dropna(subset=['date_added']).reset_index(drop=True)
print(f"\n清洗后数据规模: {df.shape[0]} 行")

# ============================================================
# 3. 特征工程
# ============================================================
def extract_duration_num(dur):
    """从 '90 min' 或 '2 Seasons' 中提取数值部分"""
    if pd.isna(dur) or dur == 'Unknown':
        return np.nan
    parts = str(dur).split()
    try:
        return float(parts[0])
    except (ValueError, IndexError):
        return np.nan

df['duration_num'] = df['duration'].apply(extract_duration_num)

df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
df['year_added'] = df['date_added'].dt.year
df['month_added'] = df['date_added'].dt.month

df['num_genres'] = df['listed_in'].apply(lambda x: len(str(x).split(', ')))
df['num_cast'] = df['cast'].apply(
    lambda x: len(str(x).split(', ')) if x != 'Not Available' else 0
)
df['desc_length'] = df['description'].fillna('').str.len()
df['desc_word_count'] = df['description'].fillna('').str.split().str.len()

df['is_movie'] = (df['type'] == 'Movie').astype(int)

print(f"\n目标变量分布:\n{df['is_movie'].value_counts()}")
print(f"Movie 占比: {df['is_movie'].mean():.2%}")

# ============================================================
# 4. 定义特征与目标
# ============================================================
numeric_features = ['release_year', 'year_added',
                    'month_added', 'num_genres', 'num_cast',
                    'desc_length', 'desc_word_count']
categorical_features = ['rating', 'country']

y = df['is_movie'].values

# ============================================================
# 5. 构建预处理管道
# ============================================================
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

text_transformer = Pipeline(steps=[
    ('tfidf', TfidfVectorizer(max_features=500, stop_words='english'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
        ('desc', text_transformer, 'description')
    ],
    remainder='drop'
)

# ============================================================
# 6. Elastic Net 超参数搜索 (LogisticRegressionCV + 并行)
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    df, y, test_size=0.2, random_state=42, stratify=y
)

l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n" + "=" * 60)
print("Elastic Net 超参数搜索 (LogisticRegressionCV, 并行)")
print("=" * 60)

best_score = -1
best_params = {}
best_model = None

for l1r in l1_ratios:
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegressionCV(
            Cs=np.logspace(-2, 1, 10),      # 搜索 10 个 C 值
            cv=cv,
            penalty='elasticnet',
            solver='saga',
            l1_ratios=[l1r],
            max_iter=3000,
            scoring='roc_auc',
            random_state=42,
            n_jobs=-1                        # ★ 多核并行
        ))
    ])
    model.fit(X_train, y_train)

    clf = model.named_steps['classifier']
    cv_auc = clf.scores_[1].mean()           # 二分类取正类 AUC
    C_best = clf.C_[0]

    print(f"  l1_ratio={l1r:.1f} → CV AUC={cv_auc:.4f}  (最优C={C_best:.4f})")

    if cv_auc > best_score:
        best_score = cv_auc
        best_params = {'l1_ratio': l1r, 'C': C_best}
        best_model = model

print(f"\n最优参数: l1_ratio={best_params['l1_ratio']:.1f}, "
      f"C={best_params['C']:.4f}")
print(f"交叉验证 AUC: {best_score:.4f}")

# 用最优模型作为最终模型
final_model = best_model

# ============================================================
# 7. 模型评估
# ============================================================
y_pred = final_model.predict(X_test)
y_prob = final_model.predict_proba(X_test)[:, 1]

print("\n" + "=" * 60)
print("测试集评估结果")
print("=" * 60)
print(f"\n准确率: {accuracy_score(y_test, y_pred):.4f}")
print(f"AUC: {roc_auc_score(y_test, y_prob):.4f}")
print(f"\n分类报告:\n{classification_report(y_test, y_pred, target_names=['TV Show', 'Movie'])}")

cm = confusion_matrix(y_test, y_pred)
print(f"混淆矩阵:\n{cm}")

# ============================================================
# 8. 可视化
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 8.1 混淆矩阵热力图
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['TV Show', 'Movie'],
            yticklabels=['TV Show', 'Movie'], ax=axes[0])
axes[0].set_xlabel('预测')
axes[0].set_ylabel('真实')
axes[0].set_title('混淆矩阵')

# 8.2 ROC 曲线
fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[1].plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, y_prob):.4f}', lw=2)
axes[1].plot([0, 1], [0, 1], 'k--', lw=1)
axes[1].set_xlabel('假正率 (FPR)')
axes[1].set_ylabel('真正率 (TPR)')
axes[1].set_title('ROC 曲线')
axes[1].legend()

# 8.3 特征重要性（取非零系数）
feature_names = (numeric_features +
                 list(final_model.named_steps['preprocessor']
                      .named_transformers_['cat']
                      .named_steps['onehot']
                      .get_feature_names_out(categorical_features)) +
                 ['tfidf_' + str(i) for i in range(500)])

coefs = final_model.named_steps['classifier'].coef_[0]

top_idx = np.argsort(np.abs(coefs))[-15:]
top_coefs = coefs[top_idx]
top_names = [feature_names[i] if i < len(feature_names) else f'feat_{i}'
             for i in top_idx]

colors = ['#dc2626' if c > 0 else '#2563eb' for c in top_coefs]
axes[2].barh(range(len(top_coefs)), top_coefs, color=colors)
axes[2].set_yticks(range(len(top_coefs)))
axes[2].set_yticklabels(top_names, fontsize=9)
axes[2].axvline(0, color='gray', lw=0.8)
axes[2].set_xlabel('系数值')
axes[2].set_title('Top 15 特征重要性 (红=Movie, 蓝=TV Show)')

plt.tight_layout()
plt.savefig('netflix_elasticnet_results.png', dpi=150)
plt.show()

# ============================================================
# 9. 稀疏性分析
# ============================================================
nonzero_idx = np.where(np.abs(coefs) > 1e-6)[0]
n_total = len(coefs)
n_nonzero = len(nonzero_idx)

print(f"\n{'=' * 60}")
print("Elastic Net 稀疏性分析")
print(f"{'=' * 60}")
print(f"总特征数: {n_total}")
print(f"非零系数数: {n_nonzero} ({n_nonzero/n_total:.1%})")
print(f"被淘汰特征数: {n_total - n_nonzero} ({(n_total-n_nonzero)/n_total:.1%})")
print(f"L1 比例 (l1_ratio): {best_params['l1_ratio']:.1f}")