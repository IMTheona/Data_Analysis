import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import (RidgeCV, LassoCV, ElasticNetCV,
                                  Lasso, Ridge, ElasticNet, lasso_path,
                                  ridge_regression)
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============ 1. 数据加载与预处理 ============
df = pd.read_csv('D:/2026SummerPractice/Data_Analysis/homework02/BostonHousing.csv').dropna()     # 删除含缺失值的行
X = df.drop(['medv'], axis=1).astype(float)        # 13 个特征
y = df['medv'].astype(float)
print(f"样本数 n = {X.shape[0]}, 特征数 p = {X.shape[1]}")
print(f"特征列表: {list(X.columns)}\n")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# ★ 正则化前必须标准化（用训练集 fit，避免信息泄漏）
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ============ 2. 三种模型 + 10 折 CV ============
cv = KFold(n_splits=10, shuffle=True, random_state=42)
alphas = np.logspace(-3, 3, 200)

ridge_cv = RidgeCV(alphas=alphas, cv=cv).fit(X_train_s, y_train)

lasso_cv = LassoCV(alphas=alphas, cv=cv, max_iter=100000,
                   random_state=42).fit(X_train_s, y_train)

enet_cv = ElasticNetCV(l1_ratio=[.1, .3, .5, .7, .9, .95, .99, 1],
                       alphas=alphas, cv=cv, max_iter=100000,
                       random_state=42).fit(X_train_s, y_train)

models = {'Ridge': ridge_cv, 'Lasso': lasso_cv, 'ElasticNet': enet_cv}

# ============ 3. 测试集性能与非零变量数 ============
print(f"{'模型':<12}{'最优α':>10}{'测试RMSE':>12}{'非零变量数':>12}")
print("-" * 48)
for name, m in models.items():
    pred = m.predict(X_test_s)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    nz = int(np.sum(np.abs(m.coef_) > 1e-8))
    alpha = getattr(m, 'alpha_', None)
    extra = f" (l1_ratio={m.l1_ratio_:.2f})" if name == 'ElasticNet' else ""
    print(f"{name:<12}{alpha:>10.4f}{rmse:>12.3f}{nz:>12d}{extra}")

# ============ 4. 系数路径图 ============
fig, axes_ = plt.subplots(1, 2, figsize=(14, 5))

# --- Lasso 路径 ---
alphas_lp, coefs_lp, _ = lasso_path(X_train_s, y_train,
                                    alphas=np.logspace(-3, 3, 100))
for i in range(coefs_lp.shape[0]):
    axes_[0].plot(np.log10(alphas_lp), coefs_lp[i], lw=1.5)
axes_[0].axvline(np.log10(lasso_cv.alpha_), color='k', ls='--', lw=1.5,
                 label=f'CV最优α={lasso_cv.alpha_:.3f}')
axes_[0].axhline(0, color='gray', lw=0.8, ls=':')
axes_[0].set_xlabel('log₁₀(α)'); axes_[0].set_ylabel('系数')
axes_[0].set_title('Lasso 系数路径：系数依次归零')
axes_[0].legend()

# --- Ridge 路径 ---
alphas_rp = np.logspace(-3, 3, 100)
coefs_rp = np.array([ridge_regression(X_train_s, y_train, alpha=a)
                     for a in alphas_rp]).T
for i in range(coefs_rp.shape[0]):
    axes_[1].plot(np.log10(alphas_rp), coefs_rp[i], lw=1.5)
axes_[1].axvline(np.log10(ridge_cv.alpha_), color='k', ls='--', lw=1.5,
                 label=f'CV最优α={ridge_cv.alpha_:.3f}')
axes_[1].axhline(0, color='gray', lw=0.8, ls=':')
axes_[1].set_xlabel('log₁₀(λ)'); axes_[1].set_ylabel('系数')
axes_[1].set_title('Ridge 系数路径：平滑收缩，无一触零')
axes_[1].legend()

plt.tight_layout()
plt.savefig('boston_regularization_paths.png', dpi=150)
plt.show()

# ============ 5. 1-SE 法则 ============
def one_se_lasso(X, y, alphas, cv):
    """返回 (lambda.min, lambda.1se) 及 CV 曲线"""
    mse_mean, mse_std = [], []
    for a in alphas:
        scores = -cross_val_score(Lasso(alpha=a, max_iter=100000),
                                  X, y, cv=cv,
                                  scoring='neg_mean_squared_error')
        mse_mean.append(scores.mean())
        mse_std.append(scores.std())
    mse_mean, mse_std = np.array(mse_mean), np.array(mse_std)
    imin = np.argmin(mse_mean)
    thr = mse_mean[imin] + mse_std[imin] / np.sqrt(10)   # 标准误
    cand = np.where(mse_mean <= thr)[0]
    i1se = cand[np.argmax(alphas[cand])]                 # 阈值内最大 α
    return alphas[imin], alphas[i1se], mse_mean, mse_std

a_min, a_1se, mse_m, mse_s = one_se_lasso(
    X_train_s, y_train, alphas, cv)

print(f"\n【1-SE 法则】")
print(f"  lambda.min = {a_min:.4f}")
print(f"  lambda.1se = {a_1se:.4f}  （更稀疏）")

lasso_min = Lasso(alpha=a_min, max_iter=100000).fit(X_train_s, y_train)
lasso_1se = Lasso(alpha=a_1se, max_iter=100000).fit(X_train_s, y_train)

print(f"\n{'方案':<12}{'测试RMSE':>12}{'非零变量数':>12}")
print("-" * 38)
for tag, m in [('λ.min', lasso_min), ('λ.1se', lasso_1se)]:
    rmse = np.sqrt(mean_squared_error(y_test, m.predict(X_test_s)))
    nz = int(np.sum(np.abs(m.coef_) > 1e-8))
    print(f"{tag:<12}{rmse:>12.3f}{nz:>12d}")

# ============ 6. 非零变量明细 ============
print("\n【Lasso(λ.min) 保留的变量】")
coefs = pd.Series(lasso_min.coef_, index=X.columns)
print(coefs[coefs != 0].sort_values(key=abs, ascending=False).round(4))

print("\n【Lasso(λ.1se) 保留的变量】")
coefs_1se = pd.Series(lasso_1se.coef_, index=X.columns)
print(coefs_1se[coefs_1se != 0].sort_values(key=abs, ascending=False).round(4))