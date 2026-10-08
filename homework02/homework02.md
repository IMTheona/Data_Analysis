# 作业 1 · 推导题

> **题目：** 证明正交设计下 $\widehat{\boldsymbol\beta}_{\text{Ridge}}=\dfrac{1}{1+\lambda}\widehat{\boldsymbol\beta}_{\text{OLS}}$

## 证明

Ridge 回归的闭合解为：

$$
\widehat{\boldsymbol\beta}_{\text{Ridge}}=\big(X^{T}X+\lambda I\big)^{-1}X^{T}Y
$$

正交设计条件为 $X^{T}X=I$（其中 $I$ 为 $p\times p$ 单位阵）。代入得：

$$
\begin{aligned}
\widehat{\boldsymbol\beta}_{\text{Ridge}}
&=\big(I+\lambda I\big)^{-1}X^{T}Y \\[4pt]
&=\big((1+\lambda)I\big)^{-1}X^{T}Y \\[4pt]
&=\frac{1}{1+\lambda}\,I\cdot X^{T}Y \\[4pt]
&=\frac{1}{1+\lambda}X^{T}Y
\end{aligned}
$$

又因为 $X^{T}X=I$ 时 OLS 解为：

$$
\widehat{\boldsymbol\beta}_{\text{OLS}}=\big(X^{T}X\big)^{-1}X^{T}Y=I\cdot X^{T}Y=X^{T}Y
$$

两式合并即得：

$$
\boxed{\ \widehat{\boldsymbol\beta}_{\text{Ridge}}=\frac{1}{1+\lambda}\widehat{\boldsymbol\beta}_{\text{OLS}}\ }
$$


# 作业 3 · 分析题

> **题目：** 为什么正则化前必须 Z-score 标准化？不做的后果？

## 解答

### 一、核心原因：惩罚项对系数"一视同仁"，但系数尺度被量纲绑架

Ridge 惩罚项 $\lambda\sum_j\beta_j^2$、Lasso 惩罚项 $\lambda\sum_j|\beta_j|$ 都不区分变量。而一个变量的系数尺度与其量纲直接相关：

- 若 $x_j$ 量纲很大（如收入以"元"计，取值上万），为了让拟合值量级合理，其系数 $\beta_j$ 天然就很小 → 惩罚项 $\beta_j^2$ 几乎可以忽略，**该变量实际上"逃过"了惩罚**；
- 若 $x_j$ 量纲很小（如 0~1 的比率变量），其系数天然偏大 → 惩罚项很大，**被过度压缩**，即使它是重要变量也会被压扁甚至归零。

**后果：** 惩罚强度不再由变量的重要性决定，而是由量纲决定，正则化失去公平性和意义。极端情况下，把同一变量从"元"换算成"万元"，最优模型和选出的变量集合都会改变——这是不可接受的。

### 二、另一个后果：截距项被误罚

若不做中心化（$y$ 不减均值、$X$ 不中心化），模型必须用截距吸收数据的整体水平。而正则化实现中通常不对截距加惩罚，若把未中心化的截距也写进 $\boldsymbol\beta$ 一起罚，会把截距强行拉向 0，导致模型系统性偏移。

### 三、标准做法

对每个特征做 Z-score 标准化：

$$
x_{ij}\leftarrow\frac{x_{ij}-\bar x_j}{s_j}\quad(\text{均值 }0,\ \text{方差 }1),
\qquad
y_i\leftarrow y_i-\bar y
$$

这样所有系数处在同一可比尺度上，惩罚才是"公平"的；同时 $\lambda$ 的含义在不同数据、不同变量组合间才具有可比性。

