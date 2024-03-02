## 感知机

### 算法原理

二分类场景下，假设数据集线性可分，数据特征为$\textbf{x}_N$，预测目标为$\textbf{y}_N$，现在用感知机算法，使用特征$\textbf{x}_N$预测$\textbf{y}_N$，感知机的假设集可表示为：
$$
h(x) = sign((\sum_{i=1}^dw_ix_i)+b)
$$
写成向量形式即为：
$$
h(x) = sign(\textbf{w}^T\textbf{x})
$$
每次循环时，从分类错误的点中随机选取点$(\textbf{x}(t), y(t))$，按如下方式更新权重$\textbf{w}$（其中$t$表示循环次数）：
$$
\textbf{w}(t+1)=\textbf{w}(t)+y(t)\textbf{x}(t)
$$

### 理解1：向量几何含义

回到假设集函数的向量形式，分两种情况讨论权重的更新逻辑：

- 当$y(t)$=1时，$h(x)$的预测值为-1，即$\textbf{w}^T\textbf{x}<0$，可知向量$\textbf{w}$与$\textbf{x}$的夹角为钝角，此时$\textbf{w}$的更新逻辑为$\textbf{w}(t+1)=\textbf{w}(t)+\textbf{x}(t)$，则更新后的$\textbf{w}$与$\textbf{x}$的夹角必然小于更新之前，因此$\textbf{w}^T\textbf{x}$会逐渐倾向取大于0的值
- 当$y(t)$=-1时，$h(x)$的预测值为1，即$\textbf{w}^T\textbf{x}>0$，可知向量$\textbf{w}$与$\textbf{x}$的夹角为锐角，此时$\textbf{w}$的更新逻辑为$\textbf{w}(t+1)=\textbf{w}(t)-\textbf{x}(t)$，则更新后的$\textbf{w}$与$\textbf{x}$的夹角必然大于更新之前，因此$\textbf{w}^T\textbf{x}$会逐渐倾向取小于0的值

### 理解2：梯度下降（是否可行待确认）