## 非线性转换
## 考虑目标函数：f(x1,x2) = sign(x1^2+x2^2-0.6)
## 在 [-1,1] X [-1,1] 的二维空间内均匀随机生成1000个点，随机选取其中
## 10%的点作为噪声，这些点的y值为随机的-1或者1，其余按照f(x1,x2)表达式
## 计算得到y
## 问题1：
## 在不使用非线性转换的情况下，特征向量表示为(1,x1,x2)，使用线性回归拟合
## 得到权重w，此时训练集误差是多少？

import numpy as np
import matplotlib.pyplot as plt

def nonlin_trans_expri(D_size, noise_rate, loop_n):
    err_rate_lin = []
    train_err_rate_nonlin = []
    test_err_rate_nonlin = []
    for i in range(loop_n):
        noise_size = round(D_size*noise_rate)
        x0 = np.full(D_size, 1)
        x1 = np.random.uniform(-1, 1, D_size)
        x2 = np.random.uniform(-1, 1, D_size)
        w0 = np.matrix([-0.6, 1, 1])
        X_nonlin = np.matrix([x0, x1**2, x2**2]).T
        ## f(x1,x2)
        y = np.asarray(np.sign(np.matmul(X_nonlin, w0.T))).flatten()
        ## 将10%的样本y值设置为噪声
        noise_idx = np.random.choice(range(len(y)),noise_size,replace=False)
        y[noise_idx] = np.random.choice([-1,1],noise_size)
        ## 直接使用常规的线性回归来分类
        X_lin = np.matrix([x0, x1, x2]).T
        w_lin = np.linalg.pinv(X_lin.T @ X_lin) @ X_lin.T @ y
        y_lin = np.asarray(np.sign(X_lin @ w_lin.T)).flatten()
        err_rate_lin.append(sum(y_lin != y)/D_size)

        ## 使用非线性转换拟合
        ### 训练集
        X_nonlin_fit = np.matrix([x0, x1, x2, x1*x2, x1 ** 2, x2 ** 2]).T
        w_nonlin = np.linalg.pinv(X_nonlin_fit.T @ X_nonlin_fit) @ X_nonlin_fit.T @ y
        y_nonlin = np.asarray(np.sign(X_nonlin_fit @ w_nonlin.T)).flatten()
        train_err_rate_nonlin.append(sum(y_nonlin != y) / D_size)
        ### 测试集
        x1_test = np.random.uniform(-1, 1, D_size)
        x2_test = np.random.uniform(-1, 1, D_size)
        X_nonlin_test = np.matrix([x0, x1_test ** 2, x2_test ** 2]).T
        y_test = np.asarray(np.sign(np.matmul(X_nonlin_test, w0.T))).flatten()
        ## 将10%的样本y值设置为噪声
        noise_idx_test = np.random.choice(range(len(y_test)), noise_size, replace=False)
        y_test[noise_idx_test] = np.random.choice([-1, 1], noise_size)
        X_nonlin_fit_test = np.matrix([x0, x1_test, x2_test, x1_test * x2_test, x1_test ** 2, x2_test ** 2]).T
        ## 计算结果函数在测试集上的误差
        y_nonlin_test = np.asarray(np.sign(X_nonlin_fit_test @ w_nonlin.T)).flatten()
        test_err_rate_nonlin.append(sum(y_nonlin_test != y_test) / D_size)
    return err_rate_lin, train_err_rate_nonlin, test_err_rate_nonlin

# D_size=1000
# noise_rate=0.1
# noise_size = round(D_size * noise_rate)
# x0 = np.full(D_size, 1)
# x1 = np.random.uniform(-1, 1, D_size)
# x2 = np.random.uniform(-1, 1, D_size)
# w0 = np.matrix([-0.6, 1, 1])
# X_nonlin = np.matrix([x0, x1 ** 2, x2 ** 2]).T
# ## f(x1,x2)
# y = np.asarray(np.sign(np.matmul(X_nonlin, w0.T))).flatten()
# ## 将10%的样本y值设置为噪声
# noise_idx = np.random.choice(range(len(y)), noise_size, replace=False)
# y[noise_idx] = np.random.choice([-1, 1], noise_size)
#
# ## 看看图像
# ax = plt.gca()
# ax.set_xlim([-1, 1])
# ax.set_ylim([-1, 1])
# plt.scatter(x1, x2, c=y)
# plt.show()
#
# ## 直接使用常规的线性回归来分类
# X_nonlin_fit = np.matrix([x0, x1, x2, x1*x2, x1 ** 2, x2 ** 2]).T
# w_nonlin = np.linalg.pinv(X_nonlin_fit.T @ X_nonlin_fit) @ X_nonlin_fit.T @ y
# y_nonlin = np.asarray(np.sign(X_nonlin_fit @ w_nonlin.T)).flatten()
# sum(y_nonlin != y) / D_size


if __name__ == "__main__":
    err_rate_lin, train_err_rate_nonlin, test_err_rate_nonlin = nonlin_trans_expri(D_size=1000, noise_rate=0.1, loop_n=1000)