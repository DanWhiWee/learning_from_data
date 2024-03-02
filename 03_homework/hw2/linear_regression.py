
# ====================================================
# linear regression for classification
# y = WX
# d = 2; x1 ~ U(-1, 1); x2 ~ U(-1, 1)
# Algorithm:
# Step1. 生成目标函数f，从[−1, 1] × [−1, 1]范围中随机取两
# ## 个点，这两点构成的直线即为目标函数f，直线f上的点x满足Wx=0，
# ## 对于区域中任意一点x_n，若W*x_n > 0，则y为1，若W*x_n < 0，则y为-1
# Step2. 使用线性回归计算最优权重w（将y视为连续值，虽然只取1与-1）
# ## w = inv(X^T * X) * X^T * y
# =====================================================

import numpy as np
import matplotlib.pyplot as plt


# 函数表达式转化，给定权重w，与x1，求x2
# w为1*3的matrix，顺序为w0, w1, w2
# x1为单个元素或者1维数组
def f(w,x1):
    return -(w[0,1]*x1+w[0,0])/w[0,2]

def lr_expri(loop_n=None, train_size=None, test_size=None):
    ## 基本设置
    iteration = loop_n
    D_size = train_size
    # seed = 42

    ## 目标函数权重
    w_f = np.empty((0, 3))
    ## 线性回归权重结果
    w_lr = np.empty((0,3))
    ## 训练集误差
    err_in = []
    err_out = []

    # 循环指定次数
    for i in range(iteration):
        ## 生成f
        a1 = np.random.uniform(-1, 1, 1)
        b1 = np.random.uniform(-1, 1, 1)
        a2 = np.random.uniform(-1, 1, 1)
        b2 = np.random.uniform(-1, 1, 1)
        w0 = np.matrix([a2 * b1 - a1 * b2, b2 - b1, a1 - a2])
        w_f = np.append(w_f, w0.T, axis=0)
        ## 生成训练样本
        x0 = np.full(D_size, 1)
        x1 = np.random.uniform(-1, 1, D_size)
        x2 = np.random.uniform(-1, 1, D_size)
        X = np.matrix([x0, x1, x2]).T
        ## h(x)的符号
        y = np.asarray(np.sign(np.matmul(X, w0))).flatten()

        ## 生成测试样本
        x0_test = np.full(test_size, 1)
        x1_test = np.random.uniform(-1, 1, test_size)
        x2_test = np.random.uniform(-1, 1, test_size)
        X_test = np.matrix([x0_test, x1_test, x2_test]).T
        ## h(x)的符号
        y_test = np.asarray(np.sign(np.matmul(X_test, w0))).flatten()

        ## 利用线性回归的公式计算权重w
        w_lin = np.linalg.pinv(X.T @ X) @ X.T @ y
        w_lr = np.append(w_lr,w_lin,axis=0)
        ## 计算拟合结果
        y_hat = np.asarray(np.sign(X @ w_lin.T)).flatten()
        err_in.append(np.sum(y != y_hat) / len(y))
        y_test_hat = np.asarray(np.sign(X_test @ w_lin.T)).flatten()
        err_out.append(np.sum(y_test != y_test_hat) / len(y_test))
    return w_f, w_lr, err_in, err_out




if __name__ == "__main__":
    w_f, w_lr, err_in, err_out = lr_expri(loop_n=1000, train_size=100, test_size=10000)

    # 目标函数的图像以及正确的y值
    x_plt = np.linspace(-1, 1, 100)
    y_plt = f(w_f[720], x_plt)
    ax = plt.gca()
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    plt.plot(x_plt, y_plt, linestyle='dashed', linewidth=2.0, label='target')
    y_hat_plt = f(w_lr[720], x_plt)
    plt.plot(x_plt, y_hat_plt, linewidth=2.0, label='fitting')
    plt.legend(loc="upper left")
    plt.show()