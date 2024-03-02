# 假设训练集样本量N=10，使用线性回归的权重作为pla算法的初始权重
# 根据pla算法更新权重，直到可以完全正确区分这10个点
# 循环实验1000次，求pla算法收敛所需的平均迭代次数


import numpy as np
import matplotlib.pyplot as plt


# 函数表达式转化，给定权重w，与x1，求x2
# w为1*3的matrix，顺序为w0, w1, w2
# x1为单个元素或者1维数组
def f(w,x1):
    return -(w[0,1]*x1+w[0,0])/w[0,2]

def pla_with_lr_expri(loop_n=None, train_size=None):
    ## 基本设置
    iteration = loop_n
    D_size = train_size
    # seed = 42

    ## 目标函数权重
    w_f = np.empty((0, 3))
    ## 线性回归权重结果
    w_lr = np.empty((0,3))

    ## 收敛次数
    conv_j = []
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


        ## 利用线性回归的公式计算权重w，将作为PLA算法的初始权重
        w_lin = np.linalg.pinv(X.T @ X) @ X.T @ y
        w_lr = np.append(w_lr,w_lin,axis=0)

        ## 将线性回归的结果作为初始的权重
        w = w_lin

        ## 计算拟合结果
        y_hat = np.asarray(np.sign(X @ w.T)).flatten()
        y_tmp = y[y != y_hat]
        X_tmp = X[y != y_hat, :]

        j = 0
        ## 随机选取一个分类错误的点，用于更新权重，直至算法收敛
        while len(y_tmp) != 0:
            rand_idx = np.random.choice(range(len(y_tmp)))
            # 取出用于更新权重的记录
            x_i = X_tmp[rand_idx]
            y_i = y_tmp[rand_idx]
            # 更新权重
            w = w + y_i * x_i
            # 计算
            y_hat = np.asarray(np.sign(X @ w.T)).flatten()
            y_tmp = y[y != y_hat]
            X_tmp = X[y != y_hat, :]
            j = j + 1
        conv_j.append(j)
    return w_lr, conv_j

if __name__ == "__main__":
    w_lr, conv_j = pla_with_lr_expri(loop_n=1000, train_size=100)

