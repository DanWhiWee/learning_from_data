
# ====================================================
# Perceptron Learning Algorithm
# y = sign(WX)
# d = 2; x1 ~ U(-1, 1); x2 ~ U(-1, 1)
# Algorithm:
# Step1. 生成目标函数f，从[−1, 1] × [−1, 1]范围中随机取两
# ## 个点，这两点构成的直线即为目标函数f，定义直线下方的
# ## 点为-1，上方的点为1；
# Step2. 在区域内随机生成N个点，根据f计算实际的y值；将PLA
# ## 初始的权重w设为0，则初始假设的y值都为0，所有点都分类
# ## 错误；后续每次循环，随机选取分类错误的1个点，用于更新
# ## 权重w；循环至算法收敛到g为止。每次循环的调整方式如下：
# ## W(t+1) = W(t) + y(t)*X(t)
# =====================================================

import numpy as np
import matplotlib.pyplot as plt
from  tqdm import tqdm


def pla_expri(loop_n=None, train_size=None, test_size=None):
    ## 基本设置
    iteration = loop_n
    D_size = train_size
    # seed = 42

    ## 生成f
    a1 = np.random.uniform(-1,1,1)
    b1 = np.random.uniform(-1,1,1)
    a2 = np.random.uniform(-1,1,1)
    b2 = np.random.uniform(-1,1,1)
    def f(x):
        return (b2-b1)/(a2-a1)*(x-a1) + b1

    w0 = np.array([a2*b1-a1*b2, b2-b1, a1-a2])

    ## 生成训练样本
    x0 = np.full(D_size,1)
    x1 = np.random.uniform(-1,1,D_size)
    x2 = np.random.uniform(-1,1,D_size)
    X = np.matrix([x0,x1,x2]).T
    ## h(x)的符号
    y = np.asarray(np.sign(np.matmul(X, w0))).flatten()

    ## 生成测试样本
    x0_test = np.full(test_size,1)
    x1_test = np.random.uniform(-1,1,test_size)
    x2_test = np.random.uniform(-1,1,test_size)
    X_test = np.matrix([x0_test,x1_test,x2_test]).T
    ## h(x)的符号
    y_test = np.asarray(np.sign(np.matmul(X_test, w0))).flatten()

    ## 目标函数的图像以及正确的y值
    # plt.style.use('default')
    # x_plt = np.linspace(-1, 1, 100)
    # y_plt = f(x_plt)
    # fig, ax = plt.subplots()
    # ax.plot(x_plt, y_plt, linestyle='dashed', linewidth=2.0, label='target')
    # ax.scatter(x1, x2, c = y)
    # ax.set(xlim=(-1, 1), 
    #        ylim=(-1, 1))
    # plt.show()

    ## 权重初始化
    w = np.array([0,0,0])
    y_plt_tmp = np.full(D_size,0)

    ## 开始循环更新权重
    for i in range(iteration):
        ## 计算当前权重下的y值
        y_hat = np.asarray(np.sign(np.matmul(X, w.T))).flatten()
        ## 根据分类错误的y取出对应的X，后续将从X中随机抽取一条用于更新权重
        y_tmp = y[y != y_hat]
        X_tmp = X[y != y_hat, :]
        if len(y_tmp) == 0:
            # print('===============================================')
            # print(f'提示: 已收敛，循环次数为{i}，退出循环')
            # print('===============================================')
            # 计算测试集上的误差
            y_hat_test = np.asarray(np.sign(np.matmul(X_test, w.T))).flatten()
            err_rate = sum(y_hat_test != y_test)/len(y_test)
            # 返回权重以及收敛次数
            return err_rate, i
            break
        rand_idx = np.random.choice(range(len(y_tmp)))
        # 取出用于更新权重的记录
        x_i = X_tmp[rand_idx]
        y_i = y_tmp[rand_idx]
        # 更新权重
        w = w + y_i*x_i

    # 作出最终函数的图像
    # color = np.full(D_size,0)
    # color[np.where(x1==x_i[0,1])] = 1
    # ax.scatter(x1, x2, c = color)
    # y_plt_tmp = -(w[0,1]*x1+w[0,0])/w[0,2]
    # ax.plot(x1, y_plt_tmp, linewidth=2.0, label='fitting')
    # plt.legend(loc="upper left")
    # plt.show()

    ## 若不收敛，收敛次数值返回-1
    y_hat_test = np.asarray(np.sign(np.matmul(X_test, w.T))).flatten()
    err_rate = np.sum(y_hat_test != y_test)/len(y_test)
    return err_rate, -1


if __name__ == "__main__":
    cnvg_i=[]
    err = []
    for n in tqdm(range(10000)):
        # 权重最高更新次数：10000，数据量：10
        err_rate,i = pla_expri(loop_n=10000, train_size=100, test_size=10000)
        if i != -1:
            cnvg_i.append(i)
        err.append(err_rate)
    print(f'平均收敛次数为：{np.mean(cnvg_i)}')
    print(f'平均误差为：{np.mean(err)}')