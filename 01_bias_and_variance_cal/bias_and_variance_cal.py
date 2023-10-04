# ====================================================
# 示例：计算假设集的偏差与方差
# 
# 真实目标函数：f(x) = sin(pi*x)
# 样本生成：x随机均匀从[-1,1]抽取
# 样本量：N = 2
# 假设：
#   1. h(x) = b
#   2. h(x) = ax + b
# 
# 求假设1与假设2的偏差与方差
# =====================================================

import numpy as np
import time
from tqdm import tqdm
from math import pi


# 对假设2来说，数据集的最优函数为：g(x) = (y1-y2)/(x1-x2)*(x-x2) + y2
# 将闭包作为函数工厂来使用，用于保存不同数据集得到的最优函数g
# 注意，不能在list comprehension中使用lambda函数来生成，由于late binding的特性
# ，只在调用时才开始对x_1,x_2,y_1,y_2求值，这样只会使用最后一个循环中的变量值
def _make_h2_g(x1,x2,y1,y2):
    def f(x):
        return (y1-y2)/(x1-x2)*(x-x2) + y2
    return f

def bias_and_variance_cal(D_size = 10000, seed = 42):
    # 数据集D的个数设为D_size
    # 设置随机数种子
    D_size = D_size
    seed = seed
    np.random.seed(seed)
    X1 = np.random.uniform(-1,1,D_size)
    y1 = np.sin(pi*X1)
    X2 = np.random.uniform(-1,1,D_size)
    y2 = np.sin(pi*X2)
    # 对假设1来说，对每个数据集的最优函数为: g(x) = (y1+y2)/2
    # 对应的平均函数为常数函数
    # 由于所有的g都为常数函数，故不同g在X上应用后得到的数据集都是一样的
    h1_g = (y1+y2)/2
    h1_g_X1 = (y1+y2)/2
    h1_g_X2 = (y1+y2)/2
    h1_g_bar = np.mean(h1_g)
    # 偏差为：0.496367324518468
    h1_bias = np.mean((h1_g_bar - np.concatenate([y1,y2]))**2)
    # 方差为： 0.24685615937109565
    # 由于g是一组常数函数，因此对不同x得到的数据集{g1(x1), g2(x1), g3(x1), ..., gn(x1)}与
    # {g1(x2), g2(x2), g3(x2), ..., gn(x2)}是相同的，因此只需要求g在一个x上的方差即可
    h1_var = np.mean((h1_g_X1 - h1_g_bar)**2)
    # 假设2
    # 利用_make_h2_g得到不同数据集下的g并保存到list中
    h2_g = [_make_h2_g(x_1,x_2,y_1,y_2) for x_1,x_2,y_1,y_2 in zip(X1,X2,y1,y2)]
    # 注意：这里的h2_g_X1, h2_g_X2是不同数据集下最优函数的拟合值
    h2_g_X1 = y1
    h2_g_X2 = y2
    # 计算各个数据集上最优函数g在X1、X2处的值、保存平均函数在X1、X2处的值、特定x下不同g(x)的方差
    # 在计算方差时，根据公式的描述，运算的第一步应该是把g当作变量，固定x，因此首先算的
    # 应该是不同的g在同一x上的取值与平均值之差的平方和（即数学上方差的概念）
    h2_g_bar_X1 = []
    h2_g_bar_X2 = []
    h2_g_D_var_X1 = []
    h2_g_D_var_X2 = []
    for x_1, x_2 in tqdm(zip(X1,X2)):
        # 对当前的x_1, x_2, 求不同数据集下最优函数g在x_1,x_2上的取值，并求平均值
        h2_g_D_X1 = np.array([g(x_1) for g in h2_g])
        h2_g_D_X2 = np.array([g(x_2) for g in h2_g])
        h2_g_bar_x1 = np.mean(h2_g_D_X1)
        h2_g_bar_x2 = np.mean(h2_g_D_X2)
        # 计算指定x下不同g(x)间的方差
        h2_g_D_var_x1 = np.mean((h2_g_D_X1 - h2_g_bar_x1)**2)
        h2_g_D_var_x2 = np.mean((h2_g_D_X2 - h2_g_bar_x2)**2)
        # 保存当前x下的均值与方差的结果
        h2_g_bar_X1.append(h2_g_bar_x1)
        h2_g_bar_X2.append(h2_g_bar_x2)
        h2_g_D_var_X1.append(h2_g_D_var_x1)
        h2_g_D_var_X2.append(h2_g_D_var_x2)
    
    # 偏差为：0.20439822202437427
    h2_bias = np.mean(np.concatenate([(h2_g_bar_X1 - y1)**2, (h2_g_bar_X2 - y2)**2]))
    # 方差为：1.664718723308858
    h2_var = np.mean(np.concatenate([h2_g_D_var_X1, h2_g_D_var_X2]))
    print("===============================================")
    print(f"数据集个数为 {D_size} 时，结果如下：")
    print(f"假设1的偏差为：{round(h1_bias,4)}, 方差为：{round(h1_var,4)}")
    print(f"假设2的偏差为：{round(h2_bias,4)}, 方差为：{round(h2_var,4)}")
    print("===============================================")
    return h1_bias, h1_var, h2_bias, h2_var

if __name__ == '__main__':
    bias_and_variance_cal()
