# 模拟1000枚硬币的抛掷过程
# 过程如下：
# 1.每枚硬币独立抛10次
# 2.关注如下硬币正面出现的频率
#   2.1 最先开始抛的硬币(v1)
#   2.2 随机选取一枚硬币(v_rand)
#   2.3 正面出现最少的硬币(v_min)

# 问题1：所关注的3枚硬币的u值是多少
# 问题2：重复实验100000次，画出v1, v_rand, v_min的分布
import matplotlib.pyplot as plt
import numpy as np

v1 = []
v_rand = []
v_min = []

for i in range(100000):
    expri_mat = np.random.choice([0,1], size=[1000, 10])
    # 统计每个硬币的正面出现的频率
    vn = expri_mat.mean(axis=1)
    v1 = np.append(v1, vn[0])
    v_rand = np.append(v_rand, np.random.choice(vn))
    v_min = np.append(v_min, vn.min())

# v1与v_rand图像基本重合，v_min集中在0与0.1附近
plt.hist(v1,bins=20)
plt.show()
plt.hist(v_rand,bins=20)
plt.show()
plt.hist(v_min,bins=20)
plt.show()

# 问题3：根据问题2的结果，画出P[|v-u| > ε] 与 霍夫丁边界 关于 ε 的图像
epsilon = np.linspace(0, 0.5, 100)
bd = 2*np.e**(-2*epsilon**2*10)

## |v1-u1|, u1=0.5
p1 = [np.mean(abs(v1-0.5) > i) for i in epsilon]
## |v_rand-u2|, u2=0.5
p2 = [np.mean(abs(v_rand-0.5) > i) for i in epsilon]
## |v_rand-u2|, u2=0.5
p3 = [np.mean(abs(v_min-0.5) > i) for i in epsilon]

plt.plot(epsilon, bd)
plt.plot(epsilon, p1, linestyle='-.')
# plt.show()
# plt.plot(epsilon, bd)
plt.plot(epsilon, p2, alpha=0.5, c='black')
# plt.show()
# plt.plot(epsilon, bd)
plt.plot(epsilon, p3)
plt.show()

# 问题4：v1, v_rand, v_min中哪些满足霍夫丁不等式，解释原因
# 答：v1与v_rand满足霍夫丁不等式，v_min不满足，从概率边界的图像可以看出，P[|v1-u| > ε] 与 P[|v_rand-u| > ε]
# 的图像均在霍夫丁边界之下，但是 P[|v_min-u| > ε] 有部分在霍夫丁边界之外

# 问题5：结合 multiple bins 时霍夫丁不等式的情形，解释问题4的结果
# 答：本次实验的隐含假设，即假设集在全集上的误差为0.5（一枚硬币代表一个假设）
# 本题中的v可以理解为假设集在训练集的误差，u可以理解为假设集在全集上的误差，
# 则v1代表假设集中仅包含一个假设的情况，故而 P[|v1-u| > ε] 满足霍夫丁不等式
# v_rand等价于v1，均是从全假设集中随机选取一个假设（当然也可以看作多假设情形，必然满足）
# v_min则涉及1000个假设集的最优选择（误差v最小），则此时假设个数为1000，不满足单条件的霍夫丁不等式