# 小批量随机梯度下降
import random  # 随机化

import matplotlib.pyplot as plt
import torch


# 生成数据集


def synthetic_data(w, b, num_examples):  # 系数与样本
    """生成y=xw+b噪声样本"""
    X = torch.normal(0, 1, (num_examples, len(w)))  # 1000*2
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


# w b真是值


true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
# 注意，[features中的每一行都包含一个二维数据样本， labels中的每一行都包含一维标签值（一个标量）]。
# 通过生成第二个特征features[:, 1]和labels的散点图， 可以直观观察到两者之间的线性关系。
plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)
plt.show()


# %% 读取数据集
# 定义生成器  作用：分批  gpu并行
def data_iter(batch_size, features, labels):  # batch_size 批量大小
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # 洗牌
    for i in range(0, num_examples, batch_size):
        # 截取区间
        batch_indices = torch.tensor(indices[i:  min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


# %% 初始化模型参数
# 在下面的代码中，我们通过从均值为0、标准差为0.01的正态分布中采样随机数来初始化权重， 并将偏置初始化为0。
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)


# %% 定义模型
def linreg(X, w, b):
    """线性回归模型"""
    return torch.matmul(X, w) + b


# 定义损失函数 均方损失
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


# 定义梯度下降sgd (优化算法
def sgd(params, lr, batch_size):
    """[w,b]  学习率  批量大小 """
    with torch.no_grad():
        # 下面不会改变梯度
        for param in params:
            param -= lr * param.grad / batch_size  # 梯度下降
            param.grad.zero_()  # 梯度清0,防止累加


# %%训练
# 初始化超参数
lr = 0.03
num_epochs = 3  # 迭代次数
net = linreg  # 选择模型方法
loss = squared_loss
batch_size = 10
#
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):  # 取一批 x，y
        l = loss(net(X, w, b), y)  # X和y的小批量损失
        # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
        # 并以此计算关于[w,b]的梯度
        l.sum().backward()  # (向量)求和，求梯度（反向求导
        sgd([w, b], lr, batch_size)  # 梯度下降
    with torch.no_grad():  # 不更新梯度
        train_l = loss(net(features, w, b), labels)  # 每次计算当前损失（减小
        print("epoch{} , loss {}".format(epoch + 1, float(train_l.mean())))
        # 显示

print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')
