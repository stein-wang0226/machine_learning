import torch
from d2l import torch as d2l
from torch.utils import data

# %% 生成数据集

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)


def load_array(data_arrays, batch_size, is_train=True):
    """"构造pytorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


batch_size = 10
data_iter = load_array((features, labels), batch_size)
# 使用data_iter的方式与我们在 从0实现中使用data_iter函数的方式相同

# %%定义模型
from torch import nn

"""neural network"""
net = nn.Sequential(nn.Linear(2, 1))
'''
seq..:类，将多层连接的容器 我们的模型只包含一个层，
因此实际上不需要Sequential。 但是由于以后几乎所有的模型都是多层的，
在这里使用Sequential会让你熟悉“标准的流水线”。
(2,1)为 输入输出的维度   
'''

# %%初始化模型参数(w,b)
"""
我们通过net[0]选择网络中的第一个图层， 然后使用weight.data和bias.data方法访问参数w,b。
我们还可以使用替换方法normal_和fill_来重写参数值。
"""
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

# 定义损失函数  ---均方差
loss = nn.MSELoss()

# 定义优化算法 --随机小批量梯度下降
# 实例化 sgd对象
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

# %%训练
# 类似之前
"""
通过调用net(X)生成预测并计算损失l（前向传播）。
通过进行反向传播来计算梯度。
通过调用优化器来更新模型参数。
"""
num_epochs = 3  # 迭代次数
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()  # 清0
        l.backward()  # 自动求和过
        trainer.step()  # 执行sgd 梯度下降
    # 观察每次的loss
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)
