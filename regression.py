"""
代码7-2
神经网络解决回归问题示例
"""
import numpy as np                          # 引入numpy模块
import torch                                # 引入torch模块
import torch.nn as nn                       # 简化nn模块
import matplotlib.pyplot as plt             # 引入并简化matplotlib模块

x = torch.unsqueeze(torch.linspace(- np.pi, np.pi, 100), dim=1)     # 构建等差数列
y = torch.sin(x) + 0.5 * torch.rand(x.size())                       # 添加随机数


class Net(nn.Module):                       # 定义类存储网络结构
    def __init__(self):
        super(Net, self).__init__()
        self.predict = nn.Sequential(       # nn模块搭建网络
            nn.Linear(1, 10),               # 全连接层，1个输入10个输出
            nn.ReLU(),                      # ReLU激活函数
            nn.Linear(10, 1)                # 全连接层，10个输入1个输出
        )

    def forward(self, x):                   # 定义前向传播过程
        prediction = self.predict(x)        # 将x传入网络
        return prediction                   # 返回预测值


net = Net()
optimizer = torch.optim.SGD(net.parameters(), lr=0.05)      # 设置优化器
loss_func = nn.MSELoss()                                    # 设置损失函数
plt.ion()                                                   # 打开交互模式
for epoch in range(1000):                                   # 训练部分
    out = net(x)                                     # 实际输出
    loss = loss_func(out, y)                         # 实际输出和期望输出传入损失函数
    optimizer.zero_grad()                                   # 清除梯度
    loss.backward()                                         # 误差反向传播
    optimizer.step()                                        # 优化器开始优化
    if epoch % 25 == 0:                                   # 每25epoch显示
        plt.cla()                                         # 清除上一次绘图
        plt.scatter(x, y)                                 # 绘制散点图
        plt.plot(x, out.data.numpy(), 'r', lw=5)   # 绘制曲线图
        plt.text(0.5, 0, f'loss={loss}', fontdict={'size': 20, 'color': 'red'}) # 添加文字来显示loss值
        plt.pause(0.1)                                    # 显示时间0.1s
    plt.show()
plt.ioff()                                                # 关闭交互模式
plt.show()                                                # 定格显示最后结果

