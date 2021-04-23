"""
代码7-3
神经网络解决分类问题示例
"""
import torch                                # 引入torch模块
import torch.nn as nn                       # 简化nn模块
import matplotlib.pyplot as plt             # 引入并简化matplotlib模块

data = torch.ones(100, 2)                   # 数据总数（总框架）
x0 = torch.normal(2*data, 1)                # 第一类坐标，从满足mean为2和std为1的正态分布中抽取随机数
y0 = torch.zeros(100)                       # 第一类标签设置为0
x1 = torch.normal(-2*data, 1)                # 第二类坐标，从满足mean为-2和std为1的正态分布中抽取随机数
y1 = torch.ones(100)                       # 第二类标签设置为1
x = torch.cat((x0, x1)).type(torch.FloatTensor)      # x0,x1合并成x，并转换成float类型的tensor变量
y = torch.cat((y0, y1)).type(torch.LongTensor)          # y0,y1合并成y，并转换成long类型的tensor变量


class Net(nn.Module):                       # 定义类存储网络结构
    def __init__(self):
        super(Net, self).__init__()
        self.classify = nn.Sequential(       # nn模块搭建网络
            nn.Linear(2, 15),               # 全连接层，1个输入10个输出
            nn.ReLU(),                      # ReLU激活函数
            nn.Linear(15, 2),               # 全连接层，10个输入1个输出
            nn.Softmax(dim=1)
        )

    def forward(self, x):                   # 定义前向传播过程
        classification = self.classify(x)        # 将x传入网络
        return classification                   # 返回预测值


net = Net()
optimizer = torch.optim.SGD(net.parameters(), lr=0.03)      # 设置优化器
loss_func = nn.CrossEntropyLoss()                           # 设置损失函数
plt.ion()                                                   # 打开交互模式
for epoch in range(100):                                   # 训练部分
    out = net(x)                                     # 实际输出
    loss = loss_func(out, y)                         # 实际输出和期望输出传入损失函数
    optimizer.zero_grad()                                   # 清除梯度
    loss.backward()                                         # 误差反向传播
    optimizer.step()                                        # 优化器开始优化
    if epoch % 2 == 0:                                      # 每2epoch显示
        plt.cla()                                           # 清除上一次绘图
        classification = torch.max(out, 1)[1]               # 返回每一行中最大值的下标
        class_y = classification.data.numpy()               # 转换成numpy数组
        target_y = y.data.numpy()                           # 标签页转换成numpy数组，方便后面计算准确率
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=class_y, s=100, cmap='RdYlGn')  # 绘制散点图
        accuracy = sum(class_y == target_y)/200.            # 计算准确率
        plt.text(1.5, -4, f'Accuracy={accuracy}', fontdict={'size': 20, 'color':  'red'})  # 显示准确率
        plt.pause(0.4)                     # 时间0.4s
    plt.show()
plt.ioff()                                                # 关闭交互模式
plt.show()                                                # 定格显示最后结果

