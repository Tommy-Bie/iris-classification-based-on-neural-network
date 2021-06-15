"""
基于全连接神经网络的鸢尾花（iris）数据分类
测试精度: 100%
Author: Tommy Bie
"""
import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


# 读取iris数据集
iris = load_iris()
iris_data = pd.DataFrame(iris['data'], columns = ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width'])
iris_data['Species'] = iris.target

# 数据和标签
data = iris_data.iloc[:, :-1]
label = iris_data.iloc[:, -1]

# 划分数据集
train_data, test_data, train_label, test_label = train_test_split(data, label, test_size=0.2, random_state=0)

# 转成tensor
train_data = np.array(train_data, dtype=float)
test_data = np.array(test_data, dtype=float)
train_label = np.array(train_label)
test_label = np.array(test_label)
train_data = torch.from_numpy(train_data)
test_data = torch.from_numpy(test_data)
train_label = torch.from_numpy(train_label)
test_label = torch.from_numpy(test_label)

# 网络结构
class my_network(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(my_network, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

Learning_rate = 0.001  # 学习率
num_epoch = 50  # 迭代伦次


if __name__ == '__main__':
    train_dataset = TensorDataset(train_data, train_label)
    test_dataset = TensorDataset(test_data, test_label)

    train_DataLoader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_DataLoader = DataLoader(test_dataset, batch_size=1)

    net = my_network(4, 10, 3)
    optimizer = torch.optim.SGD(net.parameters(), lr=Learning_rate, momentum=0.9)  # 带动量的随机梯度下降
    loss_func = torch.nn.CrossEntropyLoss()  # 交叉熵损失

    train_losses = []  # 训练损失
    train_acces = []  # 训练正确率
    test_losses = []  # 测试损失
    test_acces = []  # 测试正确率

    loss_list = [] # loss趋势记录

    for epoch in range(num_epoch):
        train_loss = 0
        train_acc = 0
        test_loss = 0
        test_acc = 0

        for data, label in train_DataLoader:  # 训练
            optimizer.zero_grad()  # 梯度清零
            data = data.float()
            label = label.long()
            out = net(data)  # 前向传播预测值
            loss = loss_func(out, label)  # 计算损失
            print("train_loss:", loss)
            loss.backward()  # 损失反向传播
            optimizer.step()

            train_loss += loss.item()

            num_correct = (torch.argmax(out) == label).sum().item()  # 本次分类是否正确，用于计算正确率
            train_acc += num_correct

        for data, label in test_DataLoader:  # 测试
            data = data.float()
            label = label.long()
            pred = net(data)

            eval_loss = loss_func(pred, label)
            print("test loss:", eval_loss)  # 每个样本的测试损失


            test_loss += eval_loss.sum().item()
            num_correct = (torch.argmax(pred) == label).sum().item()  # 是否正确
            test_acc += num_correct

        train_losses.append(train_loss / len(train_DataLoader))
        train_acces.append(train_acc / len(train_DataLoader))
        test_losses.append(test_loss / len(test_DataLoader))
        test_acces.append(test_acc / len(test_DataLoader))

        loss_list.append(train_loss)  # 记录loss

        print('Epoch {} \nTrain Loss {} Train Accuracy {} \nTest Loss {} Test Accuracy {}'.format(epoch + 1,
                        train_loss / len(train_DataLoader), train_acc / len(train_DataLoader), test_loss / len(test_DataLoader), test_acc / len(test_DataLoader)))

    # 损失下降图
    x = np.arange(1, 51)
    plt.plot(x, loss_list)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()



