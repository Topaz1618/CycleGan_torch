"""
    手写计算损失函数/pytorch 损失函数源码分析
    手写反向传播
"""


import torch
import torch.nn as nn
import numpy as np


class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out


input_dim = 1
output_dim = 1


model = LinearRegressionModel(input_dim, output_dim)
print(model)
print(model.parameters().__next__())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)


x_values = [i for i in range(11)]
x_train = np.array(x_values, dtype=np.float32)
x_train = x_train.reshape(-1, 1)
x_train.shape

# 根据设想方程构建实际 y 值.
y_values = [2*i + 1 for i in x_values]
y_train = np.array(y_values, dtype=np.float32)
y_train = y_train.reshape(-1, 1)
y_train.shape


# 指定参数和损失函数
epochs = 1000
rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=rate)
criterion = nn.MSELoss()


for epoch in range(epochs):
    inputs = torch.from_numpy(x_train).to(device)
    labels = torch.from_numpy(y_train).to(device)

    optimizer.zero_grad()

    # 输出一堆 y1
    outputs = model(inputs)
    print(outputs)

    # 计算损失函数 y1 & y 做计算
    loss = criterion(outputs, labels)

    # 反向传播
    loss.backward()

    # 更新权重参数
    optimizer.step()

    if epoch % 50 == 0:
        print(f"epoch: {epoch}, loss: {loss.item()}")

# 保存模型
# torch.save(model.state_dict(), 'model.pkl')
model.load_state_dict(torch.load('model.pkl'))


# 测试
# GPU
predicted = model(torch.from_numpy(x_train).to(device).requires_grad_())
predicted = predicted.to('cpu').data.numpy()
print(predicted)

# CPU
# predicted = model(torch.from_numpy(x_train).requires_grad_()).data.numpy()
# print(predicted)



"""

Todo
    - 基本使用 ✅
    - 看别的论文 M 部分咋写
    - gan: 
    - 自注意机制:
    - 开干
"""