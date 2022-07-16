import requests
import pickle
import gzip
from pathlib import Path
from matplotlib import pyplot
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import torch.nn.functional as F

loss_func = F.cross_entropy


def download_dataset():
    PATH.mkdir(parents=True, exist_ok=True)
    content = requests.get(URL + FILENAME).content
    (PATH / FILENAME).open("wb").write(content)


def load_data():
    with gzip.open(PATH / FILENAME) as fp:
        data = pickle.load(fp, encoding="latin-1")

    return data


def show_image(x_train):
    pyplot.imshow(x_train[0].reshape(28, 28), cmap="gray")
    print(x_train.shape)


class Mnist_NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(784, 128)  # equal to => x.mm(weights) + bias
        self.hidden2 = nn.Linear(128, 256)  #
        self.out = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.out(x)
        return x


def show_net_pararmaters():
    for name, parameter in net.named_parameters():
        print(f"{name}, {parameter.size()}, {parameter} \n")


def get_data(train_ds, vaild_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(vaild_ds, batch_size=bs * 2)
    )


def data_loader(x_train, y_train, x_vaild, y_vaild):
    x_train, y_train, x_vaild, y_vaild = map(
        torch.tensor, (x_train, y_train, x_vaild, y_vaild)
    )

    n, c = x_train.shape
    train_ds = TensorDataset(x_train, y_train)
    vaild_ds = TensorDataset(x_vaild, y_vaild)

    return n, c, train_ds, vaild_ds


def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)


def get_model():
    """ 返回模型和优化器"""
    model = Mnist_NN()
    return model, optim.SGD(model.parameters(), lr=0.001)


def fit(steps, model, loss_func, opt, train_dl, vaild_dl):
    """
        opt: 优化器
    """
    for step in range(steps):
        model.train()  # 训练使用，调 normallization 和 dropout
        for xb, yb in train_dl:
            # loss_bacth 定义了反向传播
            loss_batch(model, loss_func, xb, yb, opt)

            # 测试步骤
        model.eval()  # 测试使用，不调 normallization 和 dropout

        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in vaild_dl]
            )
            val_loss = np.sum(np.multiply(losses, nums) / np.sum(nums))
            print(f"Current step: {step} loss: {val_loss}")


def main():
    train_dl, vaild_dl = get_data(train_ds, vaild_ds, bs)
    model, opt = get_model()
    fit(25, model, loss_func, opt, train_dl, vaild_dl)


if __name__ == "__main__":
    DATA_PATH = Path("../data")
    PATH = DATA_PATH / "mnist"
    # URL = "https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/data"
    FILENAME = "mnist.pkl.gz"

    # dateset handler
    # download_dataset()
    (x_train, y_train), (x_vaild, y_vaild), _ = load_data()
    print(x_train.shape)
    print(y_train.shape)
    print(x_vaild.shape)
    print(y_vaild.shape)
    # show_image(x_train)

    net = Mnist_NN()
    show_net_pararmaters()
    bs = 8
    n, c, train_ds, vaild_ds = data_loader(x_train, y_train, x_vaild, y_vaild)
    print(net)
    main()

