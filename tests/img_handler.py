import pickle
import torch
import gzip
from matplotlib import pyplot
from pathlib import Path
import requests

DATA_PATH = Path("../data")
PATH = DATA_PATH / "mnist"
FILENAME = "mnist.pkl.gz"


def load_data():
    with gzip.open(PATH / FILENAME) as fp:
        data = pickle.load(fp, encoding="latin-1")
    return data


(x_train, y_train), (x_vaild, y_vaild), _ = load_data()

print(x_train[0])
print(x_train[0].shape)
# print(x_train.size())
# pyplot.imshow(x_train[0].reshape(28, 28), cmap="gray")

# output = torch.FloatTensor(x_train.size()[0], x_train.size()[1], 286, 286)
