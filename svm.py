# implement support vector classifier
# Modified from https://github.com/kazuto1011/svm-pytorch/blob/master/main.py (Kazuto Nakashima) MIT License

import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_blobs, load_breast_cancer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# define a linear layer
def create_model(feat_dim):
    # 创建了一个线性层，参数是 (2, 1):这个线性层将输入维度为 2（通常是一个包含两个特征的输入向量）映射到输出维度为 1（通常是一个标量输出）
    # m = nn.Linear(2,1) # one layer, 2 dimension data
    m= nn.Linear(feat_dim,1) # one layer, mutiple dimension data
    m.to(args.device) # parameter if have a GPU or not, optimize to GPU using pytorch
    return m # return a model, function

# create training data
def create_data():
    # # 随机生成 100 个数据点，每个数据点包含 2 个特征，这些数据点被分成两个类别
    # X, Y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=2) # 2 dimensions
    # Y[Y == 0] = -1 # label 0 to be -1, 1
    # X = torch.tensor(X, dtype=torch.float32) # tensor like numpy array, convert numpy X into tensor
    # Y = torch.tensor(Y, dtype=torch.float32)
    # return X, Y
    # 1. load data and seprate data into train and test subset
    data = load_breast_cancer()
    X = data.data
    Y = data.target
    # scale the data
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    # print(y)
    Y = np.array([1 if i == 1 else -1 for i in Y])
    X = torch.tensor(X, dtype=torch.float32) # tensor like numpy array, convert numpy X into tensor
    Y = torch.tensor(Y, dtype=torch.float32)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    return X_train, X_test, Y_train, Y_test

def create_dataset(X, Y):
    class dataset(Dataset): # define a clear dataset， inherit from Dataset，Dataset 是 PyTorch 的一个抽象基类（Abstract Base Class），用于定义自定义数据集的接口
        def __init__(self, X, Y):
            self.X = X
            self.Y = Y
        def __len__(self):
            return len(self.Y) # return length of Y
        def __getitem__(self, idx):
            return self.X[idx], self.Y[idx] # 根据索引 idx 获取数据集中的一个样本
    trainset = dataset(X,Y)
    return trainset # x and y label

def train(X, Y, model, args):
    optimizer = optim.SGD(model.parameters(), lr=args.lr) # choose SGD as optimizer, learning rate
    trainset = create_dataset(X, Y) # create dataset
    # batchsize are you going to pick batchsize in the data with shuffle?
    trainloader = DataLoader(trainset, batch_size=args.batchsize, shuffle=True)
    N = len(trainset)
    model.train()
    for epoch in range(args.epoch):
        sum_loss = 0.0
        for t, (x, y) in enumerate(trainloader):
            x_g = x.to(args.device)
            y_g = y.to(args.device)
            output = model(x_g).squeeze()
            weight = model.weight.squeeze() # w
            # print(output.size())
            # print(y_g)
            # loss = torch.mean(torch.clamp(1 - y_g * output, min=0)) # loss function: hing loss
            # for loss function: l = 1 - 2 y * output, if y * output < 0; l = 1 - y * output, if 1 >= y * output >= 0; l = 0, if y * output > 1
            # ERM: get the average value of loss function over all data points
            loss = torch.mean(torch.where(y_g * output < 0, 1 - 2 * y_g * output, torch.where((y_g * output >= 0) & (y_g * output <= 1), 1 - y_g * output, torch.tensor(0.0))))

            loss += args.c * (weight.t() @ weight) / 2.0  # overall loss, 这是一个正则化项，用于防止过拟合，惩罚权重过大的模型

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  # each parameter will updated according to the loss

            sum_loss += float(loss)

        print("Epoch: {:4d}\tloss: {}".format(epoch, sum_loss / N))

def calculateAccuracy(X, Y, model, args):
    # 1. calculate h(x)
    model.eval()
    output = model(X).squeeze() # h(x)
    y_h = output * Y

    # 2. calculate test error with the loss function as follows:
    # l = 1, if yh(x) < -rho; l = 1/3, if |yh(x)| < rho; l = 0, if yh(x) > rho
    error = torch.sum(torch.where(y_h < -args.rho, 1.0, torch.where(torch.abs(y_h) < args.rho, 1.0/3.0, 0.0))) / len(Y)
    return error

def visualize(X, Y, model):
    W = model.weight.squeeze().detach().cpu().numpy()
    b = model.bias.squeeze().detach().cpu().numpy()

    delta = 0.001
    x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 200)
    x2 = -b / W[1] - W[0] / W[1] * x1
    x2_p = (1-b) / W[1] - W[0] / W[1] * x1 
    x2_m = (-1-b) / W[1] - W[0] / W[1] * x1
    
    x1_data = X[:, 0].detach().cpu().numpy()
    x2_data = X[:, 1].detach().cpu().numpy()
    y_data = Y.detach().cpu().numpy()

    x1_c1 = x1_data[np.where(y_data == 1)]
    x1_c2 = x1_data[np.where(y_data == -1)]
    x2_c1 = x2_data[np.where(y_data == 1)]
    x2_c2 = x2_data[np.where(y_data == -1)]

    plt.figure(figsize=(10, 10))
    plt.xlim([x1_data.min() + delta, x1_data.max() - delta])
    plt.ylim([x2_data.min() + delta, x2_data.max() - delta])
    plt.plot(x1_c1, x2_c1, "P", c="b", ms=10)
    plt.plot(x1_c2, x2_c2, "o", c="r", ms=10)
    plt.plot(x1, x2, "k", lw=3)
    plt.plot(x1, x2_p, "b--", lw=3)
    plt.plot(x1, x2_m, "r--", lw=3)
    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", type=float, default=0.01) # hyperparameter come from loss function
    parser.add_argument("--lr", type=float, default=0.01) # algorithm
    parser.add_argument("--batchsize", type=int, default=5)
    parser.add_argument("--epoch", type=int, default=20) #
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"]) # if have GPU, use GPU.
    parser.add_argument("--rho", type=float, default=0.1)  # rho
    args = parser.parse_args()
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(args)

    X_train, X_test, Y_train, Y_test = create_data()
    feat_dim = X_train.shape[1]
    # print(Y_train)

    model = create_model(feat_dim) # do your own model

    train(X_train, Y_train, model, args) # EMR: empirical risk minimizer, args: contain hyperparameter. Weight, bias and parameter
    error = calculateAccuracy(X_test, Y_test, model, args)
    print("Test error: {}".format(error))
    # visualize(X_train, Y_train, model)
