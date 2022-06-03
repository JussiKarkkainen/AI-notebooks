import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch import nn

def get_data(batch_size):
    transform = [transforms.ToTensor()]
    transform = transforms.Compose(transform)

    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    return torch.utils.data.DataLoader(trainset, batch_size, shuffle=True, num_workers=2)

def get_train_iter():
    train_iter = get_data(batch_size=256)
    return train_iter


def relu(x):
    a = torch.zeros_like(x)
    return torch.max(x, a)



def softmax(X):
    X_exp = torch.exp(X)
    exp_sums = X_exp.sum(1, keepdim=True)
    return X_exp / exp_sums


def crossentropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])



def init_params():
    W1 = torch.normal(0, 0.01, (784, 256), requires_grad=True)
    b1 = torch.zeros(256, requires_grad=True)
    W2 = torch.normal(0, 0.01, (256, 256), requires_grad=True)
    b2 = torch.zeros(256, requires_grad=True)
    W3 = torch.normal(0, 0.01, (256, 10), requires_grad=True)
    b3 = torch.zeros(10, requires_grad=True)

    params = [W1, b1, W2, b2, W3, b3]
    return params

params = init_params()

def mlp(X):
    X = X.reshape(-1, 784)
    H1 = relu(X @ params[0] + params[1])
    H2 = relu(H1 @ params[2] + params[3])
    return softmax(relu(H2 @ params[4] + params[5]))


def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

def mlp_train_mnist():
    lr = 0.1
    num_epochs = 10
    loss = crossentropy
    net = mlp
    train_iter = get_train_iter()

    figure, ax = plt.subplots()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    x_axes = []
    y_axes = []
    for epoch in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y)
            l.sum().backward()
            sgd(params, lr, batch_size=256)
        with torch.no_grad():
            train_l = loss(net(X), y)
            print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
            ax.plot(epoch, float(train_l.mean()))
    plt.show()
            
