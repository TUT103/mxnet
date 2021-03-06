""" 5.4.1二维最大池化层和平均池化层 """
from mxnet import nd
from mxnet.gluon import nn


def pool2d(X, pool_size, mode="max"):
    p_h, p_w = pool_size
    Y = nd.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[1]):
        for j in range(Y.shape[1]):
            if mode == "max":
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == "avg":
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y


X = nd.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
print(X)
print(pool2d(X, (2, 2)))

print(pool2d(X, (2, 2), "avg"))

print(pool2d(X, (2, 2), "max"))

""" 5.4.2填充和步幅 """
X = nd.arange(16).reshape((1, 1, 4, 4))
print(X)

pool2d = nn.MaxPool2D(3)
print(pool2d)
print(pool2d(X))

pool2d = nn.MaxPool2D(3, padding=1, strides=2)
print(pool2d(X))

pool2d = nn.MaxPool2D((2, 3), padding=(1, 2), strides=(2, 3))
print(pool2d(X))

""" 5.4.3多通道 """
X = nd.concat(X, X + 1, dim=1)
print(X)

pool2d = nn.MaxPool2D(3, padding=1, strides=2)
print(pool2d(X))
















