# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 10:53:44 2020

@author: DER
"""

from IPython import display
from matplotlib import pyplot as plt
from mxnet import autograd, nd
import random

""" 3.2.1生成数据集"""
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
"""

"""

"""
print(features)
print(features[:,0]) #全部行第0列的数据
print(features[:,1])
print(features[:,0].size)
"""
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)

print(features[0], labels[0])


def use_svg_display():
    # 用矢量图表示
    display.set_matplotlib_formats("svg")


def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    # 设置图片的尺寸
    plt.rcParams["figure.figsize"] = figsize


set_figsize()
plt.scatter(features[:, 1].asnumpy(), labels.asnumpy(), 1)
plt.show()

""" 3.2.2读取数据集 """


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = nd.array(indices[i: min(i + batch_size, num_examples)])
        yield features.take(j), labels.take(j)


batch_size = 10
""" 3.2.3初始化模型参数 """
w = nd.random.normal(scale=0.01, shape=(num_inputs, 1))
b = nd.zeros(shape=(1,))
w.attach_grad()
b.attach_grad()

""" 3.2.4 定义模型 """


def linreg(X, w, b):
    return nd.dot(X, w) + b


""" 3.2.5 定义损失函数 """


def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


""" 3.2.6 定义优化算法 """


def sgd(params, lr, batch_size):
    for param in params:
        param[:] = param - lr * param.grad / batch_size


""" 3.2.7 训练模型 """
lr = 0.03
num_epochs = 10
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        with autograd.record():
            l = loss(net(X, w, b), y)
        l.backward()
        sgd([w, b], lr, batch_size)
    train_l = loss(net(features, w, b), labels)
    print("epoch %d, loss %f, " % (epoch + 1, train_l.mean().asnumpy()))
    # print("w", w, "b", b)
