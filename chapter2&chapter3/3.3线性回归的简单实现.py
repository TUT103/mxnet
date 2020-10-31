# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 14:38:35 2020

@author: DER
"""

""" 3.3.1生成数据集 """
from mxnet import autograd, nd

num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)

""" 3.3.2读取数据集 """
from mxnet.gluon import data as gdata

batch_size = 10
dataset = gdata.ArrayDataset(features, labels)
data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)
"""
for X, y in data_iter:
    print(X, y)
"""

""" 3.3.3定义模型 """
from mxnet.gluon import nn

net = nn.Sequential()

net.add(nn.Dense(1))

""" 3.3.4初始化模型函数 """
from mxnet import init

net.initialize(init.Normal(sigma=0.01))

""" 3.3.5定义损失函数 """
from mxnet.gluon import loss as gloss

loss = gloss.L2Loss()

""" 3.3.6定义优化算法 """
from mxnet import gluon

trainer = gluon.Trainer(net.collect_params(), "sgd", {"learning_rate": 0.03})

"""3.3.7训练模型 """
num_epochs = 10
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        with autograd.record():
            l = loss(net(X), y)
        l.backward()
        trainer.step(batch_size)
    l = loss(net(features), labels)
    print("epoch %d, loss:%f" %(epoch, l.mean().asnumpy()))
        















