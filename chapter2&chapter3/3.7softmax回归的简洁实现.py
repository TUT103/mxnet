# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 10:13:24 2020

@author: DER
"""

import d2lzh as d2l
from mxnet import gluon, init
from mxnet.gluon import loss as gloss, nn

""" 3.7.1获取和读取数据 """
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

""" 3.7.2定义和初始化模型 """
net = nn.Sequential()
net.add(nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))

""" 3.7.3softmax和交叉熵损失函数 """
loss = gloss.SoftmaxCrossEntropyLoss()

""" 3.7.4定义优化算法 """
trainer = gluon.Trainer(net.collect_params(), "sgd", {"learning_rate":0.1})

"""3.7.5训练模型 """
num_epochs = 5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None,
              None, trainer)






