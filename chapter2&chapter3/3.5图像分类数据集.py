# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 15:44:41 2020

@author: DER
"""

""" 3.5.1获取数据集 """
import d2lzh as d2l
from mxnet.gluon import data as gdata
import sys
import data

mnist_train = gdata.vision.FashionMNIST(train=True)# 训练集
mnist_test = gdata.vision.FashionMNIST(train=False)# 测试集

print("训练集长度 %d，测试集长度 %d" %(len(mnist_train), len(mnist_test)))

feature, label = mnist_train[0]

print("feature:", feature.shape, feature.dtype)
print("label:", label, type(label), label.dtype)

def get_fashion_mnist_labels(labels):
    text_labels = ["t-shirt", "trouser", "pullover", "dress", "coat", 
                   "sandal", "shirt", "sneaker", "bag", "ankle boot"]
    return [text_labels[int(i)] for i in labels]

def show_fashion_mnist(images, labels):
    d2l.use_svg_display()
    # 这里的_表示我们忽略的变量
    _, figs = d2l.plt.subplots(1, len(images), figsize=(12, 12))
    for f,img, lbl, in zip(figs, images, labels):
        f.imshow(img.reshape((28, 28)).asnumpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)

# 看一下训练集中前9个样本       
X, y = mnist_train[10:19]
show_fashion_mnist(X, get_fashion_mnist_labels(y))

""" 3.5.2读取小批量 """
batch_size = 256
transformer = gdata.vision.transforms.ToTensor()
if sys.platform.startswith("win"):
    num_workers = 0
else:
    num_workers = 4
    
train_iter = gdata.DataLoader(mnist_train.transform_first(transformer),
                              batch_size, shuffle=True,
                              num_workers=num_workers)
train_iter = gdata.DataLoader(mnist_test.transform_first(transformer),
                              batch_size, shuffle=False,
                              num_workers=num_workers)

# 读取一遍训练数据所需要的的时间
from time import time
start = time()
for X, y in train_iter:
    continue
print("读取一遍训练数据所需要的的时间：%0.2f sec" %(time() - start))










