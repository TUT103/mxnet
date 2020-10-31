# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 17:25:59 2020

@author: DER
"""

from mxnet import nd

x = nd.arange(12)

print(x.shape)

print(x.size) 

X = x.reshape((3, 4))

print(X)

print(nd.zeros((2, 3, 4)))

print(nd.ones((3, 4)))

Y = nd.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(Y)

print(nd.random.normal(0, 1, shape=(3, 4)))