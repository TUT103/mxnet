# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 19:17:04 2020

@author: DER
"""
# 1
from mxnet import ndarray as nd

x = nd.arange(12)
print(x)

print(x.shape)

print(x.size)

X = x.reshape((3,4))
print(X)

print(nd.zeros((2, 3, 4)))

print(nd.ones((3, 4)),end="\n\n")

Y = nd.array([[12, 11, 10, 9], [8, 7, 6, 5], [4, 3, 2, 1]])
print(Y)

print(nd.random.normal(0, 1, shape=(3, 4)))

print(Y.exp())

print(nd.dot(X, Y.T))

print(nd.concat(X, Y, dim=0))
print(nd.concat(X, Y, dim=1))

print(X==Y)

print(X.sum())

print(X.norm())