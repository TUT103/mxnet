# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 17:32:49 2020

@author: DER
"""

from mxnet import nd

x = nd.arange(12)
X = x.reshape((3, 4))
print(X)


Y = nd.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(Y)

print("X+Y")
print(X + Y)

print("X-Y")
print(X - Y)

print("X*Y")
print(X * Y)

print("X/Y")
print(X/Y)

print(Y.exp())

print(nd.dot(X, Y.T))

print("123")
print(nd.concat(X, Y, dim=0))

print(nd.concat(X, Y, dim=1))

print(X == Y)

print(X.sum())

print(X.norm().asscalar())






