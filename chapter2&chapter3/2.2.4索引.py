# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 17:53:48 2020

@author: DER
"""

from mxnet import nd

x = nd.arange(12)
X = x.reshape((3, 4))
print(X)


Y = nd.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(Y)

print(X[1:3])

print(X[1, 2])

X[1:2, :]=12
print(X)