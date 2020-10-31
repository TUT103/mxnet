# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 18:05:32 2020

@author: DER
"""

from mxnet import nd

x = nd.arange(12)
X = x.reshape((3, 4))



Y = nd.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])


before = id(Y)
print(Y)
Y = Y + X
print(id(Y)==before)



