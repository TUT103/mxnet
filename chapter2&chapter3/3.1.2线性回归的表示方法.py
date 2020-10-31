# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 10:44:54 2020

@author: DER
"""

from mxnet import nd
from time import time

a = nd.ones(shape=1000)
b = nd.ones(shape=1000)

# 方法一：将向量做标量加法
start = time()
print(start)
c = nd.zeros(shape=1000)
for i in range(1000):
    c[i] = a[i] + b[i]
print(time()-start)

# 方法二：将向量做矢量加法
start = time()
d = a + b
print(time()-start)
