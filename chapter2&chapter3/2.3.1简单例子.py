# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 20:21:32 2020

@author: DER
"""

from mxnet import autograd, nd

x = nd.arange(4).reshape((4, 1))
print("求导前", end = "：")
print(x)

# 步骤1：开辟内存
x.attach_grad()

# 步骤2：定义求导的计算公式
with autograd.record():
    y = 2 * nd.dot(x.T, x)
   
# 步骤3：自动求梯度
y.backward()

# assert (x.grad - 6 * x).norm().asscalar() == 0
print("求导后", end = "：")
print(x.grad)