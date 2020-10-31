# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 19:54:06 2020

@author: DER
"""

from mxnet import autograd, nd 
x = nd.arange(4).reshape((4, 1))
print(x)

x.attach_grad()

with autograd.record():
    y = 2 * nd.dot(x.T, x)
    
y.backward()

assert (x.grad - 4 * x).norm().asscalar() == 0
x.grad

print("======================")

print(autograd.is_training())
with autograd.record():
    print(autograd.is_training())