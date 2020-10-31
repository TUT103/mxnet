# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 20:41:12 2020

@author: DER
"""

from mxnet import autograd, nd

print(autograd.is_training())
with autograd.record():
    print(autograd.is_training())