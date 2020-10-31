# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 20:52:50 2020

@author: DER
"""

""" 3.16.2数据集 """
import d2lzh as d2l
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
import numpy as np
import pandas as pd

train_data = pd.read_csv(".../data/kaggle_house_pred_train.csv")
test_data = pd.read_csv(".../data/kaggle_house_pred_test.csv")

print(train_data.shape)
print(test_data.shape)

print(train_data.iloc([0: 4, [0, 1, 2, 3, -3, -2, -1]]))
"""  """
