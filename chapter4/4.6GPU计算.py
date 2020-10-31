""" 4.6.1计算设备 """
import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn

print(mx.cpu(), mx.gpu())

""" 4.6.2NDArray的GPU运算 """
x = nd.array([1, 2, 3])
print(x)

print(x.context)

a = nd.array([1, 2, 3], ctx=mx.gpu())
print(a)

"""GPU上的存储"""
a = nd.array([1, 2, 3], ctx=mx.gpu())
print(a)

y = x.copyto(mx.gpu())
print(y)

z = x.as_in_context(mx.gpu())
print(z)

print(y.as_in_context(mx.gpu()) is y)
print(y.copyto(mx.gpu()) is y)

""" GPU上的计算"""
print((z + 2).exp() * y)

""" 4.6.3Gluon的GPU计算 """
net = nn.Sequential()




















