""" 5.2.1填充 """
from mxnet import nd
from mxnet.gluon import nn


# 定义一个函数来计算卷积层。它初始化卷积层权重，并对输入和输出做相应的升维和降维
def comp_conv2d(conv2d, X):
    conv2d.initialize()
    # (1, 1)代表批量大小和通道数
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    return Y.reshape(Y.shape[2:])


conv2d = nn.Conv2D(1, kernel_size=3, padding=1)
X = nd.random.uniform(shape=(8, 8))
print(comp_conv2d(conv2d, X))
print(comp_conv2d(conv2d, X).shape)
print("=================================================================")
conv2d = nn.Conv2D(1, kernel_size=(5, 3), padding=(2, 1))
print(comp_conv2d(conv2d, X).shape)

""" 5.2.2步幅 """
conv2d = nn.Conv2D(1, kernel_size=3, padding=1, strides=2)
print(comp_conv2d(conv2d, X).shape)

conv2d = nn.Conv2D(1, kernel_size=(3, 5), padding=(0, 1), strides=2)
print(comp_conv2d(conv2d, X).shape)

