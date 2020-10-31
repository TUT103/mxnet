""" 4.3.1延后初始化 """
from mxnet import init, nd
from mxnet.gluon import nn


class MyInit(init.Initializer):
    def _init_weight(self, name, data):
        print("Init", name, data.shape)


net = nn.Sequential()
net.add(nn.Dense(256, activation="relu"),
        nn.Dense(10))
# net.initialize(init=MyInit())
#
# X = nd.random.uniform(shape=(2, 20))
# Y = net(X)
#
# Y = net(X)

""" 4.3.2避免延后初始化 """
# 方案一
# net.initialize(init=MyInit(), force_reinit=True)

# 方案二
net = nn.Sequential()
net.add(nn.Dense(256, in_units=20, activation="relu"))
net.add(nn.Dense(10, in_units=256))

net.initialize(init=MyInit())
