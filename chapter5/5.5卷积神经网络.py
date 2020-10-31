""" 5.5.1LeNet模型 """
import d2lzh as d2l
import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn
import time
import matplotlib.pyplot as plt

net = nn.Sequential()
net.add(nn.Conv2D(channels=6, kernel_size=5, activation="sigmoid"),
        nn.MaxPool2D(pool_size=2, strides=2),  # 池化层填充和步幅
        nn.Conv2D(channels=16, kernel_size=5, activation="sigmoid"),
        nn.MaxPool2D(pool_size=2, strides=2),

        nn.Dense(120, activation="sigmoid"),
        nn.Dense(84, activation="sigmoid"),
        nn.Dense(10))
print(net)

X = nd.random.uniform(shape=(1, 1, 28, 28))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, "output shape:\t", X.shape)

""" 5.5.2获取数据和训练模型 """
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)


def try_gpu():
    try:
        ctx = mx.gpu()
        _ = nd.zeros((1,), ctx=ctx)
    except mx.base.MXNetError:
        ctx = mx.cpu()
    return ctx


ctx = try_gpu()
print(ctx)


def evaluate_accuracy(data_iter, net, ctx):
    acc_sum, n = nd.array([0], ctx=ctx), 0
    for X, y in data_iter:
        X, y = X.as_in_context(ctx), y.as_in_context(ctx).astype("float32")
        acc_sum += (net(X).argmax(axis=1) == y).sum()
        n += y.size
    return acc_sum.asscalar() / n


def train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs):
    print("training on", ctx)
    loss = gloss.SoftmaxCrossEntropyLoss()
    count_x, count_y1, count_y2, count_y3 = [], [], [], []
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X, y = X.as_in_context(ctx), y.as_in_context(ctx)
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y).sum()
            l.backward()
            trainer.step(batch_size)
            y = y.astype("float32")
            train_l_sum += l.asscalar()
            train_acc_sum += (y_hat.argmax(axis=1) == y).sum().asscalar()
            n += y.size
        test_acc = evaluate_accuracy(test_iter, net, ctx)
        print("epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec"
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc, time.time() - start))
    #     count_x.append(epoch + 1)
    #     count_y1.append(float(loss))
    #     count_y2.append(train_acc_sum / n)
    #     count_y3.append(test_acc)
    # plt.xlabel("epochs")
    # plt.ylabel("data")
    # print(count_x, count_y1, count_y2, count_y3)
    # plt.plot(count_x, list(count_y1), count_x, list(count_y2), count_x, list(count_y3))
    # plt.show()


lr, num_epochs = 4.0, 5
net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
for lr in range(100):
    print("\n")
    print("lr", lr / 10)
    trainer = gluon.Trainer(net.collect_params(), "sgd", {"learning_rate": lr / 10})
    train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)
