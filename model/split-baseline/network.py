from mxnet.gluon import nn
from mxnet import init
from mxnet import nd
from model.utils import opt


def build_net(dropout_rate=0.7):
    _net = nn.HybridSequential(prefix='')
    for i in [16, 32, 64]:
        _net.add(nn.Conv2D(channels=i,
                           kernel_size=3,
                           strides=1,
                           activation='relu',
                           weight_initializer=init.Xavier(),
                           bias_initializer=init.Normal()))
        _net.add(nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        _net.add(nn.Dropout(rate=dropout_rate))

    _net.add(nn.Dense(units=256,
                      flatten=True,
                      activation='relu',
                      weight_initializer=init.Xavier(),
                      bias_initializer=init.Normal()))

    _net.add(nn.Dropout(rate=dropout_rate))

    _net.add(nn.Dense(units=opt.CHAR_LEN,
                      weight_initializer=init.Xavier(),
                      bias_initializer=init.Normal()))
    _net.hybridize()
    return _net


if __name__ == '__main__':
    net = build_net(0.7)
    net.initialize()
    img = nd.normal(shape=(1, 1, 60, 40))
    out = net(img)
    print(out.shape)