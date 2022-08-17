from mxnet import init
from mxnet.gluon import nn
from model.const import opt


class _Reshape(nn.HybridBlock):
    def hybrid_forward(self, F, _x, *args, **kwargs):
        # 吐了，opt.CHAR_LEN 和 opt.MAX_CHAR_LEN 换个位置，softmaxce 设置一下 axies，就说形状不匹配
        # 翻到之前有一个 issue 也碰到这样的问题
        return F.reshape(_x, (-1, opt.CHAR_LEN, opt.MAX_CHAR_LEN))


def build_net(dropout_rate=0.3):
    _net = nn.HybridSequential(prefix='')
    for i in [32, 64, 128]:
        _net.add(nn.Conv2D(channels=i,
                           kernel_size=3,
                           strides=1,
                           activation='relu',
                           weight_initializer=init.Xavier(),
                           bias_initializer=init.Normal()))
        _net.add(nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        _net.add(nn.Dropout(rate=dropout_rate))

    _net.add(nn.Dense(units=1024,
                      flatten=True,
                      activation='relu',
                      weight_initializer=init.Xavier(),
                      bias_initializer=init.Normal()))

    _net.add(nn.Dropout(rate=dropout_rate))

    _net.add(nn.Dense(units=opt.CHAR_LEN * opt.MAX_CHAR_LEN,
                      weight_initializer=init.Xavier(),
                      bias_initializer=init.Normal()))
    _net.add(_Reshape())
    return _net
