import mxnet as mx
import mxnet.gluon as gln
from mxnet import init
from mxnet.gluon import nn
from mxnet.gluon.contrib.estimator import CheckpointHandler, estimator
from mxnet.gluon.data.vision import transforms
from model.const import opt
from model.loader import wrapper_set
import os


class reshape(nn.HybridBlock):
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
        if dropout_rate > 0:
            _net.add(nn.Dropout(rate=dropout_rate))

    _net.add(nn.Dense(units=1024,
                      flatten=True,
                      activation='relu',
                      weight_initializer=init.Xavier(),
                      bias_initializer=init.Normal()))
    if dropout_rate > 0:
        _net.add(nn.Dropout(rate=dropout_rate))

    _net.add(nn.Dense(units=opt.CHAR_LEN * opt.MAX_CHAR_LEN,
                      weight_initializer=init.Xavier(),
                      bias_initializer=init.Normal()))
    _net.add(reshape())
    return _net


if __name__ == '__main__':
    net = build_net()
    net.hybridize()
    params = sorted([i for i in os.listdir('./trained') if i.endswith('.params')])
    print(f'picked param {params[len(params) - 1]}')
    net.load_parameters(f'./trained/{params[len(params) - 1]}')

    wrapper_train = wrapper_set(data_name='train',
                                label_name='label',
                                data_path=os.fspath('./../../sample/train'),
                                data_shape=(1, 80, 30),  # (c, w, h)
                                label_shape=opt.MAX_CHAR_LEN)

    wrapper_valid = wrapper_set(data_name='valid',
                                label_name='label',
                                data_path=os.fspath('./../../sample/test'),
                                data_shape=(1, 80, 30),  # (c, w, h)
                                label_shape=opt.MAX_CHAR_LEN)
    trans = transforms.Compose([
        transforms.ToTensor()
    ])

    wrapper_train = wrapper_train.transform_first(trans)
    wrapper_valid = wrapper_valid.transform_first(trans)
    train_loader = mx.gluon.data.DataLoader(dataset=wrapper_train, batch_size=128, shuffle=True)
    valid_loader = mx.gluon.data.DataLoader(dataset=wrapper_valid, batch_size=256, shuffle=False)

    loss_fn = gln.loss.SoftmaxCrossEntropyLoss(axis=1)

    learning_rate = 0.008
    num_epochs = 10
    trainer = gln.Trainer(net.collect_params(),
                          'sgd', {'learning_rate': learning_rate})

    acc = mx.metric.create(lambda y, y_hat: -loss_fn(y_hat, y))

    checkpoint_handler = CheckpointHandler(model_dir='./trained',
                                           model_prefix='baseline',
                                           monitor=acc,
                                           save_best=True)

    est = estimator.Estimator(net=net,
                              loss=loss_fn,
                              trainer=trainer)

    est.fit(train_data=train_loader,
            val_data=valid_loader,
            epochs=num_epochs,
            event_handlers=[checkpoint_handler])
