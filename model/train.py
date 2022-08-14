from mxnet import nd, gluon, init, autograd
from mxnet.gluon import nn
import mxnet as mx
from mxnet.gluon.contrib.estimator import estimator
from mxnet.gluon.data.vision import transforms
from mxnet.gluon.model_zoo import vision
from loader import wrapper_set
from const import opt
import os
from utils import acc_metric
from mxnet.gluon.contrib.estimator.event_handler import TrainBegin, TrainEnd, EpochEnd, CheckpointHandler


class OutputLayer(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(OutputLayer, self).__init__(**kwargs)
        # use name_scope to give child Blocks appropriate names.
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            # 给长度加一个 padding
            self.features.add(nn.Conv2D(channels=128, kernel_size=2, strides=2, padding=(0, 1), use_bias=False))
            self.features.add(nn.BatchNorm())
            self.features.add(nn.Activation('relu'))
            self.features.add(nn.MaxPool2D(3, 2, 1))
            self.dense = nn.Dense(opt.CHAR_LEN, flatten=False)

    def hybrid_forward(self, F, _x, *args, **kwargs):
        # 通道移动到最后
        _x = self.features(_x)
        _x = _x.transpose((0, 2, 3, 1))
        # 通道上做 softmax
        _x = F.softmax(_x, axis=-1)
        # 融合 h, w 纬度
        _x = _x.reshape((-1, 10, 128))

        # print(_x.shape)
        _x = self.dense(_x)
        return _x


def build_net():
    net = nn.HybridSequential(prefix='')
    net.add(nn.BatchNorm(scale=False, center=False))
    net.add(nn.Conv2D(channels=64, kernel_size=5, strides=2, padding=3, use_bias=False))
    net.add(nn.BatchNorm())
    net.add(nn.Activation('relu'))
    net.add(nn.MaxPool2D(3, 2, 1))

    resnet_18_v2_feat = vision.resnet18_v2(pretrained=False).features
    net.add(*resnet_18_v2_feat[5:7])
    net.add(*resnet_18_v2_feat[9:-2])

    net.add(OutputLayer())

    net.hybridize()
    return net


if __name__ == '__main__':
    wrapper_train = wrapper_set(data_name='train',
                                label_name='label',
                                data_path=os.fspath('./../sample/train'),
                                data_shape=(3, 120, 60),  # (c, w, h)
                                label_shape=opt.MAX_CHAR_LEN)

    wrapper_valid = wrapper_set(data_name='valid',
                                label_name='label',
                                data_path=os.fspath('./../sample/valid'),
                                data_shape=(3, 120, 60),  # (c, w, h)
                                label_shape=opt.MAX_CHAR_LEN)

    trans = transforms.Compose([
        transforms.ToTensor()
    ])

    wrapper_train = wrapper_train.transform_first(trans)
    wrapper_valid = wrapper_valid.transform_first(trans)

    train_loader = mx.gluon.data.DataLoader(dataset=wrapper_train, batch_size=64, shuffle=True, num_workers=4)
    valid_loader = mx.gluon.data.DataLoader(dataset=wrapper_valid, batch_size=32, shuffle=True, num_workers=2)

    net = build_net()
    net.initialize(init.Xavier())
    loss_fn = gluon.loss.CTCLoss(layout='NTC', label_layout='NT')
    learning_rate = 0.03
    num_epochs = 1000
    trainer = gluon.Trainer(net.collect_params(),
                            'sgd', {'learning_rate': learning_rate})
    acc = mx.metric.create(acc_metric)

    checkpoint_handler = CheckpointHandler(model_dir='./trained',
                                           model_prefix='my_model',
                                           monitor=acc,  # Monitors a metric
                                           save_best=True)  # Save the best model in terms of

    est = estimator.Estimator(net=net,
                              loss=loss_fn,
                              train_metrics=acc,
                              trainer=trainer)

    est.fit(train_data=train_loader,
            val_data=valid_loader,
            epochs=num_epochs,
            event_handlers=[checkpoint_handler])
