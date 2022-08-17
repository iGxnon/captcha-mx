from mxnet import gluon, init

import mxnet as mx
from mxnet.gluon.contrib.estimator import estimator
from mxnet.gluon.data.vision import transforms
from model.ace_loss_res18.network import build_net
from model.ace_loss_res18.network import ACELoss
from model.loader import wrapper_set
from model.const import opt
import os
from model.utils import acc_metric
from mxnet.gluon.contrib.estimator.event_handler import CheckpointHandler


def train():
    wrapper_train = wrapper_set(data_name='train',
                                label_name='label',
                                data_path=os.fspath('../../sample/train'),
                                data_shape=(3, 120, 40),  # (c, w, h)
                                label_shape=opt.MAX_CHAR_LEN)

    wrapper_valid = wrapper_set(data_name='valid',
                                label_name='label',
                                data_path=os.fspath('../../sample/test'),
                                data_shape=(3, 120, 40),  # (c, w, h)
                                label_shape=opt.MAX_CHAR_LEN)

    trans = transforms.Compose([
        transforms.ToTensor()
    ])

    wrapper_train = wrapper_train.transform_first(trans)
    wrapper_valid = wrapper_valid.transform_first(trans)

    train_loader = mx.gluon.data.DataLoader(dataset=wrapper_train, batch_size=128, shuffle=True)
    valid_loader = mx.gluon.data.DataLoader(dataset=wrapper_valid, batch_size=128, shuffle=True)
    net = build_net()
    params = sorted([i for i in os.listdir('./trained') if i.endswith('.params')])
    if len(params) != 0:
        print(f'picked param {params[len(params) - 1]}')
        net.load_parameters(f'./trained/{params[len(params) - 1]}')
    else:
        net.initialize(mx.init.Xavier())

    loss_fn = ACELoss()
    learning_rate = 0.0005
    num_epochs = 1
    trainer = gluon.Trainer(net.collect_params(),
                            optimizer=mx.optimizer.AdaDelta())
    acc = mx.metric.create(acc_metric)

    checkpoint_handler = CheckpointHandler(model_dir='trained',
                                           model_prefix='my_model',
                                           mode='max',
                                           monitor=acc,  # Monitors a metric
                                           save_best=True)  # Save the best model in terms of

    est = estimator.Estimator(net=net,
                              loss=loss_fn,
                              # train_metrics=acc,
                              trainer=trainer)

    est.fit(train_data=train_loader,
            # val_data=valid_loader,
            epochs=num_epochs,
            event_handlers=[checkpoint_handler])


if __name__ == '__main__':
    train()
