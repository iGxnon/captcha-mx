import mxnet as mx
import mxnet.gluon as gln
from mxnet.gluon.contrib.estimator import CheckpointHandler, estimator
from mxnet.gluon.data.vision import transforms
from model.const import opt
from model.loader import wrapper_set
import os
from network import build_net

if __name__ == '__main__':
    wrapper_train = wrapper_set(data_name='train',
                                label_name='label',
                                data_path=os.fspath('./../../sample/train-1'),
                                data_shape=(1, 40, 60),  # (c, w, h)
                                label_shape=opt.MAX_CHAR_LEN)

    wrapper_valid = wrapper_set(data_name='valid',
                                label_name='label',
                                data_path=os.fspath('./../../sample/test-1'),
                                data_shape=(1, 40, 60),  # (c, w, h)
                                label_shape=opt.MAX_CHAR_LEN)

    net = build_net(0.4)
    params = sorted([i for i in os.listdir('./trained') if i.endswith('.params')])
    if len(params) != 0:
        print(f'picked param {params[len(params) - 1]}')
        net.load_parameters(f'./trained/{params[len(params) - 1]}')
    else:
        net.initialize(mx.init.Xavier())

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
                          optimizer=mx.optimizer.AdaDelta())

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
