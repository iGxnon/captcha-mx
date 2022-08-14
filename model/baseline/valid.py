import mxnet as mx
import mxnet.gluon as gln
from mxnet.gluon.contrib.estimator import estimator
from mxnet.gluon.data.vision import transforms
from model.const import opt
from model.loader import wrapper_set
import os
from model.baseline.train import build_net

if __name__ == '__main__':
    net = build_net()
    net.hybridize()
    net.load_parameters('./trained/baseline_model-epoch9batch830.params')

    wrapper_train = wrapper_set(data_name='train',
                                label_name='label',
                                data_path=os.fspath('./../../sample/train_hard'),
                                data_shape=(1, 80, 30),  # (c, w, h)
                                label_shape=opt.MAX_CHAR_LEN)

    wrapper_test = wrapper_set(data_name='test',
                               label_name='label',
                               data_path=os.fspath('./../../sample/test_gen'),
                               data_shape=(1, 80, 30),  # (c, w, h)
                               label_shape=opt.MAX_CHAR_LEN)
    trans = transforms.Compose([
        transforms.ToTensor()
    ])

    wrapper_train = wrapper_train.transform_first(trans)
    wrapper_test = wrapper_test.transform_first(trans)
    train_loader = mx.gluon.data.DataLoader(dataset=wrapper_train, batch_size=128, shuffle=True)
    test_loader = mx.gluon.data.DataLoader(dataset=wrapper_test, batch_size=256, shuffle=False)

    loss_fn = gln.loss.SoftmaxCrossEntropyLoss(axis=1)

    learning_rate = 0.008
    num_epochs = 1
    trainer = gln.Trainer(net.collect_params(),
                          'sgd', {'learning_rate': learning_rate})

    est = estimator.Estimator(net=net,
                              loss=loss_fn,
                              trainer=trainer)

    est.fit(train_data=train_loader,
            val_data=test_loader,
            epochs=num_epochs)
