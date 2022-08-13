from mxnet import nd, gluon, init, autograd
from mxnet.gluon import nn
import mxnet as mx
import numpy as np
from mxnet.gluon.data.vision import transforms

net = nn.Sequential()
net.add(nn.Dense(128, activation='relu'))
net.add(nn.Dense(64, activation='relu'))
net.add(nn.Dense(10))


def transform(data, label):
    return data.astype(np.float32) / 255, label.astype(np.float32)


batch_size = 1
num_workers = 8
train_data = gluon.data.DataLoader(
    gluon.data.vision.MNIST(train=True).transform_first(transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, num_workers=num_workers)

mod = mx.mod.Module(symbol=net)
