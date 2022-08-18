from model.utils import get_label
from model.utils import prepare_img
import os
import random
import mxnet as mx
from network import build_net


def predict_from(root, fname, net):
    img = prepare_img(os.fspath(root), fname, shape=(1, 160, 60))
    out = ''
    for i in range(4):
        piece = img[:, :, :, i*40:(i+1)*40]
        pred = mx.nd.argmax(net(piece)[0])
        out += get_label(pred)
    return out


if __name__ == '__main__':
    net = build_net(0.)
    dataset = './../../sample/real'
    params = sorted([i for i in os.listdir('./trained') if i.endswith('.params')])
    print(f'picked param {params[len(params) - 1]}')
    net.load_parameters(f'./trained/{params[len(params) - 1]}')
    tests = random.choices([i for i in os.listdir(dataset) if i.endswith('.jpg')], k=5)
    for fname in tests:
        pred = predict_from(dataset, fname, net)
        print(fname, ' is predicted to ', pred)
