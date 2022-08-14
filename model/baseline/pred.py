import random
import os
from model.utils import prepare_img
from mxnet import nd
from model.utils import get_label
from model.utils import show_img
from train import build_net


def predict_from(root, fname, net):
    img = prepare_img(os.fspath(root), fname, shape=(1, 80, 30))
    out = net(img)
    return img.reshape(*img.shape[1:]), get_label(nd.argmax(out[0], 0))


if __name__ == '__main__':
    net = build_net()
    dataset = 'valid_hard'
    params = sorted([i for i in os.listdir('./trained') if i.endswith('.params')])
    print(f'picked param {params[len(params)-1]}')
    net.load_parameters(f'./trained/{params[len(params)-1]}')
    tests = random.choices([i for i in os.listdir(f'./../../sample/{dataset}') if i.endswith('.jpg')], k=5)
    for fname in tests:
        img = prepare_img(root=f'./../../sample/{dataset}', fname=fname, shape=(1, 80, 30))
        pred = nd.argmax(net(img), axis=1)
        show_img(img, pred, cols=1, rows=1, title_size=60)



