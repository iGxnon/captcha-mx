import os
import mxnet as mx

from model.ace_loss_res18.network import build_net
from model.utils import get_output
from model.utils import get_label
from model.utils import prepare_img


def predict_from(root, fname, net):
    img = prepare_img(os.fspath(root), fname)
    out = net(img)
    return img, get_output(out)[0], get_label(mx.nd.argmax(out, axis=-1))[0]


if __name__ == '__main__':
    net = build_net()
    params = sorted([i for i in os.listdir('./trained') if i.endswith('.params')])
    print(f'picked param {params[len(params) - 1]}')
    net.load_parameters(f'./trained/{params[len(params) - 1]}')
    # img, pred, raw = predict_from(root='./../../sample/train', fname='0EnM_1660290703.jpg', net=net)
    img, pred, raw = predict_from(root='./../../sample/train', fname='0QoD_1660645739.jpg', net=net)
    print(f'pred is "{pred}", raw is "{raw}", true is "0QoD"')
