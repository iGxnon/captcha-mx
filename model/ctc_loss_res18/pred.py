import os
import mxnet as mx

from model.ctc_loss_res18.train import build_net
from model.utils import get_output
from model.utils import get_label
from model.utils import prepare_img


def predict_from(root, fname, net):
    img = prepare_img(os.fspath(root), fname)
    out = net(img)
    return img, get_output(out)[0], get_label(mx.nd.argmax(out, axis=-1))[0]


if __name__ == '__main__':
    net = build_net()
    net.load_parameters('./trained/my_model-epoch99batch8300.params')
    img, pred, raw = predict_from(root='./../../sample/train', fname='0EnM_1660290703.jpg', net=net)
    print(f'pred is "{pred}", raw is "{raw}", true is "0arV"')
