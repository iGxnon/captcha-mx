import os
import mxnet as mx

from train import build_net
from utils import get_output


def prepare_img(root, fname, shape=(3, 120, 60)):
    img = mx.image.imresize(
        mx.image.imread(os.path.join(root, fname)),
        w=shape[1], h=shape[2]  # (w, h, c)
    )
    trans = mx.gluon.data.vision.transforms.ToTensor()
    img = trans(img)
    return mx.nd.reshape(img, (1, *img.shape))


def predict_from(root, fname, net):
    img = prepare_img(os.fspath(root), fname)
    return get_output(net(img))[0]


if __name__ == '__main__':
    net = build_net()
    net.load_parameters('./trained/my_model-best.params')
    print(predict_from(root='./../sample/test', fname='1aAi_1660291390.jpg', net=net))
