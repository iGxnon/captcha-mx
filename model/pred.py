import os
import mxnet as mx

from model.train import build_net
from model.utils import get_output
from model.utils import get_label


def prepare_img(root, fname, shape=(3, 120, 60)):
    flag = shape[0] == 3  # 判断是否加载成灰度图
    img = mx.image.imresize(
        mx.image.imread(os.path.join(root, fname), flag=1 if flag else 0),
        w=shape[1], h=shape[2]  # (w, h, c)
    )
    trans = mx.gluon.data.vision.transforms.ToTensor()
    img = trans(img)
    return mx.nd.reshape(img, (1, *img.shape))


def predict_from(root, fname, net):
    img = prepare_img(os.fspath(root), fname)
    out = net(img)
    return img, get_output(out)[0], get_label(mx.nd.argmax(out, axis=-1))[0]


if __name__ == '__main__':
    net = build_net()
    net.load_parameters('./trained/my_model-epoch19batch3000.params')
    img, pred, raw = predict_from(root='./../sample/train', fname='0arV_1660302295.jpg', net=net)
    print(f'pred is "{pred}", raw is "{raw}", true is "0arV"')
