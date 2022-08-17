import warnings

from mxnet.gluon import nn
import mxnet as mx
from model.const import opt
from mxnet.gluon.model_zoo import vision

warnings.filterwarnings('ignore')


# Aggregation Cross-Entropy (https://arxiv.org/abs/1904.08364)
class ACELoss(mx.gluon.loss.Loss):

    def __init__(self, _l=75, weight=1., batch_axis=0, **kwargs):
        """
        l 需要和输出的维度匹配，表示预测序列长度
        weight 参数没有作用
        """
        super(ACELoss, self).__init__(weight, batch_axis, **kwargs)
        self.l = _l

    def hybrid_forward(self, F, pred, label):
        # label 形状: (batch_size, opt.MAX_CHAR_LEN), Ep. [[34, 9, 13, 23], [19, 34, 25, 43]]
        label = F.one_hot(label, opt.CHAR_LEN)
        label = F.sum(label, axis=1) / self.l
        pred = F.sum(pred, axis=1) / self.l
        loss = - F.sum(F.log(pred) * label)  # 取交叉熵
        loss = F.mean(loss, axis=self._batch_axis, exclude=True)
        print(loss)
        return loss


class OutputLayer(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(OutputLayer, self).__init__(**kwargs)
        with self.name_scope():
            # 通道降维至字母表上所有的字符
            self.dense = nn.Dense(opt.CHAR_LEN, flatten=False)

    def hybrid_forward(self, F, _x, *args, **kwargs):
        # 将 h w 展开
        _x = F.reshape(_x, (0, 0, -3))
        # 通道移动到最后
        _x = F.transpose(_x, (0, 2, 1))
        # 通道降维
        _x = self.dense(_x)
        # 通道上做 softmax
        _x = F.softmax(_x, axis=-1)
        # for log(_x) 防止后续计算 ACE 损失有 log(0)
        _x = _x + 1e-10
        # out shape: (batch_size, 9, opt.CHAR_LEN)
        return _x


def build_net():
    _net = nn.HybridSequential(prefix='')
    _net.add(nn.BatchNorm(scale=False, center=False))
    _net.add(nn.Conv2D(channels=64, kernel_size=3, strides=1, padding=1))
    _net.add(nn.BatchNorm())
    _net.add(nn.Activation('relu'))
    _net.add(nn.MaxPool2D(pool_size=2, strides=2))

    _net.add(nn.Dropout(0.))

    resnet_18_v2_feat = vision.resnet18_v1(pretrained=False).features
    for layer in resnet_18_v2_feat[4:-2]:
        _net.add(layer)
        _net.add(nn.Dropout(0.))

    _net.add(OutputLayer())

    _net.hybridize()
    return _net


# For test
if __name__ == '__main__':
    build_net()
