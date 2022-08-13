import random

import matplotlib.pyplot as plt
import mxnet as mx
from mxnet import nd
import os
from const import opt

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


class SampleIter(mx.io.DataIter):

    def __init__(self, _opt, data_name, label_name, data_path, data_shape, label_shape, batch_size=10, shuffle=True):
        super().__init__(batch_size)
        self.opt = _opt
        self.fnames = os.listdir(data_path)
        self.data_path = data_path
        self.data_shape = data_shape
        if shuffle:
            random.shuffle(self.fnames)
        self.data_size = len(self.fnames)
        self.num_batch = self.data_size // batch_size
        self.data_size = int(self.num_batch * batch_size)  # 忽略多余的
        self._provide_data = [data_name, data_shape] * batch_size
        self._provide_label = [label_name, label_shape] * batch_size
        self.cur_batch = 0

    def __iter__(self):
        return self

    def reset(self):
        self.cur_batch = 0

    def __next__(self):
        return self.next()

    def iter_next(self):
        return self.cur_batch < self.num_batch

    def getdata(self):
        start = self.cur_batch * self.batch_size
        end = start + self.batch_size
        picked = self.fnames[start:end]
        return [mx.image.imresize(mx.image.imread(os.path.join(self.data_path, fname)),
                                  w=self.data_shape[0], h=self.data_shape[1]) for fname in picked]

    def getlabel(self):
        start = self.cur_batch * self.batch_size
        end = start + self.batch_size
        picked = self.fnames[start:end]
        texts = [fname.split('_')[0] for fname in picked]
        labels = [nd.array([self.opt.ALPHABET_DICT[i] for i in list(k)], dtype='float32') for k in texts]
        return labels

    @property
    def provide_data(self):
        return self._provide_data

    @property
    def provide_label(self):
        return self._provide_label


if __name__ == '__main__':
    sample = SampleIter(_opt=opt,
                        data_name='train',
                        label_name='label',
                        data_path=os.fspath('./../sample/train'),
                        data_shape=(120, 60),
                        label_shape=opt.MAX_CHAR_LEN)
    plt.imshow(sample.next().data[0].asnumpy())
    plt.show()
