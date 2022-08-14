import random

import mxnet as mx
from mxnet import nd
import os
from model.const import opt
from model.utils import show_img

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


class SampleIter(mx.io.DataIter):

    def __init__(self, _opt, data_name, label_name, data_path, data_shape, label_shape, batch_size=16, shuffle=True):
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
        self._provide_data = [(data_name, data_shape)] * batch_size  # (batch_size, c, w, h)
        self._provide_label = [(label_name, label_shape)] * batch_size  # (batch_size, c, w, h)
        self.cur_batch = 0
        self.transform = mx.gluon.data.vision.transforms.ToTensor()

    def __iter__(self):
        return self

    def reset(self):
        self.cur_batch = 0

    def __next__(self):
        return self.next()

    def iter_next(self):
        self.cur_batch += 1
        return self.cur_batch <= self.num_batch

    def getindex(self):
        return self.cur_batch

    def getdata(self):
        start = self.cur_batch * self.batch_size
        end = start + self.batch_size
        picked = self.fnames[start:end]
        flag = self.data_shape[0] == 3  # 判断是否加载成灰度图
        return [self.transform(
            mx.image.imresize(
                mx.image.imread(os.path.join(self.data_path, fname), flag=1 if flag else 0),  # (w, h, c)
                w=self.data_shape[1], h=self.data_shape[2]
            )) for fname in picked]  # (c, h, w)

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


class SampleSetWrapper(mx.gluon.data.Dataset):
    """
    可以将 DataIter 适配给 Dataset
    然后 mx.gluon.data.DataLoader 也可以加载了
    """
    def __init__(self, dataiter):
        self.dataiter = dataiter

    def __getitem__(self, idx):
        # 移动到 idx 前一个
        self.dataiter.cur_batch = idx - 1
        _batch = self.dataiter.next()
        return _batch.data[0], _batch.label[0]

    def __len__(self):
        return self.dataiter.num_batch


def wrapper_set(data_path, data_shape, label_shape, data_name='train', label_name='label'):
    _iter = SampleIter(_opt=opt,
                       data_name=data_name,
                       label_name=label_name,
                       data_path=data_path,
                       data_shape=data_shape,  # (c, w, h)
                       label_shape=label_shape,
                       batch_size=1,  # batch_size 由 Dataloader 确定
                       shuffle=False)
    _iter.transform = lambda _x: _x
    _wrapper = SampleSetWrapper(dataiter=_iter)
    return _wrapper


if __name__ == '__main__':
    sample = SampleIter(_opt=opt,
                        data_name='train',
                        label_name='label',
                        data_path=os.fspath('../sample/train'),
                        data_shape=(3, 120, 60),  # (c, w, h)
                        label_shape=opt.MAX_CHAR_LEN)
    batch = sample.next()
    print(batch.data[0].shape)
    print(batch.label[0].shape)
    print(batch.data[0].dtype)
    print(batch.label[0].dtype)
    show_img(batch.data, batch.label, 5, 2, title_size=50)

    wrapper = wrapper_set(data_name='train',
                          label_name='label',
                          data_path=os.fspath('../sample/train'),
                          data_shape=(3, 120, 60),  # (c, w, h)
                          label_shape=opt.MAX_CHAR_LEN)
    loader = mx.gluon.data.DataLoader(dataset=wrapper, batch_size=10, shuffle=True, num_workers=4)
    x, y = iter(loader).__next__()
    print(x.shape)
    show_img(x, y, 5, 2, transpose=False, title_size=50)
