from mxnet import nd
from model.const import opt
from matplotlib import pyplot as plt
import mxnet as mx
import os


def get_label(out):
    """
    获取的 out (, 4) 对应的 label(str)
    :param out:
    :return:
    """
    out = nd.array(out)
    if out.ndim > 1:
        return __get_labels(out)
    return __get_label(out)


def __get_label(out: nd.NDArray):
    # mxnet 中 NDArray 中标量也是一个数组
    return ''.join([opt.ALPHABET[int(x.asnumpy()[0])] for x in out])


def __get_labels(outs: nd.NDArray):
    return [__get_label(out) for out in outs]


def prepare_img(root, fname, shape=(3, 120, 40)):
    flag = shape[0] == 3  # 判断是否加载成灰度图
    img = mx.image.imresize(
        mx.image.imread(os.path.join(root, fname), flag=1 if flag else 0),
        w=shape[1], h=shape[2]  # (w, h, c)
    )
    trans = mx.gluon.data.vision.transforms.ToTensor()
    img = trans(img)
    return mx.nd.reshape(img, (1, *img.shape))


def show_img(X: nd.NDArray, y: nd.NDArray, rows, cols, transpose=True, title_size=30, figsize=(25, 25)):
    X = X[:rows * cols]
    y = y[:rows * cols]
    _, figs = plt.subplots(rows, cols, figsize=figsize)
    if rows * cols == 1:
        if transpose:
            figs.imshow(X[0].transpose((1, 2, 0)).asnumpy())
        else:
            figs.imshow(X[0].asnumpy())
        figs.axes.set_title(get_label(y[0]))
        figs.axes.title.set_fontsize(title_size)
        figs.axes.get_xaxis().set_visible(False)
        figs.axes.get_yaxis().set_visible(False)
        plt.show()
        return

    for fig, x, yi in zip(figs.reshape(rows * cols), X, y):
        if transpose:
            fig.imshow(x.transpose((1, 2, 0)).asnumpy())
        else:
            fig.imshow(x.asnumpy())
        fig.axes.set_title(get_label(yi))
        fig.axes.title.set_fontsize(title_size)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
    plt.show()


def count_rptch(text):
    maxch = (1, 0)
    nowch = (0, 0)
    lastch = None
    for index, i in enumerate(text):
        if lastch == i:
            nowch = (nowch[0] + 1, nowch[1])
            if nowch[0] > maxch[0]:
                maxch = nowch
        else:
            nowch = (1, index)
        lastch = i

    return maxch


def remove_rptch(text, tar_len=4):
    def rmchr(text, index):
        return text[:index] + text[index + 1:]

    while len(text) > tar_len:
        maxch = count_rptch(text)
        if maxch[0] <= 1:
            break
        text = rmchr(text, maxch[1])
    return text


def get_output(_input):
    """
    input (batch_size, pred_chars(T), dict_length)
    将模型输出转换成 label (str)
    """
    _input = nd.array(_input)
    return [remove_rptch(out) for out in get_label(nd.argmax(_input, axis=-1))]


def acc_metric(label, pred):
    pred = get_output(pred)
    label = get_label(label)
    bingo = 0
    for i in range(len(pred)):
        if pred[i] == label[i]:
            bingo += 1
    return bingo / len(pred)


if __name__ == '__main__':
    raw = nd.random.randn(4, 10, 63)
    print(get_output(raw))
