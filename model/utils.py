from mxnet import nd
from const import opt
from matplotlib import pyplot as plt


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


def show_img(X: nd.NDArray, y: nd.NDArray, rows, cols, title_size=30, figsize=(25, 25)):
    X = X[:rows * cols]
    y = y[:rows * cols]
    _, figs = plt.subplots(rows, cols, figsize=figsize)
    for fig, x, yi in zip(figs.reshape(rows * cols), X, y):
        fig.imshow(x.transpose((1, 2, 0)).asnumpy())
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


if __name__ == '__main__':
    raw = nd.random.randn(4, 10, 63)
    print(get_output(raw))