# captcha-mx

> 基于 [`MXNet`](https://mxnet.incubator.apache.org/versions/1.9.1/) 的简单验证码序列预测模型

## TOC

- [文件结构](#文件结构)
- [预测模型结构](#模型结构)
  - [Baseline](#Baseline)
  - [Res18-CTC](#Res18-CTC)
  - [Rese18-ACE](#Rese18-ACE)
  - [Baseline-split](#Baseline-split)
- [样本生成模型结构](#样本生成模型结构)
- [结果](#结果)

## 文件结构

```
...
```

## 模型结构

### Baseline

| **Conv2D(1 -> 32, kernel_size=(3, 3), stride=(1, 1), Activation(relu))** |
| :----------------------------------------------------------: |
|  **MaxPool2D(size=(2, 2), stride=(2, 2), padding=(0, 0))**   |
|                        **Dropout()**                         |
| **Conv2D(32 -> 64, kernel_size=(3, 3), stride=(1, 1), Activation(relu))** |
|  **MaxPool2D(size=(2, 2), stride=(2, 2), padding=(0, 0))**   |
|                        **Dropout()**                         |
| **Conv2D(64 -> 128, kernel_size=(3, 3), stride=(1, 1), Activation(relu))** |
|  **MaxPool2D(size=(2, 2), stride=(2, 2), padding=(0, 0))**   |
|                        **Dropout()**                         |
| **Dense([inputs] -> 1024, linear, Flatten(), Activation(relu))** |
| **Dropout()** |
| **Dense(1024 -> opt.CHAR_LEN * opt.MAX_CHAR_LEN, linear)** |
| **reshape(-1, opt.CHAR_LEN, opt.MAX_CHAR_LEN)** |

---

### Res18-CTC (已经移除)

> 有点类似 `DFCNN`，只不过卷积层用的是 Res18 的一部分，最后套了一个 `CTCLoss`

网络构建过程

```python
_net = nn.HybridSequential(prefix='')
_net.add(nn.BatchNorm(scale=False, center=False))
_net.add(nn.Conv2D(channels=64, kernel_size=5, strides=2, padding=3, use_bias=False))
_net.add(nn.BatchNorm())
_net.add(nn.Activation('relu'))
_net.add(nn.MaxPool2D(3, 2, 1))
_net.add(nn.Dropout(0.7))

resnet_18_v2_feat = vision.resnet18_v2(pretrained=False).features
for layer in resnet_18_v2_feat[5:7]:
    _net.add(layer)
    _net.add(nn.Dropout(0.7))

for layer in resnet_18_v2_feat[9:-2]:
    _net.add(layer)
    _net.add(nn.Dropout(0.7))

_net.add(OutputLayer())

_net.hybridize()
return _net
```

```python
class OutputLayer(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(OutputLayer, self).__init__(**kwargs)
        # use name_scope to give child Blocks appropriate names.
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            # 给长度加一个 padding
            self.features.add(nn.Conv2D(channels=128, kernel_size=2, strides=2, padding=(0, 1), use_bias=False))
            self.features.add(nn.BatchNorm())
            self.features.add(nn.Activation('relu'))
            self.features.add(nn.MaxPool2D(3, 2, 1))
            self.features.add(nn.Dropout(0.7))
            self.dense = nn.Dense(opt.CHAR_LEN, flatten=False)

    def hybrid_forward(self, F, _x, *args, **kwargs):
        # 通道移动到最后
        _x = self.features(_x)
        _x = _x.transpose((0, 2, 3, 1))
        # 通道上做 softmax
        # _x = F.softmax(_x, axis=-1)
        # 融合 h, w 纬度
        _x = _x.reshape((-1, 10, 128))

        # print(_x.shape)
        _x = self.dense(_x)
        return _x
```

### Rese18-ACE (暂未收敛)

### Baseline-split

## 样本生成模型结构

## 结果

> Baseline 最终的性能可以达到高难度数据集上 (即很多 Z2, S5, B8, 0O 等难以辨认的字符组合而成)
> 训练集 99% 精度，测试集 98% 以上精度，投入实际中也有 70% 左右的精度

> Res18-CTC 最终没能收敛，损失一直下降不了，估计是模型层数深了梯度消失了，也可能是我的垃圾硬件没能等到它收敛的那一刻就等得不耐烦了
> 待进一步优化
> 上网检索后发现也有一些用了 CTCLoss 的模型不能收敛，如　https://gitee.com/mindspore/mindspore/issues/I43WBR

