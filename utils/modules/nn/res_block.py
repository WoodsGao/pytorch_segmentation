import torch.nn as nn
from random import random
from . import CNS, SeparableCNS, EmptyLayer, SELayer, WSConv2d, DropConnect


class ResBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            ksize=3,
            stride=1,
            dilation=1,
            drop_rate=0.2,
            se=True,
            reps=1,
    ):
        super(ResBlock, self).__init__()
        blocks = [
            SeparableCNS(in_channels, out_channels, ksize, stride, dilation,
                         drop_rate, se)
        ]
        for i in range(reps):
            blocks.append(
                ResConv(out_channels, out_channels, ksize, 1, dilation,
                        drop_rate, se))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class ResConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize=3,
                 stride=1,
                 dilation=1,
                 drop_rate=0.2,
                 se=True):
        super(ResConv, self).__init__()
        if stride == 1 and in_channels == out_channels:
            self.add = True
        else:
            self.add = False
        self.block = nn.Sequential(
            SeparableCNS(in_channels,
                         out_channels,
                         ksize=ksize,
                         stride=stride,
                         dilation=dilation),
            SeparableCNS(out_channels,
                         out_channels,
                         ksize=ksize,
                         stride=1,
                         dilation=dilation,
                         activate=False),
            SELayer(out_channels) if se else EmptyLayer(),
            DropConnect(drop_rate)
            if drop_rate > 0 and self.add else EmptyLayer(),
        )

    def forward(self, x):
        if self.training and self.add and random() < self.drop_rate:
            return x
        f = self.block(x)
        if self.add:
            if not self.training:
                f *= (1 - self.drop_rate)
            f += x
        return f
