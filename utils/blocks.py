import torch
import torch.nn as nn
import math


relu = nn.LeakyReLU(0.1, inplace=True)
bn = nn.BatchNorm2d


class SELayer(nn.Module):
    def __init__(self, filters):
        super(SELayer, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.weight = nn.Sequential(
            # bn(filters),
            nn.Conv2d(filters, filters // 16, 1, bias=False),
            relu,
            nn.Conv2d(filters // 16, filters, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        gap = self.gap(x)
        weight = self.weight(gap)
        return x * weight


class EmptyLayer(nn.Module):
    def forward(self, x):
        return x


class ResBlock(nn.Module):
    def __init__(self, in_features, out_features, stride=1, se_block=False):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            bn(in_features),
            relu,
            SELayer(in_features) if se_block else EmptyLayer(),
            nn.Conv2d(in_features, out_features // 2, 1, 1, 0, bias=False),
            bn(out_features // 2),
            relu,
            nn.Conv2d(out_features // 2,
                      out_features,
                      3,
                      stride,
                      1,
                      bias=False,
                      groups=32 if out_features % 32 == 0 else 1,
            ),
            SELayer(out_features) if se_block else EmptyLayer(),
        )
        self.downsample = EmptyLayer()
        if stride > 1 or in_features != out_features:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_features, out_features, 3, stride, 1), )

    def forward(self, x):
        return self.downsample(x) + self.block(x)
