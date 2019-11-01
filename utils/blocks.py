import torch
import torch.nn as nn
import math
import torch.nn.functional as F

relu = nn.ReLU(True)
lrelu = nn.LeakyReLU(0.1, inplace=True)
bn = nn.BatchNorm2d


class CReLU(nn.Module):
    def forward(self, x):
        return torch.cat([relu(x), relu(-x)], 1)


crelu = CReLU()


class SELayer(nn.Module):
    def __init__(self, filters):
        super(SELayer, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.weight = nn.Sequential(
            nn.Conv2d(filters, filters // 16 if filters >= 32 else 8, 1),
            lrelu,
            nn.Conv2d(filters // 16 if filters >= 32 else 8, filters, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        gap = self.gap(x)
        weight = self.weight(gap)
        return x * weight


class EmptyLayer(nn.Module):
    def forward(self, x):
        return x


class AsppPooling(nn.Module):
    def __init__(self, in_features, out_features):
        super(AsppPooling, self).__init__()
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 BLD(in_features, out_features, 1))

    def forward(self, x):
        size = x.size()[2:]
        x = self.gap(x)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        return x


class Aspp(nn.Module):
    def __init__(self, in_features, out_features, rates=[6, 12, 18]):
        super(Aspp, self).__init__()
        conv_list = [BLD(in_features, out_features, 1)]
        conv_list.append(AsppPooling(in_features, out_features))
        for rate in rates:
            conv_list.append(BLD(in_features, out_features, dilation=rate))
        self.conv_list = nn.ModuleList(conv_list)
        self.project = BLD((2 + len(rates)) * out_features, out_features, 1)

    def forward(self, x):
        outputs = []
        for conv in self.conv_list:
            outputs.append(conv(x))
        outputs = torch.cat(outputs, 1)
        outputs = self.project(outputs)
        return outputs


class BLD(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 ksize=3,
                 stride=1,
                 groups=1,
                 dilation=1,
                 activate=lrelu,
                 se_block=False):
        super(BLD, self).__init__()
        if activate is None:
            activate = EmptyLayer()
        if se_block:
            se_block = SELayer(in_features)
        else:
            se_block = EmptyLayer()
        self.bld = nn.Sequential(
            bn(in_features), se_block, activate,
            nn.Conv2d(in_features * 2 if isinstance(activate, CReLU) else in_features,
                      out_features,
                      ksize,
                      stride=stride,
                      padding=(ksize - 1) // 2 - 1 + dilation,
                      groups=groups,
                      dilation=dilation,
                      bias=False))

    def forward(self, x):
        return self.bld(x)


class ResBlock(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 stride=1,
                 dilation=1,
                 se_block=True):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            BLD(in_features, out_features // 4, 1, activate=crelu),
            BLD(
                out_features // 4,
                out_features // 4,
                stride=stride,
                dilation=dilation,
                groups=out_features // 4,
            ),
            BLD(out_features // 4,
                out_features,
                1,
                activate=None,
                se_block=se_block))
        self.downsample = EmptyLayer()
        if stride > 1 or in_features != out_features:
            self.downsample = BLD(in_features,
                                  out_features,
                                  3,
                                  stride,
                                  activate=None)

    def forward(self, x):
        return self.downsample(x) + self.block(x)


class DenseBlock(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 stride=1,
                 dilation=1,
                 drop_rate=0.5,
                 se_block=True):
        super(DenseBlock, self).__init__()
        assert in_features == out_features
        assert in_features % 2 == 0
        assert stride == 1
        features = in_features
        self.features = features // 2
        self.block = nn.Sequential(
            BLD(features, features // 4, 1, activate=crelu),
            BLD(
                features // 4,
                features // 4,
                stride=stride,
                dilation=dilation,
                groups=features // 4,
            ),
            BLD(features // 4,
                features // 2,
                1,
                activate=None,
                se_block=se_block),
            nn.Dropout(drop_rate) if drop_rate > 0 else EmptyLayer(),
        )

    def forward(self, x):
        x = torch.cat([x[:, :self.features], self.block(x)], 1)
        return x


if __name__ == "__main__":
    a = torch.ones([2, 1024, 224, 224])
    print(DenseBlock(1024, 1024)(a).shape)