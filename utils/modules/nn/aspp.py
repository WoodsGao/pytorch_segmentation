import torch
import torch.nn as nn
import torch.nn.functional as F
from . import CNS, SeparableCNS


class AsppPooling(nn.Module):
    def __init__(self):
        super(AsppPooling, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        size = x.size()[2:]
        pool = self.gap(x)
        out = F.interpolate(pool, size, mode='bilinear', align_corners=True)
        return out


class Aspp(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates, eps=1e-4):
        super(Aspp, self).__init__()
        blocks = [AsppPooling()]
        for rate in atrous_rates:
            blocks.append(
                CNS(in_channels, in_channels, groups=in_channels, dilation=rate))
        self.blocks = nn.ModuleList(blocks)
        self.aspp_weights = nn.Parameter(torch.ones(len(atrous_rates) + 2))
        self.project = CNS(in_channels, out_channels, 1)
        self.eps = eps

    def forward(self, x):
        aspp_weights = self.aspp_weights.relu()
        aspp_weights = aspp_weights / (aspp_weights.sum(0, keepdim=True) +
                                       self.eps)
        output = x * aspp_weights[-1]
        for bi, block in enumerate(self.blocks):
            output += block(x) * aspp_weights[bi]
        output = self.project(output)
        return output
