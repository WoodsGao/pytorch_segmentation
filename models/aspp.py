import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_modules.nn import ConvNormAct, SeparableConvNormAct


class ASPPPooling(nn.Module):
    def __init__(self, inplanes, planes):
        super(ASPPPooling, self).__init__()
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 ConvNormAct(inplanes, planes, 1))

    def forward(self, x):
        size = x.shape[-2:]
        return F.interpolate(self.gap(x),
                             size=size,
                             mode='bilinear',
                             align_corners=False)


class ASPP(nn.Module):
    def __init__(self, inplanes, planes, atrous_rates=[12, 24, 36]):
        super(ASPP, self).__init__()
        self.blocks = nn.ModuleList(
            [ASPPPooling(inplanes, planes),
             ConvNormAct(inplanes, planes, 1)])
        for rate in atrous_rates:
            self.blocks.append(ConvNormAct(inplanes, planes, dilation=rate))
        self.project = ConvNormAct(planes * len(self.blocks), planes, 1)

    def forward(self, x):
        res = []
        for block in self.blocks:
            res.append(block(x))
        res = torch.cat(res, dim=1)
        return self.project(res)
