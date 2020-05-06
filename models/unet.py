import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_modules.backbones import resnet34, resnet50
from pytorch_modules.nn import ConvNormAct
from pytorch_modules.utils import initialize_weights


class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.stages = resnet34(
            pretrained=True, replace_stride_with_dilation=[False, False,
                                                           True]).stages
        self.up_convs = nn.ModuleList([
            ConvNormAct(512, 128),
            ConvNormAct(256, 64),
            ConvNormAct(128, 64)
        ])
        self.cls_conv = nn.Conv2d(128, num_classes, 3, padding=1)
        initialize_weights(self.up_convs)
        initialize_weights(self.cls_conv)

    def forward(self, x):
        x = self.stages[0](x)
        x1 = x
        x = self.stages[1](x)
        x2 = x
        x = self.stages[2](x)
        x3 = x
        x = self.stages[3](x)
        x = self.stages[4](x)
        x = self.up_convs[0](x)
        x = F.interpolate(x,
                          scale_factor=2,
                          mode='bilinear',
                          align_corners=True)
        x = torch.cat([x, x3], 1)
        x = self.up_convs[1](x)
        x = F.interpolate(x,
                          scale_factor=2,
                          mode='bilinear',
                          align_corners=True)
        x = torch.cat([x, x2], 1)
        x = self.up_convs[2](x)
        x = F.interpolate(x,
                          scale_factor=2,
                          mode='bilinear',
                          align_corners=True)
        x = torch.cat([x, x1], 1)
        x = F.interpolate(x,
                          scale_factor=2,
                          mode='bilinear',
                          align_corners=True)
        x = self.cls_conv(x)
        return x
