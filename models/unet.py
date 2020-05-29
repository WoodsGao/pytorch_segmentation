import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_modules.backbones import resnet34, resnet50, mobilenet_v2
from pytorch_modules.backbones.mobilenet import InvertedResidual
from pytorch_modules.nn import ConvNormAct
from pytorch_modules.utils import initialize_weights


class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.backbone = mobilenet_v2(
            pretrained=True)
        self.up_convs = nn.ModuleList([
            ConvNormAct(1280, 256),
            ConvNormAct(352, 128),
            ConvNormAct(160, 64)
        ])
        self.cls_conv = nn.Conv2d(88, num_classes, 3, padding=1)
        initialize_weights(self.up_convs)
        initialize_weights(self.cls_conv)

    def forward(self, x):
        x1, x2, x3, x4, x = self.backbone(x)
        x = self.up_convs[0](x)
        x = F.interpolate(x,
                          scale_factor=2,
                          mode='bilinear',
                          align_corners=True)
        x = torch.cat([x, x4], 1)
        x = self.up_convs[1](x)
        x = F.interpolate(x,
                          scale_factor=2,
                          mode='bilinear',
                          align_corners=True)
        x = torch.cat([x, x3], 1)
        x = self.up_convs[2](x)
        x = F.interpolate(x,
                          scale_factor=2,
                          mode='bilinear',
                          align_corners=True)
        x = torch.cat([x, x2], 1)
        x = F.interpolate(x,
                          scale_factor=2,
                          mode='bilinear',
                          align_corners=True)
        x = self.cls_conv(x)
        x = F.interpolate(x,
                          scale_factor=2,
                          mode='bilinear',
                          align_corners=True)
        return x
