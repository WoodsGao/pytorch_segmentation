import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_modules.backbones import resnet50
from pytorch_modules.nn import ConvNormAct
from pytorch_modules.utils import initialize_weights

from .aspp import ASPP


class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabV3Plus, self).__init__()
        self.backbone = resnet50(
            pretrained=True, replace_stride_with_dilation=[False, False,
                                                           True])
        self.project = ConvNormAct(256, 128, 1)
        self.aspp = ASPP(2048, 256, [6, 12, 18])
        self.cls_conv = nn.Conv2d(384, num_classes, 3, padding=1)
        # init weight and bias
        initialize_weights(self.aspp)
        initialize_weights(self.project)
        initialize_weights(self.cls_conv)

    def forward(self, x):
        features = self.backbone(x)
        low = features[1]
        low = self.project(low)
        x = features[-1]
        x = self.aspp(x)
        x = F.interpolate(x,
                          scale_factor=4,
                          mode='bilinear',
                          align_corners=True)
        x = torch.cat([x, low], 1)
        x = self.cls_conv(x)
        x = F.interpolate(x,
                          scale_factor=4,
                          mode='bilinear',
                          align_corners=True)
        return x
