import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_modules.backbones import resnet50
from pytorch_modules.nn import Aspp, ConvNormAct, Mish, SeparableConv
from pytorch_modules.utils import initialize_weights


class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabV3Plus, self).__init__()
        self.stages = resnet50(
            pretrained=True,
            replace_stride_with_dilation=[False, False, True]).stages
        self.project = ConvNormAct(256, 128, 1)
        self.aspp = Aspp(2048, 256, [6, 12, 18])
        self.cls_conv = nn.Conv2d(384, num_classes, 3, padding=1)
        # init weight and bias
        initialize_weights(self.aspp)
        initialize_weights(self.project)
        initialize_weights(self.cls_conv)

    def forward(self, x):
        x = self.stages[0](x)
        x = self.stages[1](x)
        low = self.project(x)
        x = self.stages[2](x)
        x = self.stages[3](x)
        x = self.stages[4](x)

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


class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.stages = resnet50(
            pretrained=True,
            replace_stride_with_dilation=[False, False, True]).stages
        self.up_convs = nn.ModuleList([
            ConvNormAct(2048, 512),
            ConvNormAct(1024, 256),
            ConvNormAct(512, 128)
        ])
        self.cls_conv = nn.Conv2d(192, num_classes, 3, padding=1)
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


if __name__ == "__main__":
    a = torch.ones([2, 3, 224, 224])
    model = UNet(30)
    o = model(a)
    model.train()
    print(o.shape)
    o.mean().backward()
