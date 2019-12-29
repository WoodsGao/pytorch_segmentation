import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_modules.nn import Aspp, Swish, ConvNormAct, SeparableConv
from pytorch_modules.backbones import BasicModel
from pytorch_modules.backbones.efficientnet import efficientnet
from pytorch_modules.backbones.resnet import resnet50
import math


class DeepLabV3Plus(BasicModel):
    def __init__(self, num_classes):
        super(DeepLabV3Plus, self).__init__()
        self.stages = efficientnet(
            4,
            pretrained=True,
            replace_stride_with_dilation=[False, False, True]).stages
        self.aspp = Aspp(448, 256, [6, 12, 18])
        self.cls_conv = SeparableConv(288, num_classes)
        # init weight and bias
        self.initialize_weights(self.aspp)
        # self.initialize_weights(self.project)
        self.initialize_weights(self.cls_conv)

    def forward(self, x):
        # freeze 0-3bn
        self.freeze_bn(self.stages[:5])
        # freeze 0,1 stage
        self.freeze(self.stages[0:2])

        x = self.stages[0](x)
        x = self.stages[1](x)
        # low = self.project(x)
        low = x
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


class UNet(BasicModel):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.stages = efficientnet(
            4,
            pretrained=True,
            replace_stride_with_dilation=[False, False, True]).stages
        self.up_convs = nn.ModuleList([
            ConvNormAct(448, 128),
            ConvNormAct(184, 112),
            ConvNormAct(144, 96)
        ])
        self.cls_conv = nn.Conv2d(120, num_classes, 3, padding=1)
        # freeze first stage
        self.freeze(self.stages[0])
        # init weight and bias
        self.initialize_weights(self.up_convs)
        self.initialize_weights(self.cls_conv)

    def forward(self, x):
        # freeze bn
        self.freeze_bn(self.stages)

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
    model = DeepLabV3Plus(30)
    o = model(a)
    model.train()
    print(o.shape)
    o.mean().backward()
