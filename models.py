import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.modules.nn import Aspp, AsppPooling, Swish, CNS
from utils.modules.backbones import BasicModel, DenseNet, ResNet, EfficientNetB2, EfficientNetB4
import math


class DeepLabV3Plus(BasicModel):
    def __init__(self, num_classes):
        super(DeepLabV3Plus, self).__init__()
        self.backbone = EfficientNetB4()
        self.high_aspp = Aspp(448, 128, [3, 6, 9])
        self.middle_aspp = Aspp(56, 64, [12, 24, 36])
        self.middle_conv = CNS(128 + 64, 96)
        self.low_conv = CNS(32, 64, 1)
        self.cls_conv = nn.Sequential(
            nn.Conv2d(96 + 64, num_classes, 3, padding=1))
        # init weight and bias
        self.init()
        self.weight_standard()

    def forward(self, x):
        x = self.backbone.block1(x)
        x = self.backbone.block2(x)
        low = self.low_conv(x)
        x = self.backbone.block3(x)
        middle = self.middle_aspp(x)
        x = self.backbone.block4(x)
        x = self.backbone.block5(x)
        x = self.high_aspp(x)
        x = F.interpolate(x,
                          scale_factor=4,
                          mode='bilinear',
                          align_corners=True)
        x = torch.cat([x, middle], 1)
        x = self.middle_conv(x)
        x = F.interpolate(x,
                          scale_factor=2,
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
        self.backbone = EfficientNetB4()
        self.up_conv1 = CNS(448, 160)
        self.up_conv2 = CNS(320, 56)
        self.up_conv3 = CNS(112, 32)
        self.up_conv4 = CNS(64, 24)
        self.cls_conv = nn.Conv2d(48, num_classes, 3, padding=1)
        # init weight and bias
        self.init()
        self.weight_standard()

    def forward(self, x):
        x = self.backbone.block1(x)
        x1 = x
        x = self.backbone.block2(x)
        x2 = x
        x = self.backbone.block3(x)
        x3 = x
        x = self.backbone.block4(x)
        x4 = x
        x = self.backbone.block5(x)
        x = self.up_conv1(x)
        x = F.interpolate(x,
                          scale_factor=2,
                          mode='bilinear',
                          align_corners=True)
        x = torch.cat([x, x4], 1)
        x = self.up_conv2(x)
        x = F.interpolate(x,
                          scale_factor=2,
                          mode='bilinear',
                          align_corners=True)
        x = torch.cat([x, x3], 1)
        x = self.up_conv3(x)
        x = F.interpolate(x,
                          scale_factor=2,
                          mode='bilinear',
                          align_corners=True)
        x = torch.cat([x, x2], 1)
        x = self.up_conv4(x)
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
