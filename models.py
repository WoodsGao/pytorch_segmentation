import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.modules.nn import Aspp, AsppPooling, Swish, CNS
from utils.modules.backbones import DenseNet, ResNet, EfficientNetB2, EfficientNetB4
import math


class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabV3Plus, self).__init__()
        self.backbone = EfficientNetB2(16)
        self.aspp = Aspp(352, 128, [6, 18, 36])
        self.cls_conv = nn.Conv2d(152, num_classes, 3, padding=1)
        # init weight and bias
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.backbone.block1(x)
        x = self.backbone.block2(x)
        low = x
        x = self.backbone.block3(x)
        x = self.backbone.block4(x)
        x = self.backbone.block5(x)
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
        self.backbone = EfficientNetB2(16)
        self.up_conv1 = CNS(352, 96)
        self.up_conv2 = CNS(96 + 48, 48)
        self.cls_conv = nn.Conv2d(48 + 24, num_classes, 1)
        # init weight and bias
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.backbone.block1(x)
        x = self.backbone.block2(x)
        x2 = x
        x = self.backbone.block3(x)
        x3 = x
        x = self.backbone.block4(x)
        x = self.backbone.block5(x)
        x = self.up_conv1(x)
        x = F.interpolate(x,
                          scale_factor=2,
                          mode='bilinear',
                          align_corners=True)
        x = torch.cat([x, x3], 1)
        x = self.up_conv2(x)
        x = F.interpolate(x,
                          scale_factor=2,
                          mode='bilinear',
                          align_corners=True)
        x = torch.cat([x, x2], 1)
        x = self.cls_conv(x)
        x = F.interpolate(x,
                          scale_factor=4,
                          mode='bilinear',
                          align_corners=True)
        return x


if __name__ == "__main__":
    a = torch.ones([2, 3, 224, 224])
    o =DeepLabV3Plus(30)(a)
    print(o.shape)
    o.mean().backward()
