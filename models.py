import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.modules.nn import Aspp, AsppPooling, Swish, BLD
from utils.modules.backbones import DenseNet, ResNet
import math


class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabV3Plus, self).__init__()
        self.backbone = DenseNet(8)
        self.aspp = Aspp(1024, 256, [12, 24, 36])
        self.low_conv = BLD(128, 64, 1)
        self.cls_conv = nn.Sequential(
            nn.GroupNorm(32, 320),
            Swish(),
            nn.Conv2d(320, num_classes, 3, padding=1),
        )
        # init weight and bias
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, var=None):
        x = self.backbone.block1(x)
        x = self.backbone.block2(x)
        low = self.low_conv(x)
        x = self.backbone.block3(x)
        x = self.backbone.block4(x)
        x = self.backbone.block5(x)
        x = self.aspp(x)
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


class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.backbone = Dense(16)
        self.up_conv1 = BLD(1024, 256)
        self.up_conv2 = BLD(512, 128)
        self.cls_conv = nn.Sequential(
            nn.GroupNorm(32, 256),
            Swish(),
            nn.Conv2d(256, num_classes, 1),
        )
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
    print(DeepLabV3Plus(30)(a).shape)
