import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.blocks import bn, relu, ResBlock, BLD, Aspp, AsppPooling, DenseBlock, XceptionBackbone, DBL
import math


class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabV3Plus, self).__init__()
        self.backbone = XceptionBackbone(16)
        self.aspp = Aspp(2048, 256, [12, 24, 36])
        self.low_conv = DBL(128, 48)
        self.cls_conv = nn.Sequential(
            DBL(304, 256, 3),
            nn.Dropout(0.5),
            DBL(256, 256, 3),
            nn.Dropout(0.1),
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

    def forward(self, x, var=None):
        x = self.backbone.conv1(x)
        x = self.backbone.block1(x)
        low = self.low_conv(x)
        x = self.backbone.block2(x)
        x = self.backbone.block3(x)
        x = self.backbone.middle_flow(x)
        x = self.backbone.exit_flow(x)
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
        self.backbone = XceptionBackbone(16)
        self.up_conv1 = DBL(2048, 256)
        self.up_conv2 = DBL(512, 128)
        self.up_conv3 = DBL(256, 64)
        self.cls_conv = nn.Sequential(
            DBL(128, 128),
            nn.Dropout(0.5),
            DBL(128, 128),
            nn.Dropout(0.1),
            nn.Conv2d(128, num_classes, 1),
        )
        # init weight and bias
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, var=None):
        x = self.backbone.conv1(x)
        x1 = x
        x = self.backbone.block1(x)
        x2 = x
        x = self.backbone.block2(x)
        x3 = x
        x = self.backbone.block3(x)
        x = self.backbone.middle_flow(x)
        x = self.backbone.exit_flow(x)
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
        x = self.up_conv3(x)
        x = F.interpolate(x,
                          scale_factor=2,
                          mode='bilinear',
                          align_corners=True)
        x = torch.cat([x, x1], 1)
        x = self.cls_conv(x)
        x = F.interpolate(x,
                          scale_factor=2,
                          mode='bilinear',
                          align_corners=True)
        return x


if __name__ == "__main__":
    a = torch.ones([2, 3, 224, 224])
    print(UNet(8)(a).shape)
