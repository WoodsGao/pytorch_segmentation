import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.blocks import bn, relu, ResBlock, BLD, Aspp, AsppPooling
import math


class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabV3Plus, self).__init__()
        # full pre-activation
        self.conv1 = nn.Conv2d(3, 32, 7, 1, 3)
        self.block1 = nn.Sequential(ResBlock(32, 64, stride=2))
        self.block2 = nn.Sequential(
            ResBlock(64, 128, stride=2),
            ResBlock(128, 128),
            ResBlock(128, 128, dilation=6),
            ResBlock(128, 128),
        )
        self.block3 = nn.Sequential(
            ResBlock(128, 256, stride=2),
            ResBlock(256, 256),
            ResBlock(256, 256, dilation=6),
            ResBlock(256, 256),
            ResBlock(256, 256, dilation=12),
            ResBlock(256, 256),
            ResBlock(256, 256, dilation=18),
            ResBlock(256, 256),
        )
        self.block4 = nn.Sequential(
            ResBlock(256, 512, stride=2),
            ResBlock(512, 512),
            ResBlock(512, 512, dilation=6),
            ResBlock(512, 512),
            ResBlock(512, 512, dilation=12),
            ResBlock(512, 512),
            ResBlock(512, 512, dilation=18),
            ResBlock(512, 512),
            ResBlock(512, 512, dilation=30),
            ResBlock(512, 512),
        )
        self.aspp_pooling1 = AsppPooling(512, 512)
        self.up_conv1 = nn.Sequential(
            ResBlock(1024, 512),
            ResBlock(512, 256, dilation=6),
            ResBlock(256, 256),
        )
        self.aspp_pooling2 = AsppPooling(256, 256)
        self.up_conv2 = nn.Sequential(
            ResBlock(768, 512),
            ResBlock(512, 256, dilation=6),
            ResBlock(256, 128),
        )
        self.aspp_pooling3 = AsppPooling(128, 128)
        self.up_conv3 = nn.Sequential(
            ResBlock(384, 256),
            ResBlock(256, 128, dilation=6),
            ResBlock(128, 64),
            nn.Dropout(0.5),
            bn(64),
            relu,
            nn.Conv2d(64, 1, 1),
        )
        self.cls_conv = nn.Sequential(
            ResBlock(128, 128),
            nn.Dropout(0.5),
            bn(128),
            relu,
            nn.Conv2d(128, num_classes - 1, 1),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        feat1 = x
        x = self.block2(x)
        feat2 = x
        x = self.block3(x)
        feat3 = x
        x = self.block4(x)

        x = torch.cat([x, self.aspp_pooling1(x)], 1)
        x = self.up_conv1(x)
        x = F.interpolate(x,
                          scale_factor=2,
                          mode='bilinear',
                          align_corners=True)

        x = torch.cat([x, feat3, self.aspp_pooling2(feat3)], 1)
        x = self.up_conv2(x)
        cls_mask = self.cls_conv(x)
        cls_mask = F.interpolate(cls_mask,
                                 scale_factor=8,
                                 mode='bilinear',
                                 align_corners=True)
        x = F.interpolate(x,
                          scale_factor=2,
                          mode='bilinear',
                          align_corners=True)

        x = torch.cat([x, feat2, self.aspp_pooling3(feat2)], 1)
        x = self.up_conv3(x)

        x = F.interpolate(x,
                          scale_factor=4,
                          mode='bilinear',
                          align_corners=True)
        return x.sigmoid(), cls_mask


if __name__ == "__main__":
    a = torch.ones([2, 3, 224, 224])
    print(DeepLabV3Plus(8)(a)[1].shape)
