import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.blocks import bn, relu, ResBlock, BLD, Aspp, AsppPooling, DenseBlock
import math


class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabV3Plus, self).__init__()
        # full pre-activation
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

        self.high_level_block = nn.Sequential(
            ResBlock(512, 512),
            ResBlock(512, 512, dilation=6),
            ResBlock(512, 512),
        )
        self.high2middle = nn.Sequential(ResBlock(512, 256), )
        self.middle_level_block = nn.Sequential(
            ResBlock(256, 256),
            ResBlock(256, 256, dilation=6),
            ResBlock(256, 256),
        )
        self.middle2low = nn.Sequential(ResBlock(256, 128), )
        self.low_level_block = nn.Sequential(
            ResBlock(128, 128),
            ResBlock(128, 128, dilation=6),
            ResBlock(128, 128),
        )

        self.cls_conv = nn.Sequential(bn(256), relu,
                                      nn.Conv2d(256, num_classes - 1, 1))

        self.obj_conv = nn.Sequential(bn(128), relu,
                                      nn.Conv2d(128, 1, 1))

    def forward(self, x, var=None):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        low_level_feat = x
        x = self.block3(x)
        middle_level_feat = x
        x = self.block4(x)
        high_level_feat = x
        high_level_feat = self.high_level_block(high_level_feat)
        high_level_feat = self.high2middle(high_level_feat)
        high_level_feat = F.interpolate(high_level_feat,
                                        scale_factor=2,
                                        mode='bilinear',
                                        align_corners=True)
        middle_level_feat = middle_level_feat + high_level_feat
        middle_level_feat = self.middle_level_block(middle_level_feat)
        cls_mask = self.cls_conv(middle_level_feat)
        cls_mask = F.interpolate(cls_mask,
                                 scale_factor=8,
                                 mode='bilinear',
                                 align_corners=True)

        middle_level_feat = self.middle2low(middle_level_feat)
        middle_level_feat = F.interpolate(middle_level_feat,
                                          scale_factor=2,
                                          mode='bilinear',
                                          align_corners=True)
        low_level_feat = low_level_feat + middle_level_feat
        low_level_feat = self.low_level_block(low_level_feat)
        obj_mask = self.obj_conv(low_level_feat)
        obj_mask = F.interpolate(obj_mask,
                                 scale_factor=4,
                                 mode='bilinear',
                                 align_corners=True)
        return obj_mask.sigmoid(), cls_mask


if __name__ == "__main__":
    a = torch.ones([2, 3, 224, 224])
    print([o.shape for o in DeepLabV3Plus(8)(a)])
