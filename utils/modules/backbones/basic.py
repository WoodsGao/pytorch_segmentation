import math
import torch
import torch.nn as nn


class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()
        self.init()

    def init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        return x
