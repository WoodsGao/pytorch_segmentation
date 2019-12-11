import math
import torch
import torch.nn as nn
from . import BasicModel
from ..nn import CNS, SeparableCNS


class MiniNet(BasicModel):
    def __init__(self):
        super(MiniNet, self).__init__()
        self.block1 = CNS(3, 32, 7, stride=2)
        self.block2 = SeparableCNS(32, 64, stride=2)
        self.block3 = SeparableCNS(64, 128, stride=2)
        self.block4 = SeparableCNS(128, 256, stride=2)
        self.block5 = SeparableCNS(256, 512, stride=2)
        self.init()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x
