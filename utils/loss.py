import math
import torch
import torch.nn as nn
import torch.optim as optim


class FocalBCELoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, weight=None):
        super(FocalBCELoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        if weight is not None:
            self.weight = weight.unsqueeze(0).unsqueeze(2)
        else:
            self.weight = 1

    def forward(self, y_pred, y_true):
        b = y_pred.size(0)
        c = y_pred.size(1)
        y_pred = y_pred.view(b, c, -1)
        y_true = y_true.view(b, c, -1)
        y_pred = torch.clamp(y_pred, 1e-5, 1 - 1e-5)
        a = self.alpha
        g = self.gamma
        loss = - a * torch.pow((1 - y_pred), g) * y_true * torch.log(y_pred) - \
            (1 - a) * torch.pow(y_pred, g) * (1 - y_true) * torch.log(1 - y_pred)
        loss *= self.weight
        return loss.mean(1)
