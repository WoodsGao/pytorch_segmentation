import math
import torch
import torch.nn as nn
from . import device


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


FOCAL = FocalBCELoss()
CE = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)
BCE = nn.BCELoss(reduction='none')


def compute_loss(outputs, targets, obj_weight=1, cls_weight=1):
    pred_obj, pred_cls = outputs
    true_obj = targets
    true_obj[true_obj > 0] = -1
    true_obj += 1
    true_obj = true_obj.unsqueeze(1)
    obj_loss = FOCAL(pred_obj, true_obj)
    obj_loss = obj_loss.view(obj_loss.size(0), -1).mean(1) * obj_weight
    true_cls = targets - 1
    cls_loss = CE(pred_cls, true_cls)
    cls_loss = cls_loss.view(cls_loss.size(0), -1).mean(1) * cls_weight
    loss = obj_loss + cls_loss
    return loss, obj_loss, cls_loss
