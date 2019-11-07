import torch
import torch.nn as nn

CE = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)
BCE = nn.BCEWithLogitsLoss(reduction='none')


def compute_loss(outputs, targets):
    cls_loss = CE(outputs, targets)
    cls_loss = torch.pow(cls_loss, 2) / cls_loss.max()
    cls_loss = cls_loss.mean()
    loss = cls_loss
    return loss
