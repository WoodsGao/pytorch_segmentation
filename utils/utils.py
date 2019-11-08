import torch
import torch.nn as nn

CE = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)
BCE = nn.BCEWithLogitsLoss(reduction='none')
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def compute_loss(outputs, targets):
    cls_loss = CE(outputs, targets)
    max_loss = cls_loss.max()
    cls_loss = torch.pow(cls_loss, 2) / max_loss
    cls_loss = cls_loss.mean()
    loss = cls_loss
    return loss
