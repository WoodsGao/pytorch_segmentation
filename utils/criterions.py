import cv2
import numpy as np
import torch
import torch.nn as nn


class RectLoss(nn.Module):
    def __init__(self, weight=None, reduction='mean'):
        super(RectLoss, self).__init__()
        if weight is None:
            self.weight = 1
        else:
            self.weight = weight
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, y_pred):
        B = y_pred.size(0)
        C = y_pred.size(1)
        unions = torch.zeros_like(y_pred)
        # y_pred = y_pred.softmax(1)
        pred_label = y_pred.max(1)[1].cpu().numpy().astype(np.uint8)
        for bi in range(B):
            for ci in range(C):
                seg = pred_label[bi].copy()
                seg[seg != ci] = 0
                seg[seg == ci] = 255
                contours = cv2.findContours(seg, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)[-2]
                max_area = 0
                max_contour = None
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > max_area:
                        max_area = area
                        max_contour = contour
                if max_contour is None:
                    continue
                rect = cv2.minAreaRect(max_contour)
                box = cv2.boxPoints(rect)
                box = np.int32(box)
                canvas = np.zeros_like(seg)
                cv2.drawContours(canvas, [box], 0, 1, -1)
                unions[bi, ci] = torch.FloatTensor(canvas).to(y_pred.device)
        loss = self.bce(y_pred, unions)
        loss *= self.weight
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'mean':
            return loss.mean()
                
class LovaszLoss(nn.Module):
    def __init__(self, weight=None, reduction='mean'):
        super(LovaszLoss, self).__init__()
        if weight is None:
            self.weight = 1
        else:
            self.weight = weight.unsqueeze(0)
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        B, C, H, W = y_pred.size()
        ypred = ypred.permute(0, 2, 3,
                              1).contiguous().view(-1,
                                                   C)  # B * H * W, C = P, C
        labels = labels.view(-1)
        if ignore is not None:
            pass
        valid = (labels != ignore)
        vypred = ypred[valid.nonzero().squeeze()]
        vlabels = labels[valid]
        return vypred, vlabels


def lovasz_softmax(probas,
                   labels,
                   classes='present',
                   per_image=False,
                   ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(
            lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0),
                                                lab.unsqueeze(0), ignore),
                                classes=classes)
            for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore),
                                   classes=classes)
    return loss


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float()  # foreground for class c
        if (classes is 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(
            torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    if probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3,
                            1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels
