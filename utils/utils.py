import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from pytorch_modules.nn import FocalBCELoss
from .datasets import VOC_COLORMAP
from .criterions import RectLoss

CE = nn.CrossEntropyLoss()
BCE = nn.BCEWithLogitsLoss()
focal = FocalBCELoss()


def compute_loss(outputs, targets, model):
    outputs = F.interpolate(outputs, (targets.size(1), targets.size(2)),
                            mode='bilinear',
                            align_corners=True)
    loss = CE(outputs, targets)
    # rect_loss = RectLoss(reduction='none')
    # rloss = rect_loss(outputs)[:, 1].mean()
    return loss # + rloss


def show_batch(inputs, targets):
    imgs = inputs.clone()[:8]
    segs = targets.clone()[:8]
    imgs *= torch.FloatTensor([58.395, 57.12,
                               57.375]).reshape(1, 3, 1, 1).to(imgs.device)
    imgs += torch.FloatTensor([123.675, 116.28,
                               103.53]).reshape(1, 3, 1, 1).to(imgs.device)

    imgs = imgs.clamp(0, 255).permute(0, 2, 3,
                                      1).byte().cpu().numpy()[..., ::-1]
    imgs = np.ascontiguousarray(imgs)
    segs = segs.cpu().numpy()
    segs = np.ascontiguousarray(segs)
    seg_rgb = np.zeros_like(imgs, dtype=np.uint8)
    for ci, color in enumerate(VOC_COLORMAP):
        seg_rgb[segs == ci] = color
    segs = seg_rgb
    imgs = imgs.reshape(-1, imgs.shape[2], imgs.shape[3])
    segs = segs.reshape(-1, segs.shape[2], segs.shape[3])

    save_img = np.concatenate([imgs, segs], 1)
    cv2.imwrite('batch.png', save_img)


def compute_metrics(tp, fn, fp):
    union = tp + fp + fn
    union[union <= 0] = 1
    miou = tp / union
    T = tp + fn
    P = tp + fp
    P[P <= 0] = 1
    P = tp / P
    R = tp + fn
    R[R <= 0] = 1
    R = tp / R
    F1 = (2 * tp + fp + fn)
    F1[F1 <= 0] = 1
    F1 = 2 * tp / F1
    return T, P, R, miou, F1
