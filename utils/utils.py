import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from .modules.nn import FocalBCELoss
from .modules.datasets import VOC_COLORMAP

CE = nn.CrossEntropyLoss()
BCE = nn.BCEWithLogitsLoss()
focal = FocalBCELoss()
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def compute_loss(outputs, targets):
    loss = focal(outputs.softmax(1), targets)
    return loss


def show_batch(save_path, inputs, targets):
    imgs = inputs.clone()[:8]
    segs = targets.clone()[:8].max(1)[1]
    imgs *= 255.
    imgs = imgs.clamp(0, 255).permute(0, 2, 3, 1).byte().numpy()[:, :, :, ::-1]
    segs = segs.numpy()
    seg_rgb = np.zeros_like(imgs, dtype=np.uint8)
    for ci, color in enumerate(VOC_COLORMAP):
        seg_rgb[segs == ci] = color
    segs = seg_rgb
    imgs = imgs.reshape(-1, imgs.shape[2], imgs.shape[3])
    segs = segs.reshape(-1, segs.shape[2], segs.shape[3])

    save_img = np.concatenate([imgs, segs], 1)
    cv2.imwrite(save_path, save_img)

def load_checkpoint(weights, model, optimizer, adam):
    state_dict = torch.load(weights, map_location=device)
    if adam:
        if 'adam' in state_dict:
            optimizer.load_state_dict(state_dict['adam'])
    else:
        if 'sgd' in state_dict:
            optimizer.load_state_dict(state_dict['sgd'])
    best_miou = state_dict['miou']
    best_loss = state_dict['loss']
    epoch = state_dict['epoch']
    model.load_state_dict(state_dict['model'], strict=False)
    return best_loss, best_miou, epoch


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