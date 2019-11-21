import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from .modules.nn import FocalBCELoss

CE = nn.CrossEntropyLoss()
BCE = nn.BCEWithLogitsLoss()
focal = FocalBCELoss()
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def compute_loss(outputs, targets):
    loss = focal(outputs.softmax(1), targets)
    return loss


def show_batch(save_path, inputs, targets, classes):
    imgs = inputs.clone()[:8]
    segs = targets.clone()[:8].max(1)[1]
    imgs *= 255.
    imgs = imgs.clamp(0, 255).permute(0, 2, 3, 1).byte().numpy()[:, :, :, ::-1]
    segs = segs.numpy()
    seg_rgb = np.zeros_like(imgs, dtype=np.uint8)
    for ci, (cn, color) in enumerate(classes):
        seg_rgb[segs == ci] = color
    segs = seg_rgb
    imgs = imgs.reshape(-1, imgs.shape[2], imgs.shape[3])
    segs = segs.reshape(-1, segs.shape[2], segs.shape[3])

    save_img = np.concatenate([imgs, segs], 1)
    cv2.imwrite(save_path, save_img)
