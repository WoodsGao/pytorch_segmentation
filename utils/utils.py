import torch
import torch.nn as nn
import numpy as np
import cv2


CE = nn.CrossEntropyLoss(reduction='none')
BCE = nn.BCEWithLogitsLoss(reduction='none')
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def compute_loss(outputs, targets):
    cls_loss = CE(outputs, targets)
    # weights = torch.pow(cls_loss, 1.1)
    # weights = weights / weights.max()
    # cls_loss = cls_loss * weights
    cls_loss = cls_loss.mean()
    loss = cls_loss
    return loss


def show_batch(save_path, inputs, targets, classes):
    inputs = inputs.clone()
    targets = targets.clone()
    imgs = []
    segs = []
    for bi, (img, seg) in enumerate(zip(inputs, targets)):
        img *= 255.
        img.clamp(0, 255)
        img = img.long().numpy().transpose(1, 2, 0)
        seg = seg.numpy()
        seg_rgb = np.zeros_like(img, dtype=np.uint8)
        for ci, (cn, color) in enumerate(classes):
            seg_rgb[seg == ci] = color

        img = img[:, :, ::-1]

        imgs.append(img)
        segs.append(seg_rgb)

    imgs = np.concatenate(imgs, 1)
    segs = np.concatenate(segs, 1)

    save_img = np.concatenate([imgs, segs], 0)
    cv2.imwrite(save_path, save_img)