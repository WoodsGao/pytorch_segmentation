import cv2
import numpy as np
import torch

from pytorch_modules.utils import device


@torch.no_grad()
def inference(model, imgs, img_size=(64, 64)):
    shapes = [img.shape for img in imgs]
    imgs = [
        cv2.resize(img, img_size)[:, :, ::-1].transpose(2, 0, 1).astype(
            np.float32) / 255. for img in imgs
    ]
    imgs = torch.FloatTensor(imgs).to(device)
    preds = model(imgs).softmax(1).cpu().numpy().transpose(0, 2, 3, 1)
    preds = [
        cv2.resize(pred, (shape[1], shape[0])).argmax(2)
        for pred, shape in zip(preds, shapes)
    ]
    return preds
