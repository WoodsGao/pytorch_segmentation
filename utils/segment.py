import numpy as np
import cv2
import torch
from pytorch_modules.utils import device


@torch.no_grad()
def segment(model, img, img_size=(64, 64)):
    im0 = img
    img = im0.copy()
    img = cv2.resize(img, img_size)
    img = img[:, :, ::-1]
    img = img.transpose(2, 0, 1)
    img = np.float32([img]) / 255.
    img = torch.FloatTensor(img).to(device)
    pred = model(img)[0]
    return pred.max(0)
