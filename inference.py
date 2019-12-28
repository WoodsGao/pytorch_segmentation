import os
import os.path as osp
import argparse
from tqdm import tqdm
import numpy as np
import cv2
import torch
from utils.models import DeepLabV3Plus
from utils.inference import inference
from utils.datasets import VOC_COLORMAP
from pytorch_modules.utils import device, IMG_EXT


def run(img_dir='data/samples',
        img_size=(512, 512),
        num_classes=21,
        output_dir='outputs',
        weights='weights/best.pt'):
    os.makedirs(output_dir, exist_ok=True)
    model = DeepLabV3Plus(num_classes)
    state_dict = torch.load(weights, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    model = model.to(device)
    model.eval()
    names = [n for n in os.listdir(img_dir) if osp.splitext(n)[1] in IMG_EXT]
    for name in tqdm(names):
        path = osp.join(img_dir, name)
        img = cv2.imread(path)
        segmap = inference(model, [img], img_size)[0]
        seg = np.zeros(img.shape, dtype=np.uint8)
        for ci, color in enumerate(VOC_COLORMAP):
            seg[segmap == ci] = color
        cv2.imwrite(osp.join(output_dir, osp.splitext(name)[0] + '.png'), seg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, default='data/samples')
    parser.add_argument('--dst', type=str, default='outputs')
    parser.add_argument('--img-size', type=str, default='512')
    parser.add_argument('--num-classes', type=int, default=21)
    parser.add_argument('--weights', type=str, default='weights/best.pt')
    opt = parser.parse_args()
    print(opt)

    img_size = opt.img_size.split(',')
    assert len(img_size) in [1, 2]
    if len(img_size) == 1:
        img_size = [int(img_size[0])] * 2
    else:
        img_size = [int(x) for x in img_size]

    inference(opt.src, tuple(img_size), opt.num_classes, opt.dst, opt.weights)
