import argparse
import os
import os.path as osp
import shutil

import cv2
import numpy as np
import torch
from tqdm import tqdm

from models import DeepLabV3Plus
from pytorch_modules.utils import IMG_EXT, device
from utils.datasets import VOC_COLORMAP
from utils.inference import inference


def run(img_dir, output_dir, img_size, num_classes, weights, show):
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)
    model = DeepLabV3Plus(num_classes)
    state_dict = torch.load(weights, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    model = model.to(device)
    model.eval()
    names = [n for n in os.listdir(img_dir) if osp.splitext(n)[1] in IMG_EXT]
    names.sort()
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
    parser.add_argument('img_dir', type=str)
    parser.add_argument('output_dir', type=str)
    parser.add_argument('-s',
                        '--img_size',
                        type=int,
                        nargs=2,
                        default=[320, 320])
    parser.add_argument('-nc', '--num-classes', type=int, default=2)
    parser.add_argument('--weights', type=str, default='weights/best.pt')
    parser.add_argument('--show', action='store_true', help='display results')
    opt = parser.parse_args()
    print(opt)

    run(opt.img_dir, opt.output_dir, opt.img_size, opt.num_classes,
        opt.weights, opt.show)