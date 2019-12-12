import os
import argparse
from tqdm import tqdm
import torch
from utils.models import DeepLabV3Plus
from pytorch_modules.utils import device, IMG_EXT
from pytorch_modules.datasets import VOC_COLORMAP
import numpy as np
import cv2


def inference(img_dir='data/samples',
              img_size=256,
              output_dir='outputs',
              weights='weights/best_miou.pt',
              unet=False):
    os.makedirs(output_dir, exist_ok=True)
    model = DeepLabV3Plus(32)
    model = model.to(device)
    state_dict = torch.load(weights, map_location=device)
    model.load_state_dict(state_dict['model'])
    model.eval()
    names = [
        n for n in os.listdir(img_dir)
        if os.path.splitext(n)[1] in IMG_EXT
    ]
    with torch.no_grad():
        for name in tqdm(names):
            path = os.path.join(img_dir, name)
            img = cv2.imread(path)
            img_shape = img.shape
            h = (img.shape[0] / max(img.shape[:2]) * img_size) // 32
            w = (img.shape[1] / max(img.shape[:2]) * img_size) // 32
            img = cv2.resize(img, (int(w * 32), int(h * 32)))
            img = img[:, :, ::-1]
            img = img.transpose(2, 0, 1)
            img = torch.FloatTensor([img]).to(device) / 255.
            output = model(img)[0].cpu().numpy().transpose(1, 2, 0)
            output = cv2.resize(output, (img_shape[1], img_shape[0]))
            output = output.argmax(2)
            seg = np.zeros(img_shape, dtype=np.uint8)
            for ci, color in enumerate(VOC_COLORMAP):
                seg[output == ci] = color
            cv2.imwrite(os.path.join(output_dir, name), seg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-dir', type=str, default='data/samples')
    parser.add_argument('--output-dir', type=str, default='outputs')
    parser.add_argument('--img-size', type=int, default=256)
    parser.add_argument('--weights', type=str, default='weights/best.pt')
    opt = parser.parse_args()
    inference(opt.img_dir,
              opt.img_size,
              opt.output_dir,
              opt.weights)
