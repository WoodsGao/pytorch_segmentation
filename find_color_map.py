import argparse
import os
import os.path as osp

import cv2
import numpy as np
from tqdm import tqdm

from pytorch_modules.utils import IMG_EXT


def run(data_dir):
    classes = np.zeros([0, 3])
    names = os.listdir(osp.join(data_dir, 'labels'))
    names = [name for name in names if osp.splitext(name)[1] in IMG_EXT]
    for name in tqdm(names):
        img = cv2.imread(osp.join(data_dir, 'labels', name)).reshape(-1, 3)
        classes = np.unique(np.concatenate(
            [classes, np.unique(img, axis=0)], 0),
                            axis=0)
    output = []
    for ci, c in enumerate(classes):
        output.append(', '.join(['%d'] * 3) % (*c))
    output = '\n'.join(output)
    with open(osp.join(data_dir, 'classes.names'), 'w') as f:
        f.write(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    args = parser.parse_args()
    run(args.data_dir)
