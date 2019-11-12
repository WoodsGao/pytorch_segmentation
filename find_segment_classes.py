import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm


def find_segment_classes(data_dir):
    classes = np.zeros([0, 3])
    names = os.listdir(os.path.join(data_dir, 'labels'))
    names = [
        name for name in names
        if os.path.splitext(name)[1] in ['.jpg', '.jpeg', '.png', '.tiff']
    ]
    for name in tqdm(names):
        img = cv2.imread(os.path.join(data_dir, 'labels', name)).reshape(-1, 3)
        classes = np.unique(np.concatenate(
            [classes, np.unique(img, axis=0)], 0),
                            axis=0)
    output = []
    for ci, c in enumerate(classes):
        output.append(', '.join(['%d'] * 4) % (ci, *c))
    output = '\n'.join(output)
    with open(os.path.join(data_dir, 'classes.names'), 'w') as f:
        f.write(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', default='data/voc')
    args = parser.parse_args()
    find_segment_classes(args.path)