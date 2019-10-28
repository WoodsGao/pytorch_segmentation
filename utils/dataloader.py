import cv2
import numpy as np
import os
from . import config
from .cv_utils import dataloader


class Dataloader(dataloader.Dataloader):
    def build_data_list(self):
        data_dir = os.path.dirname(self.path)
        with open(os.path.join(data_dir, 'classes.csv'), 'r') as f:
            lines = [l.split(',') for l in f.readlines()]
            lines = [[l[0], np.uint8(l[1:])] for l in lines if len(l) == 4]
            self.classes = lines
        image_dir = os.path.join(data_dir, 'images')
        label_dir = os.path.join(data_dir, 'labels')
        with open(self.path, 'r') as f:
            names = [n for n in f.read().split('\n') if n]
        names = [
            name for name in names
            if os.path.splitext(name)[1] in config.IMG_EXT
        ]
        for name in names:
            self.data_list.append([
                os.path.join(image_dir, name),
                os.path.join(label_dir,
                             os.path.splitext(name)[0] + '.png')
            ])

    def worker(self, message):
        img = cv2.imread(message[0])
        img = cv2.resize(img, (self.scale, self.scale))
        seg_color = cv2.imread(message[1])
        seg = np.zeros([seg_color.shape[0], seg_color.shape[1], len(self.classes)])
        for ci, c in enumerate(self.classes):
            seg[(seg_color == c[1]).all(2), ci] = 1
        seg = cv2.resize(seg, (self.scale, self.scale))
        for aug in self.augments:
            img, _, seg = aug(img, seg=seg)
        seg[seg.sum(2) == 0, 0] = 1
        seg[seg > 0.5] = 1
        seg[seg < 1] = 0
        seg_args = seg.argmax(2)
        for ci, c in enumerate(self.classes):
            seg[seg_args == ci, 1 if ci > 0 else 0] = 1
        return [img, seg]


def show_batch(save_path, messages, scale, classes, augments_list=[]):
    imgs = []
    segs = []
    for message in messages:
        img = cv2.imread(message[0])
        img = cv2.resize(img, (scale, scale))
        seg_color = cv2.imread(message[1])
        seg = np.zeros([seg_color.shape[0], seg_color.shape[1], len(classes)])
        for ci, c in enumerate(classes):
            seg[(seg_color == c[1]).all(2), ci] = 1
        seg = cv2.resize(seg, (scale, scale))
        for aug in augments_list:
            img, _, seg = aug(img, seg=seg)
        seg[seg.sum(2) == 0, 0] = 1
        seg[seg > 0.5] = 1
        seg[seg < 1] = 0
        seg_args = seg.argmax(2)
        seg_color = np.zeros_like(img)
        for ci, c in enumerate(classes):
            seg_color[seg_args == ci] = c[1]
        imgs.append(img)
        segs.append(seg_color)
    imgs = np.concatenate(imgs, 1)
    segs = np.concatenate(segs, 1)
    save_img = np.concatenate([imgs, segs], 0)
    save_img = np.clip(save_img, 0, 255)
    save_img = np.uint8(save_img)
    cv2.imwrite(save_path, save_img)

