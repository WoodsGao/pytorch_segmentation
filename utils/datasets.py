import os
import os.path as osp
import random
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from imgaug import augmenters as ia
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from pytorch_modules.utils import IMG_EXT

TRAIN_AUGS = ia.SomeOf(
    [0, 3],
    [
        ia.WithColorspace(
            to_colorspace='HSV',
            from_colorspace='RGB',
            children=ia.Sequential([
                ia.WithChannels(
                    0,
                    ia.SomeOf([0, None],
                              [ia.Add((-10, 10)),
                               ia.Multiply((0.95, 1.05))],
                              random_state=True)),
                ia.WithChannels(
                    1,
                    ia.SomeOf([0, None],
                              [ia.Add((-50, 50)),
                               ia.Multiply((0.8, 1.2))],
                              random_state=True)),
                ia.WithChannels(
                    2,
                    ia.SomeOf([0, None],
                              [ia.Add((-50, 50)),
                               ia.Multiply((0.8, 1.2))],
                              random_state=True)),
            ])),
        ia.Dropout([0.015, 0.1]),  # drop 5% or 20% of all pixels
        ia.Sharpen((0.0, 1.0)),  # sharpen the image
        ia.Affine(
            scale=(0.8, 1.2),
            translate_percent=(-0.1, 0.1),
            rotate=(-15, 15),
            shear=(-0.1,
                   0.1)),  # rotate by -45 to 45 degrees (affects heatmaps)
        ia.ElasticTransformation(
            alpha=(0, 10),
            sigma=(0, 10)),  # apply water effect (affects heatmaps)
        ia.PiecewiseAffine(scale=(0, 0.03), nb_rows=(2, 6), nb_cols=(2, 6)),
        ia.GaussianBlur((0, 3)),
        ia.Fliplr(0.1),
        ia.Flipud(0.1),
        ia.LinearContrast((0.5, 1)),
        ia.AdditiveGaussianNoise(loc=(0, 10), scale=(0, 10))
    ],
    random_state=True)


def voc_colormap(N=256):
    def bitget(val, idx):
        return ((val & (1 << idx)) != 0)

    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= (bitget(c, 0) << 7 - j)
            g |= (bitget(c, 1) << 7 - j)
            b |= (bitget(c, 2) << 7 - j)
            c >>= 3
        # print([r, g, b])
        cmap[i, :] = [b, g, r]
    return cmap


VOC_COLORMAP = voc_colormap(32)


class SegDataset(torch.utils.data.Dataset):
    def __init__(self,
                 path,
                 img_size=224,
                 augments=None,
                 multi_scale=False,
                 colormap=VOC_COLORMAP):
        super(SegDataset, self).__init__()
        self.path = path
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        assert len(img_size) == 2
        self.img_size = img_size
        self.multi_scale = multi_scale
        self.resize = ia.Resize({"height": img_size[1], "width": img_size[0]})
        self.augments = augments
        self.data = []
        self.classes = []
        self.build_data()
        self.data.sort()
        self.colormap = colormap

    def build_data(self):
        data_dir = osp.dirname(self.path)
        with open(osp.join(data_dir, 'classes.names'), 'r') as f:
            self.classes = [c for c in f.read().split('\n') if c]
        image_dir = osp.join(data_dir, 'images')
        label_dir = osp.join(data_dir, 'labels')
        with open(self.path, 'r') as f:
            names = [n for n in f.read().split('\n') if n]
        names = list(set(names))
        self.data = [[
            osp.join(image_dir, name),
            osp.join(label_dir,
                     osp.splitext(name)[0] + '.png')
        ] for name in names if osp.splitext(name)[1] in IMG_EXT]

    def get_item(self, idx):
        img = cv2.imread(self.data[idx][0])
        seg_color = cv2.imread(self.data[idx][1])
        seg = np.zeros([seg_color.shape[0], seg_color.shape[1]],
                       dtype=np.uint8)
        for ci, c in enumerate(self.colormap):
            seg[(seg_color == c).all(2)] = ci
        seg = SegmentationMapsOnImage(seg, shape=img.shape)
        img = img[:, :, ::-1]

        resize = self.resize.to_deterministic()
        img = resize.augment_image(img)
        seg = resize.augment_segmentation_maps(seg)
        # augment
        if self.augments is not None:
            augments = self.augments.to_deterministic()
            img = augments.augment_image(img)
            seg = augments.augment_segmentation_maps(seg)

        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        seg = seg.get_arr()
        # for ci, c in enumerate(self.classes):
        #     seg[seg_args == ci, 1 if ci > 0 else 0] = 1
        return torch.ByteTensor(img), torch.ByteTensor(seg)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.get_item(idx)

    def post_fetch_fn(self, batch):
        imgs, segs = batch
        imgs = imgs.float()
        imgs /= 255.
        if self.multi_scale:
            h = imgs.size(2)
            w = imgs.size(3)
            scale = random.uniform(0.7, 1.5)
            h = int(h * scale / 16) * 16
            w = int(w * scale / 16) * 16
            imgs = F.interpolate(imgs, (h, w))
        return (imgs, segs.long())
