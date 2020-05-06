import os
import os.path as osp
import json
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
        # ia.WithColorspace(
        #     to_colorspace='HSV',
        #     from_colorspace='RGB',
        #     children=ia.Sequential([
        #         ia.WithChannels(
        #             0,
        #             ia.SomeOf([0, None],
        #                       [ia.Add((-10, 10)),
        #                        ia.Multiply((0.95, 1.05))],
        #                       random_state=True)),
        #         ia.WithChannels(
        #             1,
        #             ia.SomeOf([0, None],
        #                       [ia.Add((-50, 50)),
        #                        ia.Multiply((0.8, 1.2))],
        #                       random_state=True)),
        #         ia.WithChannels(
        #             2,
        #             ia.SomeOf([0, None],
        #                       [ia.Add((-50, 50)),
        #                        ia.Multiply((0.8, 1.2))],
        #                       random_state=True)),
        #     ])),
        ia.Dropout([0.015, 0.1]),  # drop 5% or 20% of all pixels
        ia.Sharpen((0.0, 1.0)),  # sharpen the image
        ia.Affine(
            scale=(0.8, 1.2),
            translate_percent=(-0.1, 0.1),
            rotate=(-15, 15),
            shear=(-0.1,
                   0.1)),  # rotate by -45 to 45 degrees (affects heatmaps)
        # ia.ElasticTransformation(
        #     alpha=(0, 10),
        #     sigma=(0, 10)),  # apply water effect (affects heatmaps)
        # ia.PiecewiseAffine(scale=(0, 0.03), nb_rows=(2, 6), nb_cols=(2, 6)),
        ia.GaussianBlur((0, 3)),
        ia.Fliplr(0.1),
        ia.Flipud(0.1),
        # ia.LinearContrast((0.5, 1)),
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


class BasicDataset(torch.utils.data.Dataset):
    def __init__(self, img_size, augments, multi_scale, rect):
        super(BasicDataset, self).__init__()
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        assert len(img_size) == 2
        self.img_size = img_size
        self.rect = rect
        self.multi_scale = multi_scale
        self.augments = augments
        self.data = []

    def get_data(self, idx):
        return None, None

    def __getitem__(self, idx):
        img, seg = self.get_data(idx)
        img = img[..., ::-1]
        h, w, c = img.shape

        if self.rect:
            scale = min(self.img_size[0] / w, self.img_size[1] / h)
            resize = ia.Sequential([
                ia.Resize((int(w * scale), int(h * scale))),
                ia.PadToFixedSize(*self.img_size)
            ])
        else:
            resize = ia.Resize(self.img_size)

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

        return torch.ByteTensor(img), torch.ByteTensor(seg)

    def __len__(self):
        return len(self.data)

    def post_fetch_fn(self, batch):
        imgs, segs = batch
        imgs = imgs.float()
        imgs -= torch.FloatTensor([123.675, 116.28,
                                   103.53]).reshape(1, 3, 1, 1).to(imgs.device)
        imgs /= torch.FloatTensor([58.395, 57.12,
                                   57.375]).reshape(1, 3, 1, 1).to(imgs.device)
        if self.multi_scale:
            h = imgs.size(2)
            w = imgs.size(3)
            scale = random.uniform(0.7, 1.5)
            h = int(h * scale / 32) * 32
            w = int(w * scale / 32) * 32
            imgs = F.interpolate(imgs, (h, w))
        return (imgs, segs.long())


class SegImgDataset(BasicDataset):
    def __init__(self,
                 path,
                 img_size=224,
                 augments=TRAIN_AUGS,
                 multi_scale=False,
                 rect=False,
                 colormap=VOC_COLORMAP):
        super(SegImgDataset, self).__init__(img_size=img_size,
                                            augments=augments,
                                            multi_scale=multi_scale,
                                            rect=rect)
        self.path = path
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

    def get_data(self, idx):
        img = cv2.imread(self.data[idx][0])
        seg_color = cv2.imread(self.data[idx][1])
        seg = np.zeros([seg_color.shape[0], seg_color.shape[1]],
                       dtype=np.uint8)
        for ci, c in enumerate(self.colormap):
            seg[(seg_color == c).all(2)] = ci
        seg = SegmentationMapsOnImage(seg, shape=img.shape)
        return img, seg


class CocoDataset(BasicDataset):
    def __init__(self,
                 path,
                 img_size=224,
                 augments=TRAIN_AUGS,
                 multi_scale=False,
                 rect=False):
        super(CocoDataset, self).__init__(img_size=img_size,
                                          augments=augments,
                                          multi_scale=multi_scale,
                                          rect=rect)
        with open(path, 'r') as f:
            self.coco = json.loads(f.read())
        self.img_root = osp.dirname(path)
        self.augments = augments
        self.classes = []
        self.build_data()
        self.data.sort()

    def build_data(self):
        img_ids = []
        img_paths = []
        img_anns = []
        self.classes = ['background'
                        ] + [c['name'] for c in self.coco['categories']]
        for img_info in self.coco['images']:
            img_ids.append(img_info['id'])
            img_paths.append(osp.join(self.img_root, img_info['file_name']))
            img_anns.append([])
        for ann in self.coco['annotations']:
            idx = ann['image_id']
            idx = img_ids.index(idx)
            img_anns[idx].append(ann)
        self.data = list(zip(img_paths, img_anns))

    def get_data(self, idx):
        img = cv2.imread(self.data[idx][0])
        anns = self.data[idx][1]
        seg = np.zeros([img.shape[0], img.shape[1]], dtype=np.uint8)
        for ann in anns:
            points = np.int64(ann['segmentation']).reshape(-1, 2)
            seg = cv2.fillPoly(seg, [points], ann['category_id'] + 1, 0)
        seg = SegmentationMapsOnImage(seg, shape=img.shape)
        return img, seg
