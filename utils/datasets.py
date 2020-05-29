import json
import os
import os.path as osp
import random

import cv2
import imgaug as ia
import numpy as np
import torch
import torch.nn.functional as F
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from imgaug.augmentables.polys import Polygon, PolygonsOnImage

from pytorch_modules.utils import IMG_EXT

# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
sometimes = lambda aug: iaa.Sometimes(0.5, aug)

# Define our sequence of augmentation steps that will be applied to every image
# All augmenters with per_channel=0.5 will sample one value _per image_
# in 50% of all cases. In all other cases they will sample new values
# _per channel_.

TRAIN_AUGS = iaa.Sequential(
    [
        # apply the following augmenters to most images
        iaa.Fliplr(0.5),  # horizontally flip 50% of all images
        iaa.Flipud(0.2),  # vertically flip 20% of all images
        # crop images by -5% to 10% of their height/width
        sometimes(
            iaa.CropAndPad(
                percent=(-0.05, 0.1), pad_mode=ia.ALL, pad_cval=(0, 255))),
        sometimes(
            iaa.Affine(
                scale={
                    "x": (0.8, 1.2),
                    "y": (0.8, 1.2)
                },  # scale images to 80-120% of their size, individually per axis
                translate_percent={
                    "x": (-0.2, 0.2),
                    "y": (-0.2, 0.2)
                },  # translate by -20 to +20 percent (per axis)
                rotate=(-90, 90),  # rotate by -45 to +45 degrees
                shear=(-16, 16),  # shear by -16 to +16 degrees
                order=[
                    0,
                    1
                ],  # use nearest neighbour or bilinear interpolation (fast)
                cval=(
                    0,
                    255),  # if mode is constant, use a cval between 0 and 255
                mode=ia.
                ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
        # execute 0 to 5 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        iaa.SomeOf(
            (0, 5),
            [
                sometimes(
                    iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))
                ),  # convert images into their superpixel representation
                iaa.OneOf([
                    iaa.GaussianBlur(
                        (0,
                         3.0)),  # blur images with a sigma between 0 and 3.0
                    iaa.AverageBlur(
                        k=(2, 7)
                    ),  # blur image using local means with kernel sizes between 2 and 7
                    iaa.MedianBlur(
                        k=(3, 11)
                    ),  # blur image using local medians with kernel sizes between 2 and 7
                ]),
                iaa.Sharpen(alpha=(0, 1.0),
                            lightness=(0.75, 1.5)),  # sharpen images
                iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),  # emboss images
                # search either for all edges or for directed edges,
                # blend the result with the original image using a blobby mask
                iaa.BlendAlphaSimplexNoise(
                    iaa.OneOf([
                        iaa.EdgeDetect(alpha=(0.5, 1.0)),
                        iaa.DirectedEdgeDetect(alpha=(0.5, 1.0),
                                               direction=(0.0, 1.0)),
                    ])),
                iaa.AdditiveGaussianNoise(
                    loc=0, scale=(0.0, 0.05 * 255),
                    per_channel=0.5),  # add gaussian noise to images
                iaa.OneOf([
                    iaa.Dropout((0.01, 0.1), per_channel=0.5
                                ),  # randomly remove up to 10% of the pixels
                    iaa.CoarseDropout((0.03, 0.15),
                                      size_percent=(0.02, 0.05),
                                      per_channel=0.2),
                ]),
                iaa.Invert(0.05, per_channel=True),  # invert color channels
                iaa.Add(
                    (-10, 10), per_channel=0.5
                ),  # change brightness of images (by -10 to 10 of original value)
                iaa.AddToHueAndSaturation(
                    (-20, 20)),  # change hue and saturation
                # either change the brightness of the whole image (sometimes
                # per channel) or change the brightness of subareas
                iaa.OneOf([
                    iaa.Multiply((0.5, 1.5), per_channel=0.5),
                    iaa.BlendAlphaFrequencyNoise(
                        exponent=(-4, 0),
                        foreground=iaa.Multiply((0.5, 1.5), per_channel=True),
                        background=iaa.LinearContrast((0.5, 2.0)))
                ]),
                iaa.LinearContrast(
                    (0.5, 2.0),
                    per_channel=0.5),  # improve or worsen the contrast
                iaa.Grayscale(alpha=(0.0, 1.0)),
                sometimes(
                    iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
                ),  # move pixels locally around (with random strengths)
                sometimes(iaa.PiecewiseAffine(scale=(
                    0.01, 0.05))),  # sometimes move parts of the image around
                sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
            ],
            random_order=True)
    ],
    random_order=True)


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
            resize = iaa.Sequential([
                iaa.Resize({
                    'width': int(w * scale),
                    'height': int(h * scale)
                }),
                iaa.PadToFixedSize(*self.img_size,
                                  pad_cval=[123.675, 116.28, 103.53],
                                  position='center')
            ])
        else:
            resize = iaa.Resize({
                'width': self.img_size[0],
                'height': self.img_size[1]
            })

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


class CocoInstance(BasicDataset):
    def __init__(self,
                 path,
                 img_size=224,
                 augments=TRAIN_AUGS,
                 multi_scale=False,
                 rect=False):
        super(CocoInstance, self).__init__(img_size=img_size,
                                           augments=None,
                                           multi_scale=multi_scale,
                                           rect=rect)
        with open(path, 'r') as f:
            self.coco = json.loads(f.read())
        self.img_root = osp.dirname(path)
        self.det_augments = augments
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
            img_info = self.coco['images'][idx]
            if max(ann['segmentation'][0][::2]) >= img_info['width'] or max(ann['segmentation'][0][1::2]) >= img_info['height'] or min(ann['segmentation'][0]) < 0:
                continue
            idx = img_ids.index(idx)
            img_anns[idx].append(ann)
        self.data = list(zip(img_paths, img_anns))
        self.data = [d for d in self.data if len(d[1]) > 0]


    def get_data(self, idx):
        img = cv2.imread(self.data[idx][0])
        anns = self.data[idx][1]
        polygons = []
        for ann in anns:
            polygons.append(
                Polygon(
                    np.float32(ann['segmentation']).reshape(-1, 2),
                    ann['category_id']))
        polygons = PolygonsOnImage(polygons, img.shape)

        h, w, c = img.shape

        # augment
        if self.det_augments is not None:
            augments = self.det_augments.to_deterministic()
            img = augments.augment_image(img)
            polygons = augments.augment_polygons(polygons)
        
        for i in range(len(polygons.polygons)):
            polygon = random.choice(polygons.polygons)
            p = polygon.exterior.reshape(-1, 2).astype(np.int32)
            # p = p.clip(0, 1)
            if p[:, 0].min() < 0 or p[:, 1].min() < 0 or p[:, 0].max(
            ) >= img.shape[1] or p[:, 1].max() >= img.shape[0] or p[:, 0].max(
            ) - p[:, 0].min() < 50 or p[:, 1].max() - p[:, 1].min() < 50:
                continue
            break
        x1 = random.randint(p[:, 0].min() - 100, p[:, 0].min())
        x1 = max(0, x1)
        x2 = random.randint(p[:, 0].max(), p[:, 0].max() + 100)
        x2 = min(img.shape[1], x2)
        y1 = random.randint(p[:, 1].min() - 100, p[:, 1].min())
        y1 = max(0, y1)
        y2 = random.randint(p[:, 1].max(), p[:, 1].max() + 100)
        y2 = min(img.shape[0], y2)

        cimg = img[y1:y2, x1:x2]
        if cimg.size > 0:
            img = cimg
        p[:, 0] -= x1
        p[:, 1] -= y1
        seg = np.zeros([img.shape[0], img.shape[1]], dtype=np.uint8)
        seg = cv2.fillPoly(seg, [p], polygon.label + 1, 0)
        seg = SegmentationMapsOnImage(seg, shape=img.shape)
        return img, seg
