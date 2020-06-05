import argparse
import os
import os.path as osp

import find_color_map


def voc2dataset(data_dir):
    cmd = [
        'rm data/voc', 'mkdir data', 'mkdir data/voc', 'cp {} data/voc/',
        'cp {} data/voc/', 'mkdir data/voc/images data/voc/labels',
        'cp {}/* data/voc/images', 'cp {}/* data/voc/labels'
    ]
    os.system(' && '.join(cmd).format(
        osp.join(data_dir, 'VOC2012/ImageSets/Segmentation/train.txt'),
        osp.join(data_dir, 'VOC2012/ImageSets/Segmentation/valid.txt'),
        osp.join(data_dir, 'VOC2012/JPEGImages'),
        osp.join(data_dir, 'VOC2012/SegmentationClass')))
    with open('data/voc/train.txt', 'r') as f:
        lines = f.read().split('\n')
    lines = [l + '.jpg' for l in lines]
    with open('data/voc/train.txt', 'w') as f:
        f.write('\n'.join(lines))
    with open('data/voc/val.txt', 'r') as f:
        lines = f.read().split('\n')
    lines = [l + '.jpg' for l in lines]
    with open('data/voc/val.txt', 'w') as f:
        f.write('\n'.join(lines))
    find_color_map.run('data/voc')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    args = parser.parse_args()
    voc2dataset(args.data_dir)
