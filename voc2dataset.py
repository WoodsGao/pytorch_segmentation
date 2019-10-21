from find_segment_classes import find_segment_classes
import subprocess
import argparse
import os


def voc2dataset(data_dir):
    cmd = [
        'rm data/voc', 
        'mkdir data', 
        'mkdir data/voc', 
        'cp {} data/voc/', 
        'cp {} data/voc/', 
        'mkdir data/voc/images data/voc/labels', 
        'cp {}/* data/voc/images', 
        'cp {}/* data/voc/labels'
    ]
    subprocess.call('\n'.join(cmd).format(
        os.path.join(data_dir, 'VOC2012/ImageSets/Segmentation/train.txt'),
        os.path.join(data_dir, 'VOC2012/ImageSets/Segmentation/val.txt'),
        os.path.join(data_dir, 'VOC2012/JPEGImages'),
        os.path.join(data_dir, 'VOC2012/SegmentationClass')), shell=True, universal_newlines=True)
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
    find_segment_classes('data/voc')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p',
                        '--path',
                        default='/home/uisee/Datasets/VOCdevkit')
    args = parser.parse_args()
    voc2dataset(args.path)