# pytorch_segmentation

## Introduction

Implementation of some semantic segmentation models with pytorch, including DeepLabV3+, UNet, etc.

## Features

 - Advanced neural network models
 - Flexible and efficient toolkit(See [woodsgao/pytorch_modules](https://github.com/woodsgao/pytorch_modules))
 - Online data augmenting(by imgaug)
 - Mixed precision training(If you have already installed [apex](https://github.com/NVIDIA/apex))
 - Efficient distributed training(0.8x faster when using two 2080ti)
 - Add a script to convert to caffe model(By [woodsgao/pytorch2caffe](https://github.com/woodsgao/pytorch2caffe))

## Installation

    git clone https://github.com/woodsgao/pytorch_segmentation
    cd pytorch_segmentation
    pip install -r requirements.txt

## Tutorials

### Create custom data

Please organize your data in coco format(by default):

    data/
        <custom>/
            images/
            coco.json
            train.json
            val.json

You can use `split_coco_json.py` from [woodsgao/cv_utils](https://github.com/woodsgao/cv_utils)
 to split your `coco.json` file into `train.json` and `val.json`

### Training

    python3 train.py data/<custom>

### Distributed Training

    python3 -m torch.distributed.launch --nproc_per_node=<nproc> train.py data/<custom>

### Testing

    python3 test.py data/<custom>/val.json --weights weights.pth

### Inference

    python3 inference.py data/samples outputs --weights weights.pth

### Export to caffe model

    python3 export2caffe.py weights/best.pt -nc 2 -s 320 320