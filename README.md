# pytorch_segmentation

## Introduction

Implementation of some semantic segmentation models with pytorch, including DeepLabV3+, UNet, etc.

## Features

 - Advanced neural network models
 - Flexible and efficient toolkit(See [woodsgao/pytorch_modules](https://github.com/woodsgao/pytorch_modules))
 - Online data augmenting(See [woodsgao/image_augments](https://github.com/woodsgao/image_augments))
 - Mixed precision training(If you have already installed [apex](https://github.com/NVIDIA/apex))

## Installation

    git clone https://github.com/woodsgao/pytorch_segmentation
    cd pytorch_segmentation
    pip install -r requirements.txt

## Usage

### Create custom data

Please organize your data in the following format:

    data/
        <custom>/
            images/
                0001.png
                0002.png
                ...
            labels/
                0001.png
                0002.png
                ...
            classes.names

The content of `classes.names` is:

    <class_name>, <blue value of the color>, <green value of the color>, <red value of the color>
    ...
            
Then execute `python3 split_dataset.py data/<custom>`.It splits the data into training and validation sets and generates `data/<custom>/train.txt` and `data/<custom>/valid.txt`.

### Training

    python3 train.py --data-dir data/<custom> --img-size 512 --batch-size 8 --accumulate 8 --epoch 200 --lr 1e-4 --adam
