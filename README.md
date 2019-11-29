# pytorch_segmentation

## Introduction

Implementation of some semantic segmentation models with pytorch, including DeepLabV3+, UNet, etc.

## Features

 - Advanced neural network models
 - Flexible and efficient toolkit(See [woodsgao/pytorch_modules](https://github.com/woodsgao/pytorch_modules))
 - Online data augmenting(See [woodsgao/image_augments](https://github.com/woodsgao/image_augments))
 - Mixed precision training(If you have already installed [apex](https://github.com/NVIDIA/apex))
 - Efficient distributed training(0.8x faster when using two 2080ti)

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

    <class_name_1>
    ...
            
Then execute `python3 split_dataset.py data/<custom>`.It splits the data into training and validation sets and generates `data/<custom>/train.txt` and `data/<custom>/valid.txt`.

### Training

    python3 train.py --data-dir data/<custom>

### Distributed Training

Run the following command in all nodes.Every node will save your weights
    python3 -m torch.distributed.launch --nnodes <nnodes> --node_rank <node_rank> --nproc_per_node <nproc_per_node>  --master_addr <master_addr>  --master_port <master_port> train.py --data-dir data/<custom>

### Testing

    python3 test.py --val-list /data/<custom>/valid.txt

### Inference

    python3 inference.py --img-dir data/samples
