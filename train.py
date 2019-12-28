import os
import os.path as osp
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from utils.models import DeepLabV3Plus
from pytorch_modules.utils import Trainer, Fetcher
from utils.utils import compute_loss
from utils.datasets import SegDataset, TRAIN_AUGS
from test import test


def train(data_dir,
          epochs=100,
          img_size=224,
          batch_size=8,
          accumulate=2,
          lr=1e-3,
          adam=False,
          weights='',
          num_workers=0,
          multi_scale=False,
          notest=False,
          mixed_precision=False,
          nosave=False):
    os.makedirs('weights', exist_ok=True)
    train_dir = osp.join(data_dir, 'train.txt')
    val_dir = osp.join(data_dir, 'valid.txt')
    train_data = SegDataset(
        train_dir,
        img_size=img_size,
        augments=TRAIN_AUGS,
    )
    if not notest:
        val_data = SegDataset(val_dir, img_size=img_size)
    if dist.is_initialized():
        dist.barrier()
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=not dist.is_initialized(),
        sampler=DistributedSampler(train_data, dist.get_world_size(),
                                   dist.get_rank())
        if dist.is_initialized() else None,
        pin_memory=True,
        num_workers=num_workers,
    )
    train_fetcher = Fetcher(train_loader, train_data.post_fetch_fn)
    if not notest:
        val_loader = DataLoader(
            val_data,
            batch_size=batch_size,
            shuffle=not dist.is_initialized(),
            sampler=DistributedSampler(val_data, dist.get_world_size(),
                                       dist.get_rank())
            if dist.is_initialized() else None,
            pin_memory=True,
            num_workers=num_workers,
        )
        val_fetcher = Fetcher(val_loader, post_fetch_fn=val_data.post_fetch_fn)

    if osp.exists('weights/voc480.pt') and not weights:
        w = torch.load('weights/voc480.pt')
        model = DeepLabV3Plus(21)
        model.load_state_dict(w['model'])
        model.cls_conv = nn.Conv2d(304, len(train_data.classes), 3, padding=1)
    else:
        model = DeepLabV3Plus(len(train_data.classes))

    trainer = Trainer(model,
                      train_fetcher,
                      compute_loss,
                      weights,
                      accumulate,
                      adam=adam,
                      lr=lr,
                      mixed_precision=mixed_precision)
    while trainer.epoch < epochs:
        trainer.run_epoch()
        save_path_list = ['last.pt']
        if trainer.epoch % 10 == 0:
            save_path_list.append('bak%d.pt' % trainer.epoch)
        if not notest:
            metrics = test(trainer.model, val_fetcher)
            if metrics > trainer.metrics:
                trainer.metrics = metrics
                save_path_list.append('best.pt')
                print('save best, metrics: %g...' % metrics)
        save_path_list = [osp.join('weights', p) for p in save_path_list]
        if nosave:
            continue
        trainer.save(save_path_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/voc')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--img-size', type=str, default='512')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--accumulate', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--adam', action='store_true')
    parser.add_argument('--mp', action='store_true', help='mixed precision')
    parser.add_argument('--notest', action='store_true')
    parser.add_argument('--weights', type=str, default='')
    parser.add_argument('--multi-scale', action='store_true')
    parser.add_argument('--nosave', action='store_true')
    parser.add_argument('--backend',
                        type=str,
                        default='nccl',
                        help='Name of the backend to use.')
    parser.add_argument('-i',
                        '--init-method',
                        type=str,
                        default='tcp://127.0.0.1:23456',
                        help='URL specifying how to initialize the package.')
    parser.add_argument('-s',
                        '--world-size',
                        type=int,
                        help='Number of processes participating in the job.',
                        default=1)
    parser.add_argument('-r',
                        '--rank',
                        type=int,
                        help='Rank of the current process.',
                        default=0)
    parser.add_argument('--local-rank', '--local_rank', type=int, default=0)
    opt = parser.parse_args()
    print(opt)

    img_size = opt.img_size.split(',')
    assert len(img_size) in [1, 2]
    if len(img_size) == 1:
        img_size = [int(img_size[0])] * 2
    else:
        img_size = [int(x) for x in img_size]

    if dist.is_available() and opt.world_size > 1:
        dist.init_process_group(backend=opt.backend,
                                init_method=opt.init_method,
                                world_size=opt.world_size,
                                rank=opt.rank)
    train(
        data_dir=opt.data,
        epochs=opt.epochs,
        img_size=tuple(img_size),
        batch_size=opt.batch_size,
        accumulate=opt.accumulate,
        lr=opt.lr,
        weights=opt.weights,
        num_workers=opt.num_workers,
        multi_scale=opt.multi_scale,
        notest=opt.notest,
        adam=opt.adam,
        mixed_precision=opt.mp,
        nosave=opt.nosave or (opt.local_rank > 0),
    )
    if dist.is_initialized():
        dist.destroy_process_group()
