import os
import argparse
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from models import DeepLabV3Plus
from utils.modules.datasets import SegmentationDataset
from utils.modules.utils import Trainer, Fetcher
from utils.utils import compute_loss
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
          augments={},
          multi_scale=False,
          notest=False,
          mixed_precision=False,
          distributed=False,
          rank=0,
          world_size=1):
    os.makedirs('weights', exist_ok=True)
    train_dir = os.path.join(data_dir, 'train.txt')
    val_dir = os.path.join(data_dir, 'valid.txt')
    skip = distributed and rank > 0
    train_data = SegmentationDataset(
        train_dir,
        img_size=img_size,
        augments=augments,
        skip_init=skip,
    )
    if distributed:
        dist.barrier()
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=not distributed,
        sampler=DistributedSampler(train_data, world_size, rank)
        if distributed else None,
        pin_memory=True,
        num_workers=num_workers,
    )
    train_fetcher = Fetcher(train_loader, train_data.post_fetch_fn)
    if not notest:
        val_data = SegmentationDataset(
            val_dir,
            img_size=img_size,
            augments={},
            skip_init=skip,
        )
        if distributed:
            dist.barrier()
        val_loader = DataLoader(
            val_data,
            batch_size=batch_size,
            shuffle=not distributed,
            sampler=DistributedSampler(val_data, world_size, rank)
            if distributed else None,
            pin_memory=True,
            num_workers=num_workers,
        )
        val_fetcher = Fetcher(val_loader, post_fetch_fn=val_data.post_fetch_fn)

    model = DeepLabV3Plus(32)

    trainer = Trainer(model, train_fetcher, compute_loss, weights,
                      accumulate, adam, lr, distributed, mixed_precision)
    while trainer.epoch < epochs:
        trainer.run_epoch()
        save_path_list = ['last.pt']
        if trainer.epoch % 10 == 0:
            save_path_list.append('bak%d.pt' % trainer.epoch)
        if not notest:
            metrics = test(trainer.model, val_fetcher, distributed=distributed)
            if metrics > trainer.metrics:
                trainer.metrics = metrics
                save_path_list.append('best.pt')
                print('save best, metrics: %g...' % metrics)
        save_path_list = [os.path.join('weights', p) for p in save_path_list]
        if rank == 0:
            trainer.save(save_path_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data/voc')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--img-size', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--accumulate', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0)
    parser.add_argument('--adam', action='store_true')
    parser.add_argument('--mp', action='store_true', help='mixed precision')
    parser.add_argument('--notest', action='store_true')
    parser.add_argument('--weights', type=str, default='')
    parser.add_argument('--multi-scale', action='store_true')
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
                        default=1,
                        help='Number of processes participating in the job.')
    parser.add_argument('-r',
                        '--rank',
                        type=int,
                        default=0,
                        help='Rank of the current process.')
    augments = {
        'hsv': 0.1,
        'blur': 0.1,
        'pepper': 0.1,
        'shear': 0.1,
        'translate': 0.1,
        'rotate': 0.1,
        'flip': 0.1,
        'scale': 0.1,
        'noise': 0.1,
    }
    opt = parser.parse_args()
    print(opt)
    distributed = False
    if opt.world_size > 1:
        dist.init_process_group(
            backend=opt.backend,
            init_method=opt.init_method,
            world_size=opt.world_size,
            rank=opt.rank,
        )
        if dist.is_available() and dist.is_initialized():
            distributed = True
    train(
        data_dir=opt.data_dir,
        epochs=opt.epochs,
        img_size=opt.img_size,
        batch_size=opt.batch_size,
        accumulate=opt.accumulate,
        lr=opt.lr,
        weights=opt.weights,
        num_workers=opt.num_workers,
        augments=augments,
        multi_scale=opt.multi_scale,
        notest=opt.notest,
        adam=opt.adam,
        mixed_precision=opt.mp,
        distributed=distributed,
        rank=opt.rank,
        world_size=opt.world_size,
    )
