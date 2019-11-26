import os
import random
import argparse
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, DistributedSampler
import torch.nn.functional as F
import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from models import DeepLabV3Plus, UNet
from utils.modules.datasets import SegmentationDataset
from utils.modules.optims import AdaBoundW
from utils.utils import compute_loss, device, show_batch, load_checkpoint
from test import test
from time import time

print(device)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
amp = None
try:
    from apex import amp
except ImportError:
    pass


def train(data_dir,
          epochs=100,
          img_size=224,
          batch_size=8,
          accumulate=2,
          lr=1e-3,
          resume=False,
          unet=False,
          adam=False,
          weights='',
          num_workers=0,
          augments={},
          multi_scale=False,
          notest=False):
    os.makedirs('weights', exist_ok=True)
    if multi_scale:
        img_size_min = max(img_size * 0.67 // 32, 1)
        img_size_max = max(img_size * 1.5 // 32, 1)
    train_dir = os.path.join(data_dir, 'train.txt')
    val_dir = os.path.join(data_dir, 'valid.txt')
    skip = DIST and RANK > 0
    train_data = SegmentationDataset(
        train_dir,
        img_size=img_size,
        augments={} if skip else augments,
        skip_init=skip,
    )
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=False if DIST else True,
        sampler=DistributedSampler(train_data, WORLD_SIZE, RANK)
        if DIST else None,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=train_data.collate_fn,
    )
    if not notest:
        val_data = SegmentationDataset(
            val_dir,
            img_size=img_size,
            augments={},
            skip_init=skip,
        )
        val_loader = DataLoader(
            val_data,
            batch_size=batch_size,
            shuffle=False if DIST else True,
            sampler=DistributedSampler(val_data, WORLD_SIZE, RANK)
            if DIST else None,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=val_data.collate_fn,
        )
    accumulate_count = 0
    best_miou = 0
    best_loss = 1000
    epoch = 0
    if unet:
        model = UNet(32)
    else:
        model = DeepLabV3Plus(32)
    model = model.to(device)

    if adam:
        optimizer = optim.AdamW(model.parameters(),
                                lr=lr if lr > 0 else 1e-4,
                                weight_decay=1e-5)
    else:
        optimizer = optim.SGD(model.parameters(),
                              lr=lr if lr > 0 else 1e-3,
                              momentum=0.9,
                              weight_decay=1e-5,
                              nesterov=True)
    if MP:
        print('mixed precision')
        model, optimizer = amp.initialize(model,
                                          optimizer,
                                          opt_level='O1',
                                          verbosity=0)
    if DIST:
        model = torch.nn.parallel.DistributedDataParallel(
            model, find_unused_parameters=True)
    if resume:
        best_loss, best_miou, epoch = load_checkpoint(weights, model,
                                                      optimizer, adam)
        if lr > 0:
            for pg in optimizer.param_groups:
                pg['lr'] = lr
    optimizer.zero_grad()
    if DIST:
        model.require_backward_grad_sync = False
    c = 0
    t = [0, 0, 0, 0]
    while epoch < epochs:
        print('%d/%d' % (epoch, epochs))
        # train
        model.train()
        total_loss = 0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        t0 = time()
        for idx, (inputs, targets) in pbar:
            t1 = time()
            if inputs.size(0) < 2:
                continue
            c += 1
            accumulate_count += 1
            batch_idx = idx + 1
            if idx == 0:
                show_batch('train_batch.png', inputs, targets)
            inputs = inputs.to(device)
            targets = targets.to(device)
            if multi_scale:
                img_size = random.randrange(img_size_min, img_size_max) * 32
            if inputs.size(3) != img_size:
                inputs = F.interpolate(inputs,
                                       size=img_size,
                                       mode='bilinear',
                                       align_corners=False)
                targets = F.interpolate(targets,
                                        size=img_size,
                                        mode='bilinear',
                                        align_corners=False)
            if accumulate_count % accumulate == 0:
                model.require_backward_grad_sync = True
            outputs = model(inputs)
            t2 = time()
            loss = compute_loss(outputs, targets)
            total_loss += loss.item()
            loss /= accumulate
            t3 = time()
            # Compute gradient
            if MP:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            t4 = time()
            t[0] += t1 - t0
            t[1] += t2 - t1
            t[2] += t3 - t2
            t[3] += t4 - t3
            mem = torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available(
            ) else 0  # (GB)
            pbar.set_description(
                'train mem: %5.2lfGB loss: %8lf scale: %4d %g %g %g %g' %
                (mem, total_loss / batch_idx, inputs.size(2), t[0] / c,
                 t[1] / c, t[2] / c, t[3] / c))
            if accumulate_count % accumulate == 0:
                accumulate_count = 0
                optimizer.step()
                optimizer.zero_grad()
                model.require_backward_grad_sync = False
            t0 = time()
        torch.cuda.empty_cache()
        if writer is not None:
            writer.add_scalar('train_loss', total_loss / len(train_loader),
                              epoch)
        # validate
        val_loss = best_loss
        miou = best_miou
        if not notest:
            val_loss, miou = test(model, val_loader, DIST=DIST)
            if writer is not None:
                writer.add_scalar('valid_loss', val_loss, epoch)
                writer.add_scalar('miou', miou, epoch)
        print('miou: %8g    loss: %8g' % (miou, val_loss))
        epoch += 1
        for pg in optimizer.param_groups:
            pg['lr'] *= (1 - 1e-8)
        if RANK == 0:
            # Save checkpoint.
            state_dict = {
                'model': model.state_dict(),
                'miou': miou,
                'loss': val_loss,
                'epoch': epoch
            }
            if adam:
                state_dict['adam'] = optimizer.state_dict()
            else:
                state_dict['sgd'] = optimizer.state_dict()
            torch.save(state_dict, 'weights/last.pt')
            if val_loss < best_loss:
                print('\nSaving best_loss.pt..')
                torch.save(state_dict, 'weights/best_loss.pt')
                best_loss = val_loss
            if miou > best_miou:
                print('\nSaving best_miou.pt..')
                torch.save(state_dict, 'weights/best_miou.pt')
                best_miou = miou
            if epoch % 10 == 0 and epoch > 1:
                print('\nSaving backup%d.pt..' % epoch)
                torch.save(state_dict, 'weights/backup%d.pt' % epoch)


if __name__ == "__main__":
    global DIST, RANK, WORLD_SIZE, writer, MP
    writer = None
    RANK = 0
    WORLD_SIZE = 1
    DIST = False
    MP = False
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data/voc')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--img-size', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--accumulate', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--unet', action='store_true')
    parser.add_argument('--adam', action='store_true')
    parser.add_argument('--mp', action='store_true', help='mixed precision')
    parser.add_argument('--notest', action='store_true')
    parser.add_argument('--weights', type=str, default='weights/last.pt')
    parser.add_argument('--multi-scale', action='store_true')
    parser.add_argument('--backend',
                        type=str,
                        default='gloo',
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
    if opt.world_size > 1:
        dist.init_process_group(
            backend=opt.backend,
            init_method=opt.init_method,
            world_size=opt.world_size,
            rank=opt.rank,
        )
        WORLD_SIZE = opt.world_size
        RANK = opt.rank
        if dist.is_available() and dist.is_initialized():
            DIST = True
    if opt.mp:
        if amp is not None:
            MP = True
    if RANK > 0:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter()
        except ImportError:
            pass
    train(
        data_dir=opt.data_dir,
        epochs=opt.epochs,
        img_size=opt.img_size,
        batch_size=opt.batch_size,
        accumulate=opt.accumulate,
        lr=opt.lr,
        resume=opt.resume,
        weights=opt.weights,
        num_workers=opt.num_workers,
        augments=augments,
        multi_scale=opt.multi_scale,
        notest=opt.notest,
        adam=opt.adam,
        unet=opt.unet,
    )
