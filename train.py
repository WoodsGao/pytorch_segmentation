import os
import random
import argparse
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from models import DeepLabV3Plus, UNet
from utils.modules.datasets import SegmentationDataset
from utils.modules.optims import AdaBoundW
from utils.utils import compute_loss, device, show_batch
from test import test
from time import time
# from torchsummary import summary

mixed_precision = True
try:  # Mixed precision training https://github.com/NVIDIA/apex
    from apex import amp
except:
    mixed_precision = False  # not installed

print(device)
writer = SummaryWriter()
# if torch.cuda.is_available():
#     torch.backends.cudnn.benchmark = True


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
    train_data = SegmentationDataset(train_dir,
                                     img_size=img_size,
                                     augments=augments)
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_data = SegmentationDataset(
        val_dir,
        img_size=img_size,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    best_miou = 0
    best_loss = 1000
    epoch = 0
    classes = train_loader.dataset.classes
    num_classes = len(classes)
    if unet:
        model = UNet(30)
    else:
        model = DeepLabV3Plus(30)
    model = model.to(device)
    # optimizer = AdaBoundW(model.parameters(), lr=lr, weight_decay=5e-4)
    if adam:
        optimizer = AdaBoundW(model.parameters(), lr=lr, weight_decay=5e-4)
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            #   weight_decay=5e-4,
            nesterov=True)
    if resume:
        state_dict = torch.load(weights, map_location=device)
        if adam:
            if 'adam' in state_dict:
                optimizer.load_state_dict(state_dict['adam'])
        best_miou = state_dict['miou']
        best_loss = state_dict['loss']
        epoch = state_dict['epoch']
        model.load_state_dict(state_dict['model'], strict=False)

    # lf = lambda x: 1 - x / epochs  # linear ramp to zero
    # lf = lambda x: 10 ** (hyp['lrf'] * x / epochs)  # exp ramp
    # lf = lambda x: 1 - 10 ** (hyp['lrf'] * (1 - x / epochs))  # inverse exp ramp
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=range(59, 70, 1), gamma=0.8)  # gradual fall to 0.1*lr0
    # scheduler = lr_scheduler.MultiStepLR(
    #     optimizer,
    #     milestones=[round(epochs * x) for x in [0.8, 0.9]],
    #     gamma=0.1,
    # )
    # scheduler.last_epoch = epoch - 1

    # summary(model, (3, img_size, img_size))

    # Mixed precision training https://github.com/NVIDIA/apex
    if mixed_precision:
        model, optimizer = amp.initialize(model,
                                          optimizer,
                                          opt_level='O1',
                                          verbosity=0)
    # create dataset
    while epoch < epochs:
        print('%d/%d' % (epoch, epochs))
        # train
        model.train()
        total_loss = 0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        optimizer.zero_grad()
        t0 = time()
        for idx, (inputs, targets) in pbar:
            t1 = time()
            batch_idx = idx + 1
            if idx == 0:
                show_batch('train_batch.png', inputs, targets, classes)
            inputs = inputs.to(device)
            targets = targets.to(device)
            if multi_scale:
                img_size = random.randrange(img_size_min, img_size_max) * 32
            if inputs.size(3) != img_size:
                inputs = F.interpolate(inputs,
                                       size=img_size,
                                       mode='bilinear',
                                       align_corners=False)
                targets = F.one_hot(targets, num_classes).permute(0, 3, 1,
                                                                  2).float()
                targets = F.interpolate(targets,
                                        size=img_size,
                                        mode='bilinear',
                                        align_corners=False)
                targets = targets.max(1)[1]
            outputs = model(inputs)
            loss = compute_loss(outputs, targets)
            total_loss += loss.item()
            loss *= batch_size / 64.
            # Compute gradient
            if mixed_precision:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            mem = torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available(
            ) else 0  # (GB)
            pbar.set_description(
                'train mem: %5.2lfGB loss: %8lf scale: %4d' %
                (mem, total_loss / batch_idx, inputs.size(2)))
            if batch_idx % accumulate == 0 or \
                    batch_idx == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
            t0 = time()
        torch.cuda.empty_cache()
        writer.add_scalar('train_loss', total_loss / len(train_loader), epoch)
        print('')
        # validate
        val_loss = best_loss
        miou = best_miou
        if not notest:
            val_loss, miou = test(model, val_loader)
            writer.add_scalar('valid_loss', val_loss, epoch)
            writer.add_scalar('miou', miou, epoch)
        epoch += 1
        # Save checkpoint.
        state_dict = {
            'model': model.state_dict(),
            'miou': miou,
            'loss': val_loss,
            'epoch': epoch
        }
        if adam:
            state_dict['adam'] = optimizer.state_dict()
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

        # scheduler.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data/voc')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--img-size', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--accumulate', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--unet', action='store_true')
    parser.add_argument('--adam', action='store_true')
    parser.add_argument('--notest', action='store_true')
    parser.add_argument('--weights', type=str, default='weights/last.pt')
    parser.add_argument('--multi-scale', action='store_true')
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
