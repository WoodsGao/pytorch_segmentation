import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils.datasets import SegmentationDataset
from model import DeepLabV3Plus
from model import UNet
import os
from utils import device
from utils import augments
from utils.loss import FocalBCELoss
from utils.optim import AdaBoundW
from tqdm import tqdm
from test import test
from torchsummary import summary
import random
import argparse

print(device)


def train(data_dir,
          epochs=100,
          img_size=224,
          batch_size=8,
          accumulate=2,
          lr=1e-3,
          resume=False,
          weights='',
          num_workers=-1,
          augments_list=[],
          multi_scale=False):
    if not os.path.exists('weights'):
        os.mkdir('weights')
    if multi_scale:
        img_size_min = max(img_size * 0.67 // 32, 1)
        img_size_max = max(img_size * 1.5 // 32, 1)
    train_dir = os.path.join(data_dir, 'train.txt')
    val_dir = os.path.join(data_dir, 'val.txt')
    train_data = SegmentationDataset(
        train_dir,
        img_size,
        augments=augments_list + [
            augments.BGR2RGB(),
            augments.Normalize(),
            augments.NHWC2NCHW(),
        ]
    )
    train_loader = DataLoader(
        train_data, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=min([os.cpu_count(), batch_size, 16]) if num_workers < 0 else num_workers,
    )
    val_data = SegmentationDataset(
        val_dir,
        img_size,
        augments=[
            augments.BGR2RGB(),
            augments.Normalize(),
            augments.NHWC2NCHW(),
        ]
    )
    val_loader = DataLoader(
        val_data, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=min([os.cpu_count(), batch_size, 16]) if num_workers < 0 else num_workers,
    )
    best_miou = 0
    best_loss = 1000
    epoch = 0
    num_classes = len(train_loader.dataset.classes)
    # model = DeepLabV3Plus(num_classes)
    model = UNet(num_classes)
    model = model.to(device)
    criterion = FocalBCELoss(alpha=0.25, gamma=2)
    optimizer = AdaBoundW(model.parameters(), lr=lr, weight_decay=5e-4)
    # summary(model, (3, img_size, img_size))
    if resume:
        state_dict = torch.load(weights, map_location=device)
        best_miou = state_dict['miou']
        best_loss = state_dict['loss']
        epoch = state_dict['epoch']
        model.load_state_dict(state_dict['model'], strict=False)
        optimizer.load_state_dict(state_dict['optimizer'])

    # create dataset
    against_examples = []
    while epoch < epochs:
        print('%d/%d' % (epoch, epochs))
        # train
        model.train()
        total_loss = 0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        optimizer.zero_grad()
        for idx, (
                inputs,
                targets,
        ) in pbar:
            batch_idx = idx + 1
            inputs = inputs.to(device)
            targets = targets.to(device)
            if multi_scale:
                inputs = F.interpolate(inputs, size=img_size, mode='bilinear', align_corners=False)
                targets = F.interpolate(targets, size=img_size, mode='bilinear', align_corners=False)
            outputs = model(inputs)
            targets_obj = 1 - targets[:, 0:1]
            outputs_obj = outputs[:, 0:1].sigmoid()
            loss = criterion(outputs_obj, targets_obj)
            targets_cls = targets[0, 1:] * targets_obj
            outputs_cls = outputs[0, 1:].softmax(1) * targets_obj
            loss += criterion(outputs_cls, targets_cls)
            loss = loss.view(loss.size(0), -1).mean(1)
            against_examples.append(
                [inputs[loss > loss.mean()], targets[loss > loss.mean()]])
            loss.mean().backward()
            total_loss += loss.mean().item()
            pbar.set_description('train loss: %10lf scale: %10d' %
                                 (total_loss / batch_idx, inputs.size(2)))
            if batch_idx % accumulate == 0 or \
                    batch_idx == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
                # against examples training
                for example in against_examples:
                    inputs = example[0]
                    if inputs.size(0) < 2:
                        continue
                    targets = example[1]
                    outputs = model(inputs)
                    targets_obj = 1 - targets[:, 0:1]
                    outputs_obj = outputs[:, 0:1].sigmoid()
                    loss = criterion(outputs_obj, targets_obj)
                    targets_cls = targets[0, 1:] * targets_obj
                    outputs_cls = outputs[0, 1:].softmax(1) * targets_obj
                    loss += criterion(outputs_cls, targets_cls)
                    loss.mean().backward()
                optimizer.step()
                optimizer.zero_grad()
                against_examples = []

                # multi scale
                if multi_scale:
                    img_size = random.randrange(img_size_min, img_size_max) * 32
        print('')
        # validate
        val_loss, miou = test(model, val_loader, criterion, )
        # Save checkpoint.
        state_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'miou': miou,
            'loss': val_loss,
            'epoch': epoch
        }
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
        epoch += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data/voc')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--img-size', type=int, default=320)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--accumulate', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=-1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--weights', type=str, default='weights/last.pt')
    parser.add_argument('--multi-scale', action='store_true')
    augments_list = [
        augments.PerspectiveProject(0.3, 0.2),
        augments.HSV_H(0.3, 0.5),
        augments.HSV_S(0.3, 0.5),
        augments.HSV_V(0.3, 0.5),
        augments.Rotate(1, 0.2),
        augments.Blur(0.03, 0.2),
        augments.Noise(0.3, 0.2),
        augments.H_Flap(0.5),
        augments.V_Flap(0.5)
    ]
    opt = parser.parse_args()
    train(data_dir=opt.data_dir,
          epochs=opt.epochs,
          img_size=opt.img_size,
          batch_size=opt.batch_size,
          accumulate=opt.accumulate,
          lr=opt.lr,
          resume=opt.resume,
          weights=opt.weights,
          num_workers=opt.num_workers,
          augments_list=augments_list,
          multi_scale=opt.multi_scale)
