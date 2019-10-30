import torch
from models import DeepLabV3Plus
import os
from torch.utils.data import DataLoader
from utils.datasets import SegmentationDataset
from utils import augments
from utils.loss import FocalBCELoss
from utils import device
from tqdm import tqdm
import argparse


def test(model, val_loader, criterion, obj_conf=0.5):
    model.eval()
    val_loss = 0
    classes = val_loader.dataset.classes
    num_classes = len(classes)
    total_size = 0
    # true positive / intersection
    tp = torch.zeros(num_classes)
    fp = torch.zeros(num_classes)
    fn = torch.zeros(num_classes)
    with torch.no_grad():
        pbar = tqdm(enumerate(val_loader), total=len(val_loader))
        for idx, (inputs, targets) in pbar:
            batch_idx = idx + 1
            inputs = inputs.to(device)
            targets = targets.to(device)
            pred_obj, pred_cls = model(inputs)
            true_obj = targets[:, 0:1, :, :]
            loss = criterion(pred_obj, true_obj)
            true_cls = targets[:, 1:, :, :]
            loss += criterion(pred_cls, true_cls)
            val_loss += loss.mean().item()
            predicted = torch.cat([pred_obj, pred_cls], 1)
            predicted[:, 0, :, :][predicted[:, 0, :, :] > (1 - obj_conf)] = 1
            predicted[:, 0, :, :][predicted[:, 0, :, :] < 1] = 0
            predicted = predicted.max(1)[1].view(-1)
            targets = targets.max(1)[1].view(-1)
            eq = predicted.eq(targets)
            total_size += predicted.size(0)
            for c_i, c in enumerate(classes):
                indices = targets.eq(c_i)
                positive = indices.sum().item()
                tpi = eq[indices].sum().item()
                fni = positive - tpi
                fpi = predicted.eq(c_i).sum().item() - tpi
                tp[c_i] += tpi
                fn[c_i] += fni
                fp[c_i] += fpi

            pbar.set_description('loss: %10lf, miou: %10lf' %
                                 (val_loss / batch_idx,
                                  (tp / (tp + fp + fn)).mean()))
    print('')
    for c_i, c in enumerate(classes):
        print('cls: %10s, targets: %10d, pre: %10lf, rec: %10lf, iou: %10lf' %
              (c[0], tp[c_i] + fn[c_i], tp[c_i] /
               (tp[c_i] + fp[c_i]), tp[c_i] /
               (tp[c_i] + fn[c_i]), tp[c_i] / (tp + fp + fn)[c_i]))
    val_loss /= len(val_loader)
    return val_loss, (tp / (tp + fp + fn)).mean().item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data/voc/val.txt')
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--weight-path', type=str, default='weights/last.pt')
    parser.add_argument('--num-workers', type=int, default=0)

    opt = parser.parse_args()

    criterion = FocalBCELoss(alpha=0.25, gamma=2)
    val_data = SegmentationDataset(
        opt.data_dir,
        img_size=opt.img_size,
        augments=[
            augments.BGR2RGB(),
            augments.Normalize(),
            augments.NHWC2NCHW(),
        ]
    )
    val_loader = DataLoader(
        val_data, 
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
    )
    classes = val_loader.dataset.classes
    num_classes = len(classes)
    model = DeepLabV3Plus(num_classes)
    model = model.to(device)
    # state_dict = torch.load(opt.weight_path, map_location=device)
    # model.load_state_dict(state_dict['model'])
    val_loss, acc = test(model, val_loader, criterion)
    print('val_loss: %10g   acc: %10g' % (val_loss, acc))
