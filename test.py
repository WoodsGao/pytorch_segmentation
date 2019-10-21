import torch
from model import DeepLabV3Plus
import os
from utils.dataloader import Dataloader
from utils import augments
from utils.loss import FocalBCELoss
from utils import device
from tqdm import tqdm
import argparse


def test(model, val_loader, criterion):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    num_classes = len(val_loader.classes)
    total_c = torch.zeros(num_classes)
    tp = torch.zeros(num_classes)
    fp = torch.zeros(num_classes)
    tn = torch.zeros(num_classes)
    fn = torch.zeros(num_classes)
    with torch.no_grad():
        pbar = tqdm(range(1, val_loader.iter_times + 1))
        for batch_idx in pbar:
            inputs, targets = val_loader.next()
            inputs = torch.FloatTensor(inputs).to(device)
            targets = torch.FloatTensor(targets).to(device)
            outputs = model(inputs)[0].sigmoid()
            loss = criterion(outputs, targets)
            loss = loss.view(loss.size(0), -1).mean(1)
            val_loss += loss.mean().item()
            predicted = outputs.max(1)[1].view(-1)
            targets = targets.max(1)[1].view(-1)
            eq = predicted.eq(targets)
            total += targets.size(0)
            correct += eq.sum().item()
            acc = 100. * correct / total

            for c_i, c in enumerate(val_loader.classes):
                indices = targets.eq(c_i).nonzero()
                total_c[c_i] += targets.eq(c_i).sum().item()
                tp[c_i] += eq[indices].sum().item()
                fn[c_i] += targets.eq(c_i).sum().item() - \
                    eq[indices].sum().item()
                indices = predicted.eq(c_i).nonzero()
                tn[c_i] += eq[indices].sum().item()
                fp[c_i] += predicted.eq(c_i).sum().item() - \
                    eq[indices].sum().item()

            pbar.set_description('loss: %10lf, acc: %10lf' %
                                 (val_loss / batch_idx, acc))

    for c_i, c in enumerate(val_loader.classes):
        print('cls: %10s, targets: %10d, pre: %10lf, rec: %10lf' %
              (c, total_c[c_i], tp[c_i] / (tp[c_i] + fp[c_i]), tp[c_i] /
               (tp[c_i] + fn[c_i])))
    val_loss /= val_loader.iter_times
    return val_loss, acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--weight_path', type=str, required=True)
    opt = parser.parse_args()

    criterion = FocalBCELoss(alpha=0.25, gamma=2)
    val_loader = Dataloader(
        opt.data_dir,
        img_size=opt.img_size,
        batch_size=opt.batch_size,
        augments=[
            augments.BGR2RGB(),
            augments.Normalize(),
            augments.NHWC2NCHW(),
        ],
    )
    num_classes = len(val_loader.classes)
    model = DeepLabV3Plus(3, num_classes)
    model = model.to(device)
    state_dict = torch.load(opt.weight_path, map_location=device)
    model.load_state_dict(state_dict['model'])
    val_loss, acc = test(model, val_loader, criterion)
    print('val_loss: %10g   acc: %10g' % (val_loss, acc))
