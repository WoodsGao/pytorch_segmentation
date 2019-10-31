import torch
from models import DeepLabV3Plus
from torch.utils.data import DataLoader
from utils.datasets import SegmentationDataset, show_batch
from utils import augments
from utils.losses import compute_loss
from utils import device
from tqdm import tqdm
import argparse


def test(model, val_loader, obj_conf=0.5, test_iters=0):
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
        pbar = tqdm(enumerate(val_loader), total=len(val_loader) if test_iters <= 0 else test_iters)
        for idx, (inputs, targets) in pbar:
            batch_idx = idx + 1
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = compute_loss(outputs, targets)[0]
            val_loss += loss.mean().item()
            predicted = torch.cat([outputs[0], outputs[1].softmax(1)], 1)
            predicted[:, 0, :, :][predicted[:, 0, :, :] > obj_conf] = 1
            predicted[:, 0, :, :][predicted[:, 0, :, :] < 1] = 0
            predicted = predicted.max(1)[1]
            if idx == 0:
                show_batch('test_batch.png', inputs.cpu(), predicted.cpu(), classes)
            predicted = predicted.view(-1)
            targets = targets.view(-1)
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
            union = tp + fp + fn
            union[union <= 0] = 1
            pbar.set_description('loss: %10lf, miou: %10lf' %
                                 (val_loss / batch_idx,
                                  (tp / union).mean()))
            if test_iters > 0 and batch_idx == test_iters:
                break
    print('')
    for c_i, c in enumerate(classes):
        print('cls: %10s, targets: %10d, pre: %10lf, rec: %10lf, iou: %10lf' %
              (c[0], tp[c_i] + fn[c_i], tp[c_i] /
               (tp[c_i] + fp[c_i]), tp[c_i] /
               (tp[c_i] + fn[c_i]), tp[c_i] / union[c_i]))
    val_loss /= len(val_loader)
    return val_loss, (tp / union).mean().item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data/voc/val.txt')
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--weight-path', type=str, default='weights/last.pt')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--test-iters', type=int, default=0)

    opt = parser.parse_args()

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
    val_loss, acc = test(model, val_loader, test_iters=opt.test_iters)
    print('val_loss: %10g   acc: %10g' % (val_loss, acc))
