import torch
from models import DeepLabV3Plus
from torch.utils.data import DataLoader
from utils.datasets import SegmentationDataset, show_batch
from utils import augments
from utils.losses import compute_loss
from utils import device
from tqdm import tqdm
import argparse


def test(model, val_loader, obj_conf=0.5):
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
            outputs = model(inputs)
            loss = compute_loss(outputs, targets)
            val_loss += loss.mean().item()
            predicted = torch.cat([outputs[0], outputs[1].softmax(1)], 1)
            predicted[:, 0, :, :][predicted[:, 0, :, :] > obj_conf] = 1
            predicted[:, 0, :, :][predicted[:, 0, :, :] < 1] = 0
            predicted = predicted.max(1)[1]
            if idx == 0:
                show_batch('test_batch.png', inputs.cpu(), predicted.cpu(),
                           classes)
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
            pbar.set_description('loss: %8lf, miou: %8lf' %
                                 (val_loss / batch_idx, (tp / union).mean()))
    print('')
    miou = tp / union
    T = tp + fn
    P = tp / (tp + fp)
    R = tp / (tp + fn)
    F1 = 2 * tp / (2 * tp + fp + fn)
    for c_i, c in enumerate(classes):
        print(
            'cls: %8s, targets: %8d, pre: %8lf, rec: %8lf, iou: %8lf, F1: %8lf'
            % (c[0], T[c_i], P[c_i], R[c_i], miou[c_i], F1[c_i]))
    val_loss /= len(val_loader)
    return val_loss, miou.mean().item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data/voc/val.txt')
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--weights', type=str, default='')
    parser.add_argument('--num-workers', type=int, default=0)

    opt = parser.parse_args()

    val_data = SegmentationDataset(opt.data_dir,
                                   img_size=opt.img_size,
                                   augments=[
                                       augments.BGR2RGB(),
                                       augments.Normalize(),
                                       augments.NHWC2NCHW(),
                                   ])
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
    if opt.weights:
        state_dict = torch.load(opt.weights, map_location=device)
        model.load_state_dict(state_dict['model'])
    val_loss, miou, F1 = test(model, val_loader)
    print('val_loss: %8g   miou: %8g' % (val_loss, miou))
