import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from pytorch_modules.utils import device, Fetcher
from utils.models import DeepLabV3Plus
from utils.datasets import CocoDataset
from utils.utils import compute_loss, show_batch, compute_metrics
from tqdm import tqdm
import argparse


@torch.no_grad()
def test(model, fetcher):
    model.eval()
    val_loss = 0
    classes = fetcher.loader.dataset.classes
    num_classes = len(classes)
    total_size = 0
    # true positive / intersection
    tp = torch.zeros(num_classes)
    fp = torch.zeros(num_classes)
    fn = torch.zeros(num_classes)
    pbar = tqdm(fetcher)
    for idx, (inputs, targets) in enumerate(pbar):
        batch_idx = idx + 1
        outputs = model(inputs)
        loss = compute_loss(outputs, targets, model)
        val_loss += loss.item()
        predicted = outputs.max(1)[1]
        if idx == 0:
            show_batch(inputs, predicted)
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
        T, P, R, miou, F1 = compute_metrics(tp, fn, fp)
        pbar.set_description(
            'loss: %8g, mAP: %8g, F1: %8g, miou: %8g' %
            (val_loss / batch_idx, P.mean(), F1.mean(), miou.mean()))
    if dist.is_initialized():
        tp = tp.to(device)
        fn = fn.to(device)
        fp = fp.to(device)
        dist.all_reduce(tp, op=dist.ReduceOp.SUM)
        dist.all_reduce(fn, op=dist.ReduceOp.SUM)
        dist.all_reduce(fp, op=dist.ReduceOp.SUM)
        T, P, R, miou, F1 = compute_metrics(tp.cpu(), fn.cpu(), fp.cpu())
    if len(classes) < 10:
        for c_i, c in enumerate(classes):
            print(
                'cls: %8s, targets: %8d, pre: %8g, rec: %8g, iou: %8g, F1: %8g'
                % (c, T[c_i], P[c_i], R[c_i], miou[c_i], F1[c_i]))
    else:
        print('top error 5')
        copy_miou = miou.clone()
        for i in range(5):
            c_i = copy_miou.min(0)[1]
            copy_miou[c_i] = 1
            print(
                'cls: %8s, targets: %8d, pre: %8g, rec: %8g, iou: %8g, F1: %8g'
                % (classes[c_i], T[c_i], P[c_i], R[c_i], miou[c_i], F1[c_i]))
    return miou.mean().item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco', type=str, default='data/coco.json')
    parser.add_argument('--img-size', type=str, default='512')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--weights', type=str, default='')
    parser.add_argument('--num-workers', type=int, default=4)
    opt = parser.parse_args()

    img_size = opt.img_size.split(',')
    assert len(img_size) in [1, 2]
    if len(img_size) == 1:
        img_size = [int(img_size[0])] * 2
    else:
        img_size = [int(x) for x in img_size]

    val_data = CocoDataset(opt.coco, img_size=img_size, augments=None)
    val_loader = DataLoader(
        val_data,
        batch_size=opt.batch_size,
        pin_memory=True,
        num_workers=opt.num_workers,
    )
    val_fetcher = Fetcher(val_loader, post_fetch_fn=val_data.post_fetch_fn)
    model = DeepLabV3Plus(len(val_data.classes))
    if opt.weights:
        state_dict = torch.load(opt.weights, map_location='cpu')
        model.load_state_dict(state_dict['model'])
    metrics = test(model, val_fetcher)
    print('metrics: %8g' % (metrics))
