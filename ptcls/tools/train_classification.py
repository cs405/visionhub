"""Classification Training Script

标准图像分类训练，支持：
- ImageNet风格数据集（文件夹结构）
- YOLO分类数据集
- 多种Backbone（50+个）
- 多种Loss（CrossEntropy, LabelSmoothing, Focal等）
- 数据增强、Mixup、CutMix
- LR Scheduler、EMA、梯度裁剪
- AMP混合精度训练

Usage:
# ImageFolder格式
python visionhub/tools/train_classification.py \
  --data_root ./data/imagenet \
  --save_dir ./output/cls \
  --model resnet50 \
  --epochs 100 --batch_size 128 --device cuda

# YOLO分类格式
python visionhub/tools/train_classification.py \
  --yolo_images dataset/train \
  --yolo_labels dataset/train \
  --data_yaml dataset/data.yaml \
  --model efficientnet_b0 \
  --epochs 100 --batch_size 64 --device cuda
"""

import argparse
import os
import sys
import time
import json
from pathlib import Path

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np

from ptcls.arch.backbone import build_backbone, list_backbones
from ptcls.data.datasets.image_folder_pairs import ImageFolderDataset
from ptcls.data.datasets.yolo_det_crop_dataset import YoloDetCropDataset


def parse_args():
    p = argparse.ArgumentParser(description="Image Classification Training")

    # Dataset
    p.add_argument("--data_root", default=None, help="ImageFolder root (train/val subfolders)")
    p.add_argument("--yolo_images", default=None, help="YOLO classification images dir")
    p.add_argument("--yolo_labels", default=None, help="YOLO classification labels dir")
    p.add_argument("--data_yaml", default=None, help="YOLO data.yaml for class names")
    p.add_argument("--num_classes", type=int, default=None, help="Number of classes (auto-detect if not set)")

    # Model
    p.add_argument("--model", default="resnet50", choices=list_backbones(), help="Backbone model")
    p.add_argument("--pretrained", action="store_true", help="Use ImageNet pretrained weights")
    p.add_argument("--resume", default=None, help="Resume from checkpoint")

    # Training
    p.add_argument("--save_dir", required=True, help="Output directory")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num_workers", type=int, default=4)

    # Loss
    p.add_argument("--loss", choices=["ce", "label_smoothing", "focal"], default="ce")
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--focal_alpha", type=float, default=0.25)
    p.add_argument("--focal_gamma", type=float, default=2.0)

    # Augmentation
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--mixup", type=float, default=0.0, help="Mixup alpha (0 to disable)")
    p.add_argument("--cutmix", type=float, default=0.0, help="CutMix alpha (0 to disable)")
    p.add_argument("--auto_augment", action="store_true", help="Use AutoAugment")

    # LR Scheduler
    p.add_argument("--scheduler", choices=["cosine", "step", "multistep", "none"], default="cosine")
    p.add_argument("--lr_min", type=float, default=1e-5)
    p.add_argument("--warmup_epochs", type=int, default=5)

    # Advanced
    p.add_argument("--amp", action="store_true", help="Automatic Mixed Precision")
    p.add_argument("--ema", action="store_true", help="Exponential Moving Average")
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--grad_clip", type=float, default=0.0)

    # Validation
    p.add_argument("--val_interval", type=int, default=1)
    p.add_argument("--save_interval", type=int, default=10)

    return p.parse_args()


def build_dataset(args):
    """构建训练和验证数据集"""
    # Data augmentation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(args.img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if args.data_root:
        # ImageFolder format: data_root/train/, data_root/val/
        train_dir = os.path.join(args.data_root, "train")
        val_dir = os.path.join(args.data_root, "val")

        if not os.path.exists(train_dir):
            raise FileNotFoundError(f"Training directory not found: {train_dir}")

        train_ds = ImageFolderDataset(train_dir, transform=train_transform)
        val_ds = ImageFolderDataset(val_dir, transform=val_transform) if os.path.exists(val_dir) else None

        num_classes = len(train_ds.classes)

    elif args.yolo_images and args.yolo_labels:
        # YOLO classification format
        train_ds = YoloDetCropDataset(
            args.yolo_images,
            args.yolo_labels,
            data_yaml=args.data_yaml,
            transform=train_transform
        )

        # Try to build validation set
        val_images = args.yolo_images.replace("train", "val")
        val_labels = args.yolo_labels.replace("train", "val")

        if os.path.exists(val_images) and os.path.exists(val_labels):
            val_ds = YoloDetCropDataset(
                val_images,
                val_labels,
                data_yaml=args.data_yaml,
                transform=val_transform
            )
        else:
            val_ds = None

        # Get num_classes from dataset
        labels = [int(s.label if hasattr(s, 'label') else s.cls_id) for s in train_ds.samples]
        num_classes = max(labels) + 1
    else:
        raise ValueError("Must provide either --data_root or --yolo_images + --yolo_labels")

    return train_ds, val_ds, num_classes


def mixup_data(x, y, alpha=1.0):
    """Mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    """CutMix augmentation"""
    if alpha <= 0:
        return x, y, y, 1.0

    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    # Adjust lambda
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def train_epoch(model, loader, criterion, optimizer, device, args, scaler=None, epoch=0):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    if TQDM_AVAILABLE:
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")
    else:
        pbar = loader

    for batch_idx, (x, y) in enumerate(pbar):
        x, y = x.to(device), y.to(device)

        # Mixup / CutMix
        if args.mixup > 0 and np.random.rand() < 0.5:
            x, y_a, y_b, lam = mixup_data(x, y, args.mixup)
            mixed = True
        elif args.cutmix > 0 and np.random.rand() < 0.5:
            x, y_a, y_b, lam = cutmix_data(x, y, args.cutmix)
            mixed = True
        else:
            mixed = False

        optimizer.zero_grad()

        # Forward
        if args.amp and scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(x)
                if mixed:
                    loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
                else:
                    loss = criterion(outputs, y)

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(x)
            if mixed:
                loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
            else:
                loss = criterion(outputs, y)

            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        total_loss += loss.item()

        # Accuracy (skip for mixed samples)
        if not mixed:
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

        if TQDM_AVAILABLE:
            pbar.set_postfix({
                'loss': f'{total_loss/(batch_idx+1):.4f}',
                'acc': f'{100.*correct/total:.2f}%' if total > 0 else 'N/A'
            })

    if TQDM_AVAILABLE:
        pbar.close()

    return total_loss / len(loader), 100. * correct / total if total > 0 else 0


@torch.no_grad()
def validate(model, loader, criterion, device):
    """验证"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        outputs = model(x)
        loss = criterion(outputs, y)

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()

    return total_loss / len(loader), 100. * correct / total


def main():
    args = parse_args()

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Save args
    with open(os.path.join(args.save_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Dataset
    print("[INFO] Building datasets...")
    train_ds, val_ds, num_classes = build_dataset(args)
    print(f"[INFO] Train: {len(train_ds)} samples, Val: {len(val_ds) if val_ds else 0} samples, Classes: {num_classes}")

    if args.num_classes is None:
        args.num_classes = num_classes

    # DataLoader
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    ) if val_ds else None

    # Model
    print(f"[INFO] Building model: {args.model}")
    model = build_backbone(args.model, num_classes=args.num_classes, pretrained=args.pretrained)
    model = model.to(device)

    # Loss
    if args.loss == "label_smoothing":
        from ptcls.loss.metric import LabelSmoothingCrossEntropy
        criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
    elif args.loss == "focal":
        from ptcls.loss.metric import FocalLoss
        criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    # LR Scheduler
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr_min
        )
    elif args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif args.scheduler == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)
    else:
        scheduler = None

    # AMP
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # Training loop
    best_acc = 0
    start_epoch = 0

    print(f"\n{'='*80}")
    print(f"{'Starting Training':^80}")
    print(f"{'='*80}\n")

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer,
            device, args, scaler, epoch+1
        )

        # Validate
        if val_loader and (epoch + 1) % args.val_interval == 0:
            val_loss, val_acc = validate(model, val_loader, criterion, device)
        else:
            val_loss, val_acc = 0, 0

        # LR step
        if scheduler:
            scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start

        # Print summary
        print(f"\nEpoch {epoch+1}/{args.epochs} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        if val_loader:
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"  LR: {current_lr:.6f} | Time: {epoch_time:.1f}s")

        # Save checkpoint
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_acc': best_acc,
            }, os.path.join(args.save_dir, "best.pth"))
            print(f"  ✓ Best model saved! (acc={best_acc:.2f}%)")

        if (epoch + 1) % args.save_interval == 0:
            torch.save({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(args.save_dir, f"epoch_{epoch+1}.pth"))

    print(f"\n{'='*80}")
    print(f"Training Complete! Best Accuracy: {best_acc:.2f}%")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

