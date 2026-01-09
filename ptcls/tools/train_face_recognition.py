"""Face Recognition Training Script

人脸识别模型训练，支持：
- ArcFace Loss
- CosFace Loss
- SphereFace Loss
- 多种Backbone (ResNet, MobileNet, EfficientNet等)
- 人脸对数据集（验证用）
- 人脸识别数据集（训练用）

数据格式：
1. 训练集：images/person_id/face1.jpg
2. 验证集对：pairs.txt (path1 path2 label)
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ptcls.arch.backbone import build_backbone
from ptcls.loss.metric import ArcFaceLoss, CosFaceLoss
from ptcls.face import FaceAccuracy


class FaceDataset(Dataset):
    """人脸识别训练数据集

    目录结构：
    root/
        person_0/
            face1.jpg
            face2.jpg
        person_1/
            face1.jpg
    """

    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []
        self.person_to_idx = {}

        # 扫描所有人
        person_dirs = sorted([d for d in self.root.iterdir() if d.is_dir()])

        for person_idx, person_dir in enumerate(person_dirs):
            person_name = person_dir.name
            self.person_to_idx[person_name] = person_idx

            # 扫描该人的所有照片
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                for img_path in person_dir.glob(ext):
                    self.samples.append((str(img_path), person_idx))

        self.num_classes = len(person_dirs)
        print(f"[INFO] Found {len(self.samples)} faces from {self.num_classes} persons")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label


class FacePairsDataset(Dataset):
    """人脸对数据集（用于验证）

    pairs.txt格式：
    path1 path2 1  # 同一人
    path1 path2 0  # 不同人
    """

    def __init__(self, pairs_file, transform=None):
        self.transform = transform
        self.pairs = []

        with open(pairs_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    self.pairs.append((parts[0], parts[1], int(parts[2])))

        print(f"[INFO] Loaded {len(self.pairs)} face pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        path1, path2, label = self.pairs[idx]

        img1 = Image.open(path1).convert('RGB')
        img2 = Image.open(path2).convert('RGB')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, label


def parse_args():
    p = argparse.ArgumentParser()

    # Data
    p.add_argument('--train_root', required=True, help='Training data root')
    p.add_argument('--val_pairs', default=None, help='Validation pairs.txt')

    # Model
    p.add_argument('--model', default='resnet50', help='Backbone model')
    p.add_argument('--embedding_size', type=int, default=512)
    p.add_argument('--pretrained', action='store_true')

    # Loss
    p.add_argument('--loss', choices=['arcface', 'cosface', 'sphereface'], default='arcface')
    p.add_argument('--s', type=float, default=64.0, help='Scale for ArcFace/CosFace')
    p.add_argument('--m', type=float, default=0.5, help='Margin for ArcFace/CosFace')

    # Training
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr', type=float, default=0.1)
    p.add_argument('--weight_decay', type=float, default=5e-4)
    p.add_argument('--device', default='cuda')
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--save_dir', required=True)

    # LR Scheduler
    p.add_argument('--scheduler', default='cosine', choices=['cosine', 'step'])
    p.add_argument('--lr_min', type=float, default=1e-5)

    # Validation
    p.add_argument('--val_interval', type=int, default=5)

    return p.parse_args()


def train_epoch(model, criterion, loader, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        # Forward
        features = model(x)
        if isinstance(features, dict):
            features = features.get('embedding', features.get('features', features))

        # ArcFace/CosFace Loss
        loss = criterion(features, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total += y.size(0)

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, device):
    """验证1:1准确率"""
    model.eval()

    metric = FaceAccuracy()

    for img1, img2, label in loader:
        img1 = img1.to(device)
        img2 = img2.to(device)

        feat1 = model(img1)
        feat2 = model(img2)

        if isinstance(feat1, dict):
            feat1 = feat1.get('embedding', feat1.get('features', feat1))
        if isinstance(feat2, dict):
            feat2 = feat2.get('embedding', feat2.get('features', feat2))

        metric.update(feat1, feat2, label)

    results = metric.compute()
    return results


def main():
    args = parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Dataset
    train_ds = FaceDataset(args.train_root, transform=train_transform)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = None
    if args.val_pairs:
        val_ds = FacePairsDataset(args.val_pairs, transform=val_transform)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # Model - 构建Backbone
    from ptcls.arch.head import EmbeddingHead

    backbone = build_backbone(args.model, num_classes=args.embedding_size, pretrained=args.pretrained)

    # 如果backbone没有embedding head，添加一个
    if not hasattr(backbone, 'embedding_size'):
        # 获取backbone输出维度
        with torch.no_grad():
            dummy = torch.randn(1, 3, 112, 112)
            out = backbone(dummy)
            if isinstance(out, dict):
                feat_dim = out.get('features', out.get('embedding', out)).shape[1]
            else:
                feat_dim = out.shape[1]

        # 添加embedding head
        class FaceModel(nn.Module):
            def __init__(self, backbone, feat_dim, embedding_size):
                super().__init__()
                self.backbone = backbone
                self.head = EmbeddingHead(feat_dim, embedding_size)

            def forward(self, x):
                feat = self.backbone(x)
                if isinstance(feat, dict):
                    feat = feat.get('features', feat.get('embedding', feat))
                return self.head(feat)

        model = FaceModel(backbone, feat_dim, args.embedding_size)
    else:
        model = backbone

    model = model.to(device)

    # Loss
    if args.loss == 'arcface':
        criterion = ArcFaceLoss(
            in_features=args.embedding_size,
            num_classes=train_ds.num_classes,
            s=args.s,
            m=args.m
        )
    elif args.loss == 'cosface':
        criterion = CosFaceLoss(
            in_features=args.embedding_size,
            num_classes=train_ds.num_classes,
            s=args.s,
            m=args.m
        )
    else:
        raise ValueError(f"Unknown loss: {args.loss}")

    criterion = criterion.to(device)

    # Optimizer
    optimizer = torch.optim.SGD(
        list(model.parameters()) + list(criterion.parameters()),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay
    )

    # Scheduler
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr_min
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # Training
    best_acc = 0

    print(f"\n{'='*80}")
    print(f"{'Starting Face Recognition Training':^80}")
    print(f"{'='*80}\n")

    for epoch in range(args.epochs):
        # Train
        train_loss = train_epoch(model, criterion, train_loader, optimizer, device, epoch+1)

        # Validate
        if val_loader and (epoch + 1) % args.val_interval == 0:
            val_results = validate(model, val_loader, device)
            val_acc = val_results['accuracy']
            val_thresh = val_results['threshold']
        else:
            val_acc = 0
            val_thresh = 0

        # LR step
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Print
        print(f"Epoch {epoch+1}/{args.epochs}:")
        print(f"  Train Loss: {train_loss:.4f} | LR: {current_lr:.6f}")
        if val_acc > 0:
            print(f"  Val Acc: {val_acc*100:.2f}% | Threshold: {val_thresh:.4f}")

        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'best_acc': best_acc
            }, os.path.join(args.save_dir, 'best.pth'))
            print(f"  ✓ Best model saved! (acc={best_acc*100:.2f}%)")

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model': model.state_dict(),
            }, os.path.join(args.save_dir, f'epoch_{epoch+1}.pth'))

    print(f"\n{'='*80}")
    print(f"Training Complete! Best Accuracy: {best_acc*100:.2f}%")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

