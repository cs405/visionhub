"""Classification Model Evaluation

评估分类模型性能：
- Top-1/Top-5 Accuracy
- Confusion Matrix
- Per-class Accuracy
- Classification Report
"""

import argparse
import os
import sys
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ptcls.arch.backbone import build_backbone
from ptcls.data.datasets.yolo_cls_dataset import YOLOClassificationDataset
from ptcls.data.datasets.image_folder_pairs import ImageFolderDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Model name or checkpoint path")
    p.add_argument("--checkpoint", default=None, help="Checkpoint .pth file")
    p.add_argument("--data_root", required=True, help="Test dataset root")
    p.add_argument("--num_classes", type=int, required=True)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--save_dir", default="eval_results")
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    """评估模型"""
    model.eval()

    all_preds = []
    all_targets = []
    all_probs = []

    for x, y in tqdm(loader, desc="Evaluating"):
        x = x.to(device)
        outputs = model(x)

        probs = torch.softmax(outputs, dim=1)
        _, preds = outputs.max(1)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(y.numpy())
        all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    # Top-1 Accuracy
    top1_acc = 100.0 * (all_preds == all_targets).sum() / len(all_targets)

    # Top-5 Accuracy
    top5_preds = np.argsort(all_probs, axis=1)[:, -5:]
    top5_correct = sum([t in p for t, p in zip(all_targets, top5_preds)])
    top5_acc = 100.0 * top5_correct / len(all_targets)

    # Per-class accuracy
    per_class_acc = {}
    for cls in range(num_classes):
        mask = all_targets == cls
        if mask.sum() > 0:
            acc = 100.0 * (all_preds[mask] == all_targets[mask]).sum() / mask.sum()
            per_class_acc[cls] = float(acc)

    # Confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(all_targets, all_preds)

    results = {
        'top1_accuracy': float(top1_acc),
        'top5_accuracy': float(top5_acc),
        'per_class_accuracy': per_class_acc,
        'confusion_matrix': cm.tolist(),
        'num_samples': len(all_targets)
    }

    return results


def main():
    args = parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load dataset
    print("[INFO] Loading dataset...")
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if os.path.exists(os.path.join(args.data_root, "class_0")):
        # YOLO-CLS format
        dataset = YOLOClassificationDataset(args.data_root, transform=transform)
    else:
        # ImageFolder format
        dataset = ImageFolderDataset(args.data_root, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"[INFO] Dataset: {len(dataset)} samples")

    # Load model
    print(f"[INFO] Loading model: {args.model}")
    model = build_backbone(args.model, num_classes=args.num_classes)

    if args.checkpoint:
        print(f"[INFO] Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location='cpu')
        if 'model' in ckpt:
            model.load_state_dict(ckpt['model'])
        else:
            model.load_state_dict(ckpt)

    model = model.to(device)

    # Evaluate
    print("\n[INFO] Starting evaluation...")
    results = evaluate(model, loader, device, args.num_classes)

    # Print results
    print(f"\n{'='*80}")
    print(f"{'Evaluation Results':^80}")
    print(f"{'='*80}")
    print(f"Top-1 Accuracy: {results['top1_accuracy']:.2f}%")
    print(f"Top-5 Accuracy: {results['top5_accuracy']:.2f}%")
    print(f"Total Samples: {results['num_samples']}")

    print(f"\n{'Per-Class Accuracy':^80}")
    print(f"{'-'*80}")
    for cls, acc in sorted(results['per_class_accuracy'].items()):
        print(f"  Class {cls}: {acc:.2f}%")

    # Save results
    results_path = os.path.join(args.save_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[INFO] Results saved to: {results_path}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

