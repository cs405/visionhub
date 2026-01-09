"""Batch Inference Tool

批量推理工具，支持：
- 批量图片分类
- 批量特征提取
- 多进程加速
- 结果导出（CSV, JSON）
"""

import argparse
import os
import sys
import time
import json
import csv
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ptcls.arch.backbone import build_backbone


class ImageFolderDataset(Dataset):
    """图片文件夹数据集"""

    def __init__(self, image_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform

        self.images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            self.images.extend(list(self.image_dir.rglob(ext)))

        print(f"[INFO] Found {len(self.images)} images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, str(img_path)


def load_model(args):
    """加载模型"""
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    model = build_backbone(args.model, num_classes=args.num_classes)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location='cpu')
        if 'model' in ckpt:
            model.load_state_dict(ckpt['model'], strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)

    model = model.to(device)
    model.eval()

    return model, device


def batch_classify(model, loader, device, class_names=None, top_k=5):
    """批量分类"""
    all_results = []

    with torch.no_grad():
        for images, paths in tqdm(loader, desc="Classifying"):
            images = images.to(device)

            outputs = model(images)
            if isinstance(outputs, dict):
                outputs = outputs.get('logits', outputs.get('output', list(outputs.values())[0]))

            probs = torch.softmax(outputs, dim=1)
            topk_probs, topk_idx = probs.topk(top_k, dim=1)

            for i in range(len(paths)):
                predictions = []
                for prob, idx in zip(topk_probs[i], topk_idx[i]):
                    cls_name = class_names[idx] if class_names and idx < len(class_names) else f"class_{idx}"
                    predictions.append({
                        'class_id': int(idx),
                        'class_name': cls_name,
                        'probability': float(prob)
                    })

                all_results.append({
                    'image_path': paths[i],
                    'predictions': predictions
                })

    return all_results


def batch_extract_features(model, loader, device):
    """批量特征提取"""
    all_features = []
    all_paths = []

    with torch.no_grad():
        for images, paths in tqdm(loader, desc="Extracting features"):
            images = images.to(device)

            outputs = model(images)
            if isinstance(outputs, dict):
                feat = outputs.get('embedding', outputs.get('features', list(outputs.values())[0]))
            else:
                feat = outputs

            # L2归一化
            feat = torch.nn.functional.normalize(feat, p=2, dim=1)

            all_features.append(feat.cpu().numpy())
            all_paths.extend(paths)

    features = np.concatenate(all_features, axis=0)

    return features, all_paths


def save_classification_results(results, output_path, format='json'):
    """保存分类结果"""
    if format == 'json':
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

    elif format == 'csv':
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Image Path', 'Top1 Class', 'Top1 Probability'])

            for res in results:
                img_path = res['image_path']
                top1 = res['predictions'][0]
                writer.writerow([img_path, top1['class_name'], top1['probability']])

    print(f"[INFO] Results saved to: {output_path}")


def save_features(features, paths, output_path):
    """保存特征向量"""
    np.savez(
        output_path,
        features=features,
        paths=np.array(paths)
    )
    print(f"[INFO] Features saved to: {output_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Batch Inference Tool")

    # Task
    p.add_argument('--task', required=True,
                   choices=['classify', 'extract_features'],
                   help='Inference task')

    # Model
    p.add_argument('--model', required=True, help='Model name')
    p.add_argument('--checkpoint', required=True, help='Model checkpoint')
    p.add_argument('--num_classes', type=int, default=1000)
    p.add_argument('--class_names', help='Class names file')

    # Data
    p.add_argument('--image_dir', required=True, help='Input image directory')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--num_workers', type=int, default=4)

    # Classification specific
    p.add_argument('--top_k', type=int, default=5)

    # Output
    p.add_argument('--output', required=True, help='Output file path')
    p.add_argument('--format', choices=['json', 'csv', 'npz'], default='json')

    # Device
    p.add_argument('--device', default='cuda')

    return p.parse_args()


def main():
    args = parse_args()

    # 加载模型
    print("[INFO] Loading model...")
    model, device = load_model(args)

    # 加载类别名
    class_names = None
    if args.class_names and os.path.exists(args.class_names):
        with open(args.class_names) as f:
            class_names = [line.strip() for line in f]

    # 准备数据
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = ImageFolderDataset(args.image_dir, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # 执行推理
    print(f"\n[INFO] Starting {args.task}...")
    start_time = time.time()

    if args.task == 'classify':
        results = batch_classify(model, loader, device, class_names, args.top_k)
        save_classification_results(results, args.output, args.format)

    elif args.task == 'extract_features':
        features, paths = batch_extract_features(model, loader, device)
        save_features(features, paths, args.output)

    elapsed_time = time.time() - start_time

    print(f"\n{'='*80}")
    print(f"{'Batch Inference Complete':^80}")
    print(f"{'='*80}")
    print(f"Task: {args.task}")
    print(f"Images processed: {len(dataset)}")
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    print(f"Speed: {len(dataset)/elapsed_time:.2f} images/sec")
    print(f"Output: {args.output}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

