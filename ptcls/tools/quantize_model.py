"""Model Quantization Tool

模型量化工具，支持：
- 动态量化（Dynamic Quantization）
- 静态量化（Static Quantization with Calibration）
- QAT（Quantization-Aware Training）

量化可以显著减小模型大小和加速推理
"""

import argparse
import os
import sys
from pathlib import Path
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ptcls.arch.backbone import build_backbone


class CalibrationDataset(Dataset):
    """校准数据集"""

    def __init__(self, image_dir, transform=None, max_images=100):
        self.image_dir = Path(image_dir)
        self.transform = transform

        self.images = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            self.images.extend(list(self.image_dir.glob(ext)))

        self.images = self.images[:max_images]
        print(f"[INFO] Loaded {len(self.images)} calibration images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img


def load_model(args):
    """加载模型"""
    model = build_backbone(args.model, num_classes=args.num_classes)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location='cpu')
        if 'model' in ckpt:
            model.load_state_dict(ckpt['model'], strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)
        print(f"[INFO] Loaded checkpoint: {args.checkpoint}")

    return model


def dynamic_quantization(model, args):
    """动态量化

    优点：
    - 无需校准数据
    - 简单快速
    - 减小模型大小

    缺点：
    - 性能提升有限
    - 主要用于CPU推理
    """
    print("\n[INFO] Applying dynamic quantization...")

    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.Conv2d},  # 量化层类型
        dtype=torch.qint8
    )

    return quantized_model


def static_quantization(model, args, calibration_loader):
    """静态量化

    优点：
    - 最大的性能提升
    - 最小的模型大小

    缺点：
    - 需要校准数据
    - 可能轻微影响精度
    """
    print("\n[INFO] Applying static quantization with calibration...")

    # 设置量化配置
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')  # 'fbgemm' for x86, 'qnnpack' for ARM

    # 准备模型
    torch.quantization.prepare(model, inplace=True)

    # 校准
    print("[INFO] Calibrating model...")
    model.eval()
    with torch.no_grad():
        for i, images in enumerate(calibration_loader):
            model(images)
            if (i + 1) % 10 == 0:
                print(f"  Calibrated {i+1}/{len(calibration_loader)} batches")

    # 转换为量化模型
    torch.quantization.convert(model, inplace=True)

    return model


def evaluate_model(model, data_loader, device='cpu'):
    """评估模型准确率"""
    model.eval()
    model.to(device)

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            if isinstance(outputs, dict):
                outputs = outputs.get('logits', outputs.get('output', list(outputs.values())[0]))

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / total
    return accuracy


def compare_model_size(model1, model2):
    """比较模型大小"""
    import tempfile

    # 保存模型并比较文件大小
    with tempfile.NamedTemporaryFile() as tmp1:
        torch.save(model1.state_dict(), tmp1.name)
        size1 = os.path.getsize(tmp1.name)

    with tempfile.NamedTemporaryFile() as tmp2:
        torch.save(model2.state_dict(), tmp2.name)
        size2 = os.path.getsize(tmp2.name)

    return size1, size2


def parse_args():
    p = argparse.ArgumentParser(description="Model Quantization Tool")

    # Model
    p.add_argument('--model', required=True, help='Model name')
    p.add_argument('--checkpoint', required=True, help='Model checkpoint')
    p.add_argument('--num_classes', type=int, default=1000)

    # Quantization method
    p.add_argument('--method', required=True,
                   choices=['dynamic', 'static'],
                   help='Quantization method')

    # Calibration (for static quantization)
    p.add_argument('--calib_data', help='Calibration data directory')
    p.add_argument('--calib_images', type=int, default=100,
                   help='Number of calibration images')
    p.add_argument('--batch_size', type=int, default=32)

    # Evaluation (optional)
    p.add_argument('--eval_data', help='Evaluation data directory')

    # Output
    p.add_argument('--output', required=True, help='Output quantized model path')

    return p.parse_args()


def main():
    args = parse_args()

    # 加载原始模型
    print("[INFO] Loading original model...")
    model = load_model(args)
    model.eval()

    # 准备校准数据（静态量化需要）
    calibration_loader = None
    if args.method == 'static':
        if not args.calib_data:
            print("Error: --calib_data required for static quantization")
            return

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        calib_dataset = CalibrationDataset(
            args.calib_data,
            transform=transform,
            max_images=args.calib_images
        )

        calibration_loader = DataLoader(
            calib_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2
        )

    # 执行量化
    if args.method == 'dynamic':
        quantized_model = dynamic_quantization(model, args)
    elif args.method == 'static':
        quantized_model = static_quantization(copy.deepcopy(model), args, calibration_loader)

    # 比较模型大小
    size_orig, size_quant = compare_model_size(model, quantized_model)

    print(f"\n{'='*80}")
    print(f"{'Quantization Results':^80}")
    print(f"{'='*80}")
    print(f"Method: {args.method.upper()}")
    print(f"Original Size: {size_orig / 1024 / 1024:.2f} MB")
    print(f"Quantized Size: {size_quant / 1024 / 1024:.2f} MB")
    print(f"Compression Ratio: {size_orig / size_quant:.2f}x")
    print(f"{'='*80}\n")

    # 保存量化模型
    torch.save(quantized_model.state_dict(), args.output)
    print(f"[INFO] Quantized model saved to: {args.output}")

    # 可选：评估精度
    if args.eval_data:
        print("\n[INFO] Evaluating models...")

        # 这里需要准备评估数据集
        # eval_loader = ...

        # orig_acc = evaluate_model(model, eval_loader)
        # quant_acc = evaluate_model(quantized_model, eval_loader)

        # print(f"Original Accuracy: {orig_acc:.2f}%")
        # print(f"Quantized Accuracy: {quant_acc:.2f}%")
        # print(f"Accuracy Drop: {orig_acc - quant_acc:.2f}%")

        print("[INFO] Evaluation requires a labeled dataset (not implemented in this demo)")


if __name__ == "__main__":
    main()

