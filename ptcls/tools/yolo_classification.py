"""YOLO Classification Model Training Integration

集成Ultralytics YOLO-CLS训练能力到visionhub框架
"""

import os
import sys
import yaml
from pathlib import Path


def train_yolo_cls(data_yaml, model='yolov8n-cls.pt', epochs=100, imgsz=224, batch=128, device='cuda', project='runs/classify', name='exp'):
    """使用Ultralytics训练YOLO分类模型

    Args:
        data_yaml: YOLO分类数据配置文件
        model: YOLO-CLS模型（yolov8n-cls.pt, yolov8s-cls.pt等）
        epochs: 训练轮数
        imgsz: 图片尺寸
        batch: batch size
        device: 设备
        project: 项目目录
        name: 实验名称

    Returns:
        训练结果路径
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("ultralytics not installed. Run: pip install ultralytics")

    # 初始化模型
    model = YOLO(model)

    # 训练
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        pretrained=True,
        optimizer='SGD',
        lr0=0.01,
        momentum=0.9,
        weight_decay=5e-4,
        warmup_epochs=3,
        save_period=10,
        verbose=True
    )

    return results


def prepare_yolo_cls_yaml(train_dir, val_dir, save_path='data_cls.yaml', nc=None, names=None):
    """准备YOLO分类数据配置文件

    Args:
        train_dir: 训练集目录（包含类别子文件夹）
        val_dir: 验证集目录
        save_path: 保存路径
        nc: 类别数（自动检测）
        names: 类别名称列表（自动检测）

    Returns:
        yaml文件路径
    """
    train_dir = Path(train_dir)
    val_dir = Path(val_dir)

    # 自动检测类别
    if nc is None or names is None:
        class_dirs = [d for d in train_dir.iterdir() if d.is_dir()]
        nc = len(class_dirs)
        names = [d.name for d in sorted(class_dirs)]

    # 构建YAML
    data = {
        'path': str(train_dir.parent.absolute()),  # 数据集根目录
        'train': str(train_dir.name),  # 相对于path的训练目录
        'val': str(val_dir.name),      # 相对于path的验证目录
        'nc': nc,
        'names': names
    }

    # 保存
    with open(save_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

    print(f"[INFO] YOLO-CLS data.yaml saved to: {save_path}")
    print(f"[INFO] Classes: {nc}, Names: {names}")

    return save_path


def export_yolo_cls(model_path, format='onnx', imgsz=224, simplify=True):
    """导出YOLO分类模型

    Args:
        model_path: 训练好的.pt模型路径
        format: 导出格式（onnx, torchscript, tflite等）
        imgsz: 输入尺寸
        simplify: 简化ONNX模型

    Returns:
        导出路径
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("ultralytics not installed")

    model = YOLO(model_path)

    # 导出
    export_path = model.export(
        format=format,
        imgsz=imgsz,
        simplify=simplify
    )

    print(f"[INFO] Model exported to: {export_path}")
    return export_path


def predict_yolo_cls(model_path, source, imgsz=224, conf=0.25, save=True, project='runs/classify', name='predict'):
    """使用YOLO分类模型预测

    Args:
        model_path: 模型路径
        source: 图片路径或目录
        imgsz: 图片尺寸
        conf: 置信度阈值
        save: 是否保存结果
        project: 项目目录
        name: 实验名称

    Returns:
        预测结果
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("ultralytics not installed")

    model = YOLO(model_path)

    # 预测
    results = model.predict(
        source=source,
        imgsz=imgsz,
        conf=conf,
        save=save,
        project=project,
        name=name
    )

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YOLO Classification Training")
    parser.add_argument('--task', choices=['train', 'export', 'predict', 'prepare_yaml'], required=True)
    parser.add_argument('--model', default='yolov8n-cls.pt', help='YOLO-CLS model')
    parser.add_argument('--data', help='data.yaml path')
    parser.add_argument('--train_dir', help='Training directory (for prepare_yaml)')
    parser.add_argument('--val_dir', help='Validation directory (for prepare_yaml)')
    parser.add_argument('--source', help='Image source for prediction')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--imgsz', type=int, default=224)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--format', default='onnx', help='Export format')

    args = parser.parse_args()

    if args.task == 'prepare_yaml':
        if not args.train_dir or not args.val_dir:
            print("Error: --train_dir and --val_dir required for prepare_yaml")
            sys.exit(1)
        prepare_yolo_cls_yaml(args.train_dir, args.val_dir, args.data or 'data_cls.yaml')

    elif args.task == 'train':
        if not args.data:
            print("Error: --data required for training")
            sys.exit(1)
        train_yolo_cls(args.data, args.model, args.epochs, args.imgsz, args.batch, args.device)

    elif args.task == 'export':
        if not args.model:
            print("Error: --model required for export")
            sys.exit(1)
        export_yolo_cls(args.model, args.format, args.imgsz)

    elif args.task == 'predict':
        if not args.model or not args.source:
            print("Error: --model and --source required for predict")
            sys.exit(1)
        predict_yolo_cls(args.model, args.source, args.imgsz)

