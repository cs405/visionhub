"""YOLO Detection + Classification System

完整的检测+分类流程：
1. YOLO检测目标
2. Crop检测框
3. 分类模型识别
4. 返回结果（检测框 + 类别 + 置信度）

类似于检索系统，但用分类模型代替检索
"""

import os
import sys
from pathlib import Path
import yaml

import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ptcls.arch.backbone import build_backbone


class YOLODetectionClassification:
    """YOLO检测 + 分类识别系统"""

    def __init__(self, yolo_model_path, cls_model_name, cls_checkpoint, num_classes,
                 class_names=None, device='cuda', conf_threshold=0.25):
        """
        Args:
            yolo_model_path: YOLO检测模型路径 (.pt)
            cls_model_name: 分类模型名称 (resnet50, efficientnet_b0等)
            cls_checkpoint: 分类模型权重路径
            num_classes: 分类类别数
            class_names: 类别名称列表
            device: 设备
            conf_threshold: YOLO检测置信度阈值
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.conf_threshold = conf_threshold

        # 加载YOLO模型
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO(yolo_model_path)
            print(f"[INFO] YOLO model loaded: {yolo_model_path}")
        except ImportError:
            raise ImportError("Please install ultralytics: pip install ultralytics")

        # 加载分类模型
        self.cls_model = build_backbone(cls_model_name, num_classes=num_classes)

        if cls_checkpoint:
            ckpt = torch.load(cls_checkpoint, map_location='cpu')
            if 'model' in ckpt:
                self.cls_model.load_state_dict(ckpt['model'])
            else:
                self.cls_model.load_state_dict(ckpt)
            print(f"[INFO] Classification model loaded: {cls_checkpoint}")

        self.cls_model = self.cls_model.to(self.device)
        self.cls_model.eval()

        # 分类预处理
        self.cls_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.num_classes = num_classes

    @classmethod
    def from_config(cls, config_path):
        """从配置文件加载"""
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        return cls(
            yolo_model_path=cfg['yolo_model'],
            cls_model_name=cfg['cls_model'],
            cls_checkpoint=cfg['cls_checkpoint'],
            num_classes=cfg['num_classes'],
            class_names=cfg.get('class_names'),
            device=cfg.get('device', 'cuda'),
            conf_threshold=cfg.get('conf_threshold', 0.25)
        )

    def detect_and_classify(self, image_path, save_result=False, save_path=None):
        """检测+分类完整流程

        Args:
            image_path: 输入图片路径
            save_result: 是否保存可视化结果
            save_path: 保存路径

        Returns:
            results: 列表，每个元素包含：
                - box: [x1, y1, x2, y2]
                - det_class: YOLO检测类别
                - det_conf: YOLO检测置信度
                - cls_class: 分类识别类别
                - cls_conf: 分类置信度
                - cls_class_name: 分类类别名称
        """
        # 1. YOLO检测
        det_results = self.yolo_model.predict(
            image_path,
            conf=self.conf_threshold,
            verbose=False
        )[0]

        # 读取原图
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")

        results = []

        # 2. 对每个检测框进行分类
        if det_results.boxes is not None and len(det_results.boxes) > 0:
            boxes = det_results.boxes.xyxy.cpu().numpy()
            det_confs = det_results.boxes.conf.cpu().numpy()
            det_classes = det_results.boxes.cls.cpu().numpy().astype(int)

            for i, (box, det_conf, det_cls) in enumerate(zip(boxes, det_confs, det_classes)):
                x1, y1, x2, y2 = map(int, box)

                # Crop目标区域
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                # 转换为PIL Image
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crop_pil = Image.fromarray(crop_rgb)

                # 分类预测
                crop_tensor = self.cls_transform(crop_pil).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    logits = self.cls_model(crop_tensor)
                    probs = torch.softmax(logits, dim=1)
                    cls_conf, cls_class = probs.max(1)
                    cls_conf = cls_conf.item()
                    cls_class = cls_class.item()

                cls_name = self.class_names[cls_class] if cls_class < len(self.class_names) else f"class_{cls_class}"

                results.append({
                    'box': [x1, y1, x2, y2],
                    'det_class': int(det_cls),
                    'det_conf': float(det_conf),
                    'cls_class': int(cls_class),
                    'cls_conf': float(cls_conf),
                    'cls_class_name': cls_name
                })

        # 3. 可视化
        if save_result and results:
            self.visualize_results(img, results, save_path or 'result.jpg')

        return results

    def visualize_results(self, img, results, save_path):
        """可视化检测+分类结果"""
        vis_img = img.copy()

        for res in results:
            x1, y1, x2, y2 = res['box']
            cls_name = res['cls_class_name']
            cls_conf = res['cls_conf']

            # 绘制框
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 绘制标签
            label = f"{cls_name} {cls_conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(vis_img, (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(vis_img, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        cv2.imwrite(save_path, vis_img)
        print(f"[INFO] Result saved to: {save_path}")

    def batch_predict(self, image_dir, save_dir=None):
        """批量预测"""
        image_dir = Path(image_dir)
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        all_results = {}

        for img_path in image_dir.glob('*.jpg'):
            results = self.detect_and_classify(
                str(img_path),
                save_result=(save_dir is not None),
                save_path=str(save_dir / img_path.name) if save_dir else None
            )
            all_results[img_path.name] = results

            print(f"[{img_path.name}] Found {len(results)} objects")
            for i, res in enumerate(results):
                print(f"  {i+1}. {res['cls_class_name']} (conf={res['cls_conf']:.3f})")

        return all_results


def main():
    """命令行接口"""
    import argparse

    parser = argparse.ArgumentParser(description="YOLO Detection + Classification")
    parser.add_argument('--config', help='Config YAML file')
    parser.add_argument('--yolo_model', default='yolov8n.pt', help='YOLO model path')
    parser.add_argument('--cls_model', default='resnet50', help='Classification model name')
    parser.add_argument('--cls_checkpoint', required=True, help='Classification checkpoint')
    parser.add_argument('--num_classes', type=int, required=True, help='Number of classes')
    parser.add_argument('--class_names', nargs='+', help='Class names')
    parser.add_argument('--image', help='Input image path')
    parser.add_argument('--image_dir', help='Input image directory')
    parser.add_argument('--save_dir', default='results', help='Output directory')
    parser.add_argument('--device', default='cuda', help='Device')
    parser.add_argument('--conf', type=float, default=0.25, help='Detection confidence threshold')

    args = parser.parse_args()

    # 构建系统
    if args.config:
        system = YOLODetectionClassification.from_config(args.config)
    else:
        system = YOLODetectionClassification(
            yolo_model_path=args.yolo_model,
            cls_model_name=args.cls_model,
            cls_checkpoint=args.cls_checkpoint,
            num_classes=args.num_classes,
            class_names=args.class_names,
            device=args.device,
            conf_threshold=args.conf
        )

    # 预测
    if args.image:
        results = system.detect_and_classify(args.image, save_result=True,
                                            save_path=f"{args.save_dir}/result.jpg")
        print(f"\nFound {len(results)} objects:")
        for i, res in enumerate(results):
            print(f"{i+1}. {res['cls_class_name']} (conf={res['cls_conf']:.3f})")

    elif args.image_dir:
        system.batch_predict(args.image_dir, args.save_dir)

    else:
        print("Error: Please provide --image or --image_dir")


if __name__ == "__main__":
    main()

