"""YOLO Classification Dataset Adapter

支持YOLO-CLS格式数据集：
- images/
  - train/
    - class1/
      - img1.jpg
      - img2.jpg
  - val/
    - class1/
      - img1.jpg

或YOLO-Det crop为分类数据集
"""

import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class YOLOClassificationDataset(Dataset):
    """YOLO分类数据集

    支持两种格式：
    1. YOLO-CLS标准格式（文件夹结构）
    2. 从YOLO-Det crop转换
    """

    def __init__(self, root, transform=None, class_names=None):
        """
        Args:
            root: 数据根目录
            transform: torchvision transforms
            class_names: 类别名称列表（可选，从data.yaml读取）
        """
        self.root = Path(root)
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        self.classes = []

        # 扫描目录
        self._scan_directory()

        if class_names:
            self.class_names = class_names
        else:
            self.class_names = [f"class_{i}" for i in range(len(self.classes))]

    def _scan_directory(self):
        """扫描目录构建样本列表"""
        # 检查是否是标准分类目录结构
        subdirs = [d for d in self.root.iterdir() if d.is_dir()]

        if not subdirs:
            raise ValueError(f"No subdirectories found in {self.root}")

        # 遍历每个类别文件夹
        for class_idx, class_dir in enumerate(sorted(subdirs)):
            class_name = class_dir.name
            self.class_to_idx[class_name] = class_idx
            self.classes.append(class_name)

            # 收集该类别的所有图片
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                for img_path in class_dir.glob(ext):
                    self.samples.append((str(img_path), class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Load image
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label


def convert_yolo_det_to_cls(det_images_dir, det_labels_dir, output_dir, data_yaml=None):
    """将YOLO检测数据集转换为分类数据集

    从检测框crop出来，按类别组织

    Args:
        det_images_dir: YOLO检测images目录
        det_labels_dir: YOLO检测labels目录
        output_dir: 输出的分类数据集目录
        data_yaml: YOLO data.yaml路径，用于获取类别名
    """
    import cv2
    import yaml
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 读取类别名
    class_names = {}
    if data_yaml:
        with open(data_yaml) as f:
            data = yaml.safe_load(f)
            class_names = {i: name for i, name in enumerate(data.get('names', []))}

    # 遍历所有标签文件
    label_files = list(Path(det_labels_dir).glob("*.txt"))

    for label_file in label_files:
        # 读取对应图片
        img_name = label_file.stem
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            candidate = Path(det_images_dir) / f"{img_name}{ext}"
            if candidate.exists():
                img_path = candidate
                break

        if not img_path:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]

        # 读取标签
        with open(label_file) as f:
            for line_idx, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                class_id = int(parts[0])
                x_center, y_center, box_w, box_h = map(float, parts[1:5])

                # 转换为像素坐标
                x1 = int((x_center - box_w / 2) * w)
                y1 = int((y_center - box_h / 2) * h)
                x2 = int((x_center + box_w / 2) * w)
                y2 = int((y_center + box_h / 2) * h)

                # Crop
                crop = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                if crop.size == 0:
                    continue

                # 保存到对应类别文件夹
                class_name = class_names.get(class_id, f"class_{class_id}")
                class_dir = output_dir / class_name
                class_dir.mkdir(exist_ok=True)

                save_path = class_dir / f"{img_name}_{line_idx}.jpg"
                cv2.imwrite(str(save_path), crop)

    print(f"[INFO] Conversion complete! Output: {output_dir}")


if __name__ == "__main__":
    # 示例：转换YOLO检测数据集为分类数据集
    import sys
    if len(sys.argv) < 4:
        print("Usage: python yolo_cls_dataset.py <det_images> <det_labels> <output_dir> [data.yaml]")
        sys.exit(1)

    det_images = sys.argv[1]
    det_labels = sys.argv[2]
    output = sys.argv[3]
    data_yaml = sys.argv[4] if len(sys.argv) > 4 else None

    convert_yolo_det_to_cls(det_images, det_labels, output, data_yaml)

