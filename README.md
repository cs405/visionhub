# visionhub - Professional Visual Intelligence Toolkit

<div align="center">

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/pytorch-1.10%2B-orange)
![License](https://img.shields.io/badge/license-Apache%202.0-green)

**全功能端到端视觉智能工具包**

[快速开始](#快速开始) • [文档](docs/USER_GUIDE.md) • [特性](#核心特性) • [安装](#安装)

</div>

---

## 🎯 项目简介

**visionhub** 是一个由 **JKDCPPZzz** 独立开发的全功能端到端视觉智能工具包，专为工业级应用和前沿研究设计，提供高性能、模块化的视觉解决方案。

- ✅ **图像分类**：支持1000+类别的标准分类任务
- ✅ **图像检索**：基于向量检索的高性能相似图片搜索
- ✅ **人脸识别**：包含1:1验证与1:N识别的完整流水线
- ✅ **目标检测集成**：无缝对接YOLO系列实现检测+识别联合推理
- ✅ **工业级部署**：支持ONNX、TensorRT、模型量化及HTTP服务
- ✅ **进阶训练**：内置知识蒸馏框架与丰富的度量学习算子

---

## ⭐ 核心特性

| 特性 | 数量/描述 |
|------|----------|
| **Backbone模型** | 85+ (ResNet, EfficientNet, ViT, Swin, MobileFaceNet, IR-Net等) |
| **Loss函数** | 50+ (度量学习、蒸馏、分类Loss) |
| **数据增强** | 16种 (AutoAugment, Mixup, CutMix, GridMask等) |
| **部署支持** | ONNX, TensorRT, TorchScript, 量化 |
| **第三方集成** | YOLO (Ultralytics), Faiss检索 |
| **架构特点** | 模块化设计，易于扩展与重构 |

---

## 🚀 快速开始

### 安装

```bash
# 从源码安装
git clone https://github.com/cs405/visionhub.git
cd visionhub
pip install -e .
```

### 图像分类示例

```python
from visionhub.ptcls.arch.backbone import build_backbone
from torchvision import transforms
from PIL import Image
import torch

# 1. 加载模型
model = build_backbone('resnet50', num_classes=1000, pretrained=True)
model.eval()

# 2. 预处理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. 推理
img = Image.open('demo.jpg')
img_tensor = transform(img).unsqueeze(0)

with torch.no_grad():
    output = model(img_tensor)
    prob = torch.softmax(output, dim=1)
    top5_prob, top5_idx = prob.topk(5)

print(f"Top-5: {list(zip(top5_idx[0].tolist(), top5_prob[0].tolist()))}")
```

### 图像检索示例

```python
from visionhub.ptcls.rec import RecPredictor

# 构建检索与搜索
predictor = RecPredictor(model_name='resnet50', gallery_path='label_gallery/index')
results = predictor.search('demo.jpg', top_k=5)
```

### YOLO + 识别系统

```python
from visionhub.ptcls.tools.yolo_det_classification import YOLODetectionClassification

# 创建检测+识别系统
system = YOLODetectionClassification(
    yolo_model_path='yolov12n.pt',
    cls_model_name='resnet50',
    cls_checkpoint='classifier.pth',
    num_classes=100
)

# 推理
results = system.detect_and_classify('image.jpg', save_result=True)
```

---

## 📚 完整文档

| 文档 | 描述 |
|------|------|
| [**完整使用指南**](docs/USER_GUIDE.md) | 所有功能的详细使用教程 |
| [数据增强&训练](docs/PRIORITY1_GUIDE.md) | 数据增强与训练技巧 |
| [部署指南](docs/DEPLOYMENT_GUIDE.md) | ONNX/TensorRT部署完整教程 |

---

## 💡 主要功能

### 1. 图像分类

```bash
# 训练
python tools/train_classification.py \
  --data_root ./data \
  --model resnet50 \
  --epochs 100 \
  --batch_size 128 \
  --device cuda

# 评估
python tools/eval_classification.py \
  --model resnet50 \
  --checkpoint best.pth \
  --data_root ./data/test
```

### 2. 图像检索

```bash
# 训练检索模型（支持YOLO数据）
python tools/train_rec_kd.py \
  --yolo_images dataset/train \
  --yolo_labels dataset/train \
  --data_yaml dataset/data.yaml \
  --use_pk --P 8 --K 4 \
  --w_triplet 1.0 --w_circle 0.2

# 构建检索库
python tools/build_gallery.py -c configs/shitu/rec_faiss_demo.yaml

# 评估检索效果
python tools/eval_retrieval.py \
  -c configs/shitu/rec_faiss_demo.yaml \
  --gallery_images dataset/val \
  --query_images dataset/test
```

### 3. 人脸识别

```bash
# 训练
python tools/train_face_recognition.py \
  --train_root faces/train \
  --model ir_net_50 \
  --loss arcface

# 1:1验证
python tools/face_recognition_inference.py \
  --task verify \
  --image1 face1.jpg \
  --image2 face2.jpg

# 1:N识别
python tools/face_recognition_inference.py \
  --task identify \
  --query query.jpg \
  --gallery_dir faces/gallery
```

### 4. 模型部署

```bash
# 导出ONNX
python tools/export_model.py \
  --model resnet50 \
  --checkpoint best.pth \
  --format onnx --simplify

# TensorRT加速
python deploy/tensorrt_predictor.py \
  --mode build \
  --onnx model.onnx \
  --engine model.engine --fp16

# HTTP服务
python deploy/http_server.py \
  --model resnet50 \
  --checkpoint best.pth \
  --port 8080
```

---

## 📊 Backbone模型库

### CNN系列（60+）
- **ResNet家族**: ResNet18/34/50/101/152, ResNeXt, WideResNet, SE-ResNet
- **轻量级**: MobileNetV2/V3, EfficientNet B0-B7, GhostNet, ShuffleNet
- **人脸识别**: MobileFaceNet, IR-Net-50/100/152
- **高级CNN**: DenseNet, DLA, DPN, Inception, Xception

### Transformer系列（25+）
- **ViT**: ViT-Tiny/Small/Base, DeiT
- **Swin**: Swin-Tiny/Small/Base
- **其他**: ConvNeXt, MobileViT, CSWin, LeViT, PVT-V2

👉 [查看完整模型列表](docs/USER_GUIDE.md#5-backbone模型库)

---

## 🎓 Loss函数库

### 度量学习Loss（28个）
- **基础**: ArcFace, CosFace, SphereFace, Triplet, Center
- **高级**: Circle, SupCon, MSM, XBM, SoftTriple, Angular

### 蒸馏Loss（16个）
- **基础**: KLDiv, DKD, RKD
- **高级**: AFD, ReviewKD, CRD, MGD

### 分类Loss（4个）
- Focal, LabelSmoothing, Asymmetric

👉 [查看完整Loss列表](docs/USER_GUIDE.md#6-loss函数库)

---

## 📦 数据增强

**16种增强策略**：
- 基础：Flip, Rotate, Crop, ColorJitter
- 高级：AutoAugment, RandAugment
- 混合：Mixup, CutMix, FMix
- 遮挡：RandomErasing, GridMask, HideAndSeek

---

## 🎯 YOLO集成

visionhub完美集成YOLO（Ultralytics），支持：

### YOLO检测 + 分类
```python
system = YOLODetectionClassification(
    yolo_model_path='yolov8n.pt',
    cls_model_name='resnet50',
    cls_checkpoint='classifier.pth'
)
results = system.detect_and_classify('image.jpg')
```

### YOLO检测 + 检索
```bash
python run_pipeline.py all \
  -c configs/shitu/rec_faiss_demo.yaml \
  --yolo_train_images dataset/train \
  --yolo_train_labels dataset/train \
  --data_yaml dataset/data.yaml
```

**数据格式（YOLO标准）**：
```
dataset/
  train/
    img1.jpg
    img1.txt  # class_id x_center y_center width height
  data.yaml   # names: [class1, class2, ...]
```

---

## 🚀 性能对比

### 推理速度

| 引擎 | 延迟 | FPS | 加速比 |
|------|------|-----|--------|
| PyTorch (CPU) | 5.2ms | 192 | 1.0x |
| ONNX Runtime (CPU) | 3.8ms | 263 | 1.4x |
| ONNX Runtime (GPU) | 2.5ms | 400 | 2.1x |
| **TensorRT FP16** | **0.9ms** | **1111** | **5.8x** ✨ |

### 模型大小

| 方法 | 大小 | 压缩比 |
|------|------|--------|
| FP32 | 98 MB | 1.0x |
| FP16 | 49 MB | 2.0x |
| **INT8量化** | **25 MB** | **3.9x** ✨ |

---

## 📖 教程和示例

### 初学者教程
1. [5分钟快速开始](docs/USER_GUIDE.md#3-快速开始)
2. [图像分类训练](docs/USER_GUIDE.md#81-标准分类训练)
3. [模型评估](docs/USER_GUIDE.md#41-图像分类)

### 进阶教程
1. [图像检索完整流程](docs/USER_GUIDE.md#82-检索模型训练)
2. [YOLO + 检索集成](docs/USER_GUIDE.md#9-yolo集成)
3. [知识蒸馏训练](docs/USER_GUIDE.md#44-知识蒸馏)

### 部署教程
1. [ONNX导出和推理](docs/DEPLOYMENT_GUIDE.md#1-模型导出)
2. [TensorRT加速](docs/DEPLOYMENT_GUIDE.md#build-engine-from-onnx)
3. [模型量化](docs/DEPLOYMENT_GUIDE.md#3-模型量化)
4. [HTTP服务部署](docs/DEPLOYMENT_GUIDE.md#4-http服务部署)

---

## 🗂️ 项目结构

```
visionhub/
├── ptcls/                    # 核心代码
│   ├── arch/                 # 模型架构
│   │   ├── backbone/         # Backbone模型
│   │   ├── head/             # 分类头
│   │   └── __init__.py
│   ├── loss/                 # Loss函数
│   │   ├── metric/           # 度量学习Loss
│   │   ├── distillation/     # 蒸馏Loss
│   │   └── __init__.py
│   ├── data/                 # 数据加载
│   │   ├── datasets/         # 数据集
│   │   ├── augmentation/     # 数据增强
│   │   └── __init__.py
│   ├── metric/               # 评估指标
│   ├── rec/                  # 检索模块
│   ├── face/                 # 人脸识别
│   └── utils/                # 工具函数
├── tools/                    # 训练/评估工具
│   ├── train_classification.py
│   ├── train_rec_kd.py
│   ├── train_face_recognition.py
│   ├── export_model.py
│   └── ...
├── deploy/                   # 部署工具
│   ├── onnx_predictor.py
│   ├── tensorrt_predictor.py
│   ├── http_server.py
│   └── batch_inference.py
├── configs/                  # 配置文件
├── docs/                     # 文档
├── README.md                 # 本文件
├── setup.py                  # 安装脚本
└── pyproject.toml            # 项目配置
```

---

## 🤝 贡献

欢迎贡献！请查看 [CONTRIBUTING.md](CONTRIBUTING.md)

---

## 📄 许可证

本项目采用 [Apache 2.0](LICENSE) 许可证

---

## 🙏 致谢

感谢以下开源项目和框架的支持：

- **PyTorch**: 深度学习框架
- **Ultralytics**: YOLO实现
- **Faiss**: 高性能向量检索库
- **ONNX Runtime**: 跨平台推理引擎
- **TensorRT**: NVIDIA推理加速引擎

---

## 📞 联系我们

- **Issue**: [GitHub Issues](https://github.com/cs405/visionhub/issues)
- **文档**: [完整文档](docs/)
- **示例**: [examples/](examples/)

---

<div align="center">

**⭐ 如果这个项目对您有帮助，请给我们一个Star！⭐**

Made with ❤️ by JKDCPPZzz

</div>

