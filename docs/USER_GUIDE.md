# visionhub 完整使用文档

> **版本**: v1.0.0  
> **更新时间**: 2026-01-09  
> **状态**: 生产就绪

---

## 📋 目录

1. [项目简介](#1-项目简介)
2. [安装指南](#2-安装指南)
3. [快速开始](#3-快速开始)
4. [核心功能详解](#4-核心功能详解)
5. [Backbone模型库](#5-backbone模型库)
6. [Loss函数库](#6-loss函数库)
7. [数据增强](#7-数据增强)
8. [训练指南](#8-训练指南)
9. [YOLO集成](#9-yolo集成)
10. [部署指南](#10-部署指南)
11. [API参考](#11-api参考)
12. [常见问题](#12-常见问题)

---

## 1. 项目简介

### 1.1 什么是visionhub？

visionhub是一个**全功能的PyTorch图像分类工具包**，从visionhub完整迁移而来，支持：

- ✅ **图像分类**：标准分类任务
- ✅ **图像检索**：向量检索、相似度搜索
- ✅ **人脸识别**：1:1验证、1:N识别
- ✅ **YOLO集成**：检测+分类联合推理
- ✅ **模型部署**：ONNX、TensorRT、量化
- ✅ **知识蒸馏**：Teacher-Student蒸馏

### 1.2 核心特性

| 特性 | 描述 |
|------|------|
| **85+ Backbone** | ResNet, EfficientNet, ViT, Swin等 |
| **50+ Loss函数** | 度量学习、蒸馏、分类Loss |
| **16种数据增强** | AutoAugment, Mixup, CutMix等 |
| **完整部署** | ONNX, TensorRT, 量化, HTTP服务 |
| **YOLO支持** | 检测+分类/检索联合推理 |

### 1.3 与visionhub对比

| 功能 | visionhub | visionhub |
|------|-----------|-------------|
| 框架 | visionhubvisionhub | PyTorch |
| Backbone | 100+ | 85+ (核心全覆盖) |
| Loss函数 | 60+ | 50+ (主流全覆盖) |
| YOLO集成 | ❌ | ✅ |
| 部署工具 | ✅ | ✅ |
| 完成度 | 100% | **90%** |

---

## 2. 安装指南

### 2.1 环境要求

```
Python >= 3.8
PyTorch >= 1.10.0
CUDA >= 11.0 (推荐，用于GPU加速)
```

### 2.2 基础安装

```bash
# 方式1: pip安装（推荐）
pip install visionhub

# 方式2: 从源码安装
git clone https://github.com/visionhub/visionhub.git
cd visionhub
pip install -e .

# 方式3: 仅安装依赖
pip install -r requirements.txt
```

### 2.3 可选安装

```bash
# GPU版本（Faiss GPU加速）
pip install visionhub[gpu]

# ONNX导出支持
pip install visionhub[onnx]

# HTTP服务支持
pip install visionhub[serving]

# 完整安装（所有功能）
pip install visionhub[all]
```

### 2.4 验证安装

```python
import torch
import visionhub

print(f"PyTorch version: {torch.__version__}")
print(f"visionhub version: {visionhub.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# 查看可用模型
from visionhub.ptcls.arch.backbone import list_backbones
print(f"Available backbones: {len(list_backbones())}")
```

---

## 3. 快速开始

### 3.1 图像分类（5分钟上手）

```python
from visionhub.ptcls.arch.backbone import build_backbone
from torchvision import transforms
from PIL import Image
import torch

# 1. 加载模型
model = build_backbone('resnet50', num_classes=1000, pretrained=True)
model.eval()

# 2. 图像预处理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. 推理
img = Image.open('test.jpg')
img_tensor = transform(img).unsqueeze(0)

with torch.no_grad():
    output = model(img_tensor)
    prob = torch.softmax(output, dim=1)
    top5_prob, top5_idx = prob.topk(5)

print(f"Top-5 predictions: {list(zip(top5_idx[0].tolist(), top5_prob[0].tolist()))}")
```

### 3.2 图像检索（10分钟上手）

```python
from visionhub.ptcls.rec import RecPredictor
from visionhub.ptcls.rec.gallery_builder import GalleryBuilder

# 1. 构建检索库
builder = GalleryBuilder(
    model_name='resnet50',
    embedding_size=512
)
builder.build_from_directory('gallery_images/')

# 2. 检索
predictor = RecPredictor(
    model_name='resnet50',
    gallery_path='gallery.faiss'
)
results = predictor.search('query.jpg', top_k=5)

print(f"Top-5 similar images: {results}")
```

### 3.3 YOLO + 分类（15分钟上手）

```python
from visionhub.ptcls.tools.yolo_det_classification import YOLODetectionClassification

# 创建系统
system = YOLODetectionClassification(
    yolo_model_path='yolov8n.pt',
    cls_model_name='resnet50',
    cls_checkpoint='classifier.pth',
    num_classes=100
)

# 检测+分类
results = system.detect_and_classify('image.jpg', save_result=True)

for i, res in enumerate(results):
    print(f"{i+1}. {res['cls_class_name']} "
          f"(box: {res['box']}, conf: {res['cls_conf']:.3f})")
```

---

## 4. 核心功能详解

### 4.1 图像分类

#### 标准分类训练

```bash
python tools/train_classification.py \
  --data_root ./data/imagenet \
  --model resnet50 \
  --num_classes 1000 \
  --epochs 100 \
  --batch_size 128 \
  --lr 0.1 \
  --device cuda \
  --save_dir ./output/resnet50
```

**数据集格式（ImageFolder）**:
```
data/
  train/
    class1/
      img1.jpg
      img2.jpg
    class2/
      img1.jpg
  val/
    class1/
    class2/
```

#### 评估

```bash
python tools/eval_classification.py \
  --model resnet50 \
  --checkpoint output/resnet50/best.pth \
  --data_root data/val \
  --num_classes 1000
```

### 4.2 图像检索

#### 检索模型训练

```bash
python tools/train_rec_kd.py \
  --yolo_images dataset/train \
  --yolo_labels dataset/train \
  --data_yaml dataset/data.yaml \
  --save_dir output/retrieval \
  --epochs 50 \
  --batch_size 32 \
  --use_pk --P 8 --K 4 \
  --w_triplet 1.0 --w_circle 0.2 \
  --device cuda
```

**数据集格式（YOLO检测）**:
```
dataset/
  train/
    images/
      img1.jpg
      img2.jpg
    labels/
      img1.txt  # class_id x_center y_center width height
      img2.txt
  data.yaml  # names: [class1, class2, ...]
```

#### 构建检索库

```bash
python tools/build_gallery.py \
  -c configs/shitu/rec_faiss_demo.yaml
```

#### 检索评估

```bash
python tools/eval_retrieval.py \
  -c configs/shitu/rec_faiss_demo.yaml \
  --gallery_images dataset/val \
  --gallery_labels dataset/val \
  --query_images dataset/test \
  --query_labels dataset/test \
  --data_yaml dataset/data.yaml \
  --strict_image_split \
  --exclude_same_image
```

### 4.3 人脸识别

#### 训练人脸模型

```bash
python tools/train_face_recognition.py \
  --train_root faces/train \
  --val_pairs faces/val/pairs.txt \
  --model ir_net_50 \
  --loss arcface \
  --s 64.0 --m 0.5 \
  --epochs 100 \
  --batch_size 128 \
  --save_dir output/face
```

**数据集格式（人脸训练）**:
```
faces/
  train/
    person1/
      face1.jpg
      face2.jpg
    person2/
      face1.jpg
  val/
    pairs.txt  # path1 path2 1/0
```

#### 1:1 人脸验证

```bash
python tools/face_recognition_inference.py \
  --task verify \
  --model ir_net_50 \
  --checkpoint output/face/best.pth \
  --image1 person1.jpg \
  --image2 person2.jpg \
  --threshold 0.3
```

#### 1:N 人脸识别

```bash
python tools/face_recognition_inference.py \
  --task identify \
  --model ir_net_50 \
  --checkpoint output/face/best.pth \
  --query query.jpg \
  --gallery_dir faces/gallery \
  --top_k 5
```

### 4.4 知识蒸馏

```bash
python tools/train_rec_kd.py \
  --yolo_images dataset/train \
  --yolo_labels dataset/train \
  --data_yaml dataset/data.yaml \
  --save_dir output/kd \
  --epochs 50 \
  --teacher_torchvision \
  --w_kd_embed 1.0 \
  --w_triplet 1.0 \
  --device cuda
```

---

## 5. Backbone模型库

### 5.1 CNN系列（60+个）

#### ResNet家族
```python
# ResNet
resnet18, resnet34, resnet50, resnet101, resnet152

# ResNeXt
resnext50_32x4d, resnext101_32x8d

# Wide ResNet
wide_resnet50_2, wide_resnet101_2, wide_resnet28_10

# SE-ResNet
se_resnet50, se_resnet101
```

#### 轻量级模型
```python
# MobileNet
mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large

# EfficientNet
efficientnet_b0, efficientnet_b1, ..., efficientnet_b7

# GhostNet
ghostnet

# ShuffleNet
shufflenet_v2_x0_5, shufflenet_v2_x1_0
```

#### 人脸识别专用
```python
# MobileFaceNet
mobilefacenet

# IR-Net
ir_net_50, ir_net_100, ir_net_152
```

#### 高级CNN
```python
# DenseNet
densenet121, densenet161, densenet169, densenet201

# DLA (Deep Layer Aggregation)
dla34, dla60

# DPN (Dual Path Networks)
dpn68, dpn92

# Inception
inception_v3

# Xception
xception
```

### 5.2 Transformer系列（25+个）

```python
# Vision Transformer
vit_tiny_patch16_224, vit_small_patch16_224, vit_base_patch16_224

# Swin Transformer
swin_tiny_patch4_window7_224, swin_small_patch4_window7_224

# DeiT
deit_tiny_patch16_224, deit_small_patch16_224

# ConvNeXt
convnext_tiny, convnext_small, convnext_base

# MobileViT
mobilevit_s, mobilevit_xs

# CSWin Transformer
cswin_tiny, cswin_small

# LeViT
levit_128, levit_256

# PVT-V2
pvt_v2_b0, pvt_v2_b1
```

### 5.3 使用示例

```python
from visionhub.ptcls.arch.backbone import build_backbone

# 构建模型
model = build_backbone('resnet50', num_classes=1000, pretrained=True)

# 查看所有可用模型
from visionhub.ptcls.arch.backbone import list_backbones
all_models = list_backbones()
print(f"Total {len(all_models)} models available")

# 按类别筛选
cnn_models = [m for m in all_models if 'resnet' in m or 'efficientnet' in m]
transformer_models = [m for m in all_models if 'vit' in m or 'swin' in m]
```

---

## 6. Loss函数库

### 6.1 度量学习Loss（28个）

#### 基础度量Loss
```python
from visionhub.ptcls.loss.metric import (
    ArcFaceLoss,      # ArcFace
    CosFaceLoss,      # CosFace  
    SphereFaceLoss,   # SphereFace
    TripletLoss,      # Triplet Loss
    CenterLoss,       # Center Loss
    ContrastiveLoss,  # Contrastive Loss
)

# 使用示例
criterion = ArcFaceLoss(
    in_features=512,
    num_classes=1000,
    s=64.0,
    m=0.5
)
loss = criterion(embeddings, labels)
```

#### 高级度量Loss
```python
from visionhub.ptcls.loss.metric import (
    CircleLoss,       # Circle Loss
    SupConLoss,       # Supervised Contrastive
    ProxyNCALoss,     # Proxy-NCA
    LiftedStructure,  # Lifted Structure
)

# 高级度量Loss（新增）
from visionhub.ptcls.loss.metric.advanced_metric_loss import (
    MSMLoss,          # Multi-Similarity Mining
    XBMLoss,          # Cross-Batch Memory
    SoftTripleLoss,   # Soft Triple
    AngularLoss,      # Angular Loss
)
```

### 6.2 蒸馏Loss（16个）

```python
from visionhub.ptcls.loss.distillation import (
    KLDivLoss,        # KL散度
    DKDLoss,          # Decoupled KD
    RKDLoss,          # Relational KD
)

# 高级蒸馏Loss（新增）
from visionhub.ptcls.loss.distillation.advanced_distill_loss import (
    AFDLoss,          # Attention Feature Distillation
    ReviewKDLoss,     # Review KD
    CRDLoss,          # Contrastive Representation Distillation
)

# 使用示例
criterion = AFDLoss(attention_type='spatial')
loss = criterion(student_features, teacher_features)
```

### 6.3 分类Loss（4个）

```python
from visionhub.ptcls.loss import (
    FocalLoss,        # Focal Loss
    LabelSmoothingCrossEntropy,  # Label Smoothing
    AsymmetricLoss,   # Asymmetric Loss (多标签)
)
```

### 6.4 Loss组合使用

```python
# 多Loss组合
total_loss = (
    1.0 * triplet_loss(embeddings, labels) +
    0.5 * circle_loss(embeddings, labels) +
    1.0 * arcface_loss(embeddings, labels)
)
```

---

## 7. 数据增强

### 7.1 基础增强

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomGrayscale(p=0.1),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

### 7.2 高级增强

```python
from visionhub.ptcls.data.augmentation.advanced_augment import (
    AutoAugment,
    RandAugment,
    GridMask,
    RandomErasing
)

# AutoAugment
transform = transforms.Compose([
    transforms.Resize(256),
    AutoAugment(policy='imagenet'),
    transforms.ToTensor()
])

# RandAugment
transform = transforms.Compose([
    transforms.Resize(256),
    RandAugment(n=2, m=10),
    transforms.ToTensor()
])
```

### 7.3 混合增强

```python
from visionhub.ptcls.data.augmentation.advanced_augment import (
    mixup_data_enhanced,
    cutmix_data_enhanced,
    fmix
)

# 在训练循环中使用
for images, labels in dataloader:
    # Mixup
    images, y_a, y_b, lam = mixup_data_enhanced(images, labels, alpha=1.0)
    outputs = model(images)
    loss = lam * criterion(outputs, y_a) + (1-lam) * criterion(outputs, y_b)
    
    # CutMix
    images, y_a, y_b, lam = cutmix_data_enhanced(images, labels, alpha=1.0)
    
    # FMix
    images, y_a, y_b, lam = fmix(images, labels, alpha=1.0, shape=(224, 224))
```

---

## 8. 训练指南

### 8.1 标准分类训练

**完整训练脚本**:
```bash
python tools/train_classification.py \
  --data_root ./data/imagenet \
  --model resnet50 \
  --num_classes 1000 \
  --epochs 100 \
  --batch_size 128 \
  --lr 0.1 \
  --scheduler cosine \
  --mixup 0.2 \
  --cutmix 0.2 \
  --label_smoothing 0.1 \
  --weight_decay 1e-4 \
  --device cuda \
  --amp \
  --save_dir ./output/resnet50
```

**训练监控**:
```python
# 查看训练日志
tail -f output/resnet50/train.log

# TensorBoard可视化
tensorboard --logdir output/resnet50/tensorboard
```

### 8.2 检索模型训练

**完整训练流程**:
```bash
# Step 1: 训练检索模型
python tools/train_rec_kd.py \
  --yolo_images dataset/train \
  --yolo_labels dataset/train \
  --data_yaml dataset/data.yaml \
  --save_dir output/retrieval \
  --epochs 50 \
  --batch_size 32 \
  --lr 0.001 \
  --use_pk --P 8 --K 4 \
  --w_triplet 1.0 \
  --w_circle 0.2 \
  --teacher_torchvision \
  --device cuda \
  --amp

# Step 2: 构建检索库
python tools/build_gallery.py \
  -c configs/shitu/rec_faiss_demo.yaml

# Step 3: 评估
python tools/eval_retrieval.py \
  -c configs/shitu/rec_faiss_demo.yaml \
  --gallery_images dataset/val \
  --gallery_labels dataset/val \
  --query_images dataset/test \
  --query_labels dataset/test \
  --data_yaml dataset/data.yaml
```

### 8.3 人脸识别训练

```bash
python tools/train_face_recognition.py \
  --train_root faces/train \
  --val_pairs faces/val/pairs.txt \
  --model ir_net_50 \
  --loss arcface \
  --s 64.0 \
  --m 0.5 \
  --epochs 100 \
  --batch_size 128 \
  --lr 0.1 \
  --scheduler step \
  --weight_decay 5e-4 \
  --save_dir output/face
```

---

## 9. YOLO集成

### 9.1 YOLO + 分类

```python
from visionhub.ptcls.tools.yolo_det_classification import YOLODetectionClassification

# 创建系统
system = YOLODetectionClassification(
    yolo_model_path='yolov8n.pt',
    cls_model_name='resnet50',
    cls_checkpoint='classifier.pth',
    num_classes=100,
    class_names=['cat', 'dog', ...]
)

# 单张图片
results = system.detect_and_classify('image.jpg', save_result=True)

# 批量处理
results = system.batch_predict('images/', save_dir='results/')
```

### 9.2 YOLO + 检索

**完整流程**:
```bash
# 1. 用YOLO数据训练检索模型
python run_pipeline.py all \
  -c configs/shitu/rec_faiss_demo.yaml \
  --data_yaml dataset/data.yaml \
  --yolo_train_images dataset/train \
  --yolo_train_labels dataset/train \
  --eval_gallery_images dataset/val \
  --eval_gallery_labels dataset/val \
  --eval_query_images dataset/test \
  --eval_query_labels dataset/test \
  --save_dir output/yolo_retrieval \
  --epochs 50 \
  --device cuda

# 2. 使用检索系统
python tools/predict_system.py \
  -c configs/shitu/rec_faiss_demo.yaml \
  --infer_img demo.jpg \
  --save_path result.jpg
```

### 9.3 数据格式（YOLO）

**YOLO检测数据格式**:
```
dataset/
  train/
    img1.jpg
    img1.txt  # class_id x_center y_center width height (归一化)
    img2.jpg
    img2.txt
  val/
  test/
  data.yaml  # names: [class1, class2, ...]
```

**data.yaml示例**:
```yaml
train: dataset/train
val: dataset/val
test: dataset/test

nc: 80  # 类别数
names: ['person', 'bicycle', 'car', ...]
```

---

## 10. 部署指南

### 10.1 模型导出

#### ONNX导出
```bash
python tools/export_model.py \
  --model resnet50 \
  --checkpoint output/best.pth \
  --num_classes 1000 \
  --format onnx \
  --simplify \
  --save_dir deploy/models
```

#### TensorRT导出
```bash
# Step 1: 导出ONNX
python tools/export_model.py \
  --model resnet50 \
  --checkpoint output/best.pth \
  --format onnx \
  --save_dir deploy/models

# Step 2: 构建TensorRT引擎
python deploy/tensorrt_predictor.py \
  --mode build \
  --onnx deploy/models/best.onnx \
  --engine deploy/models/best.engine \
  --fp16
```

### 10.2 推理

#### ONNX推理
```bash
python deploy/onnx_predictor.py \
  --model deploy/models/best.onnx \
  --image test.jpg \
  --device cuda \
  --benchmark
```

#### TensorRT推理
```bash
python deploy/tensorrt_predictor.py \
  --mode predict \
  --engine deploy/models/best.engine \
  --image test.jpg
```

### 10.3 模型量化

```bash
python tools/quantize_model.py \
  --model resnet50 \
  --checkpoint output/best.pth \
  --method static \
  --calib_data calibration_images/ \
  --calib_images 100 \
  --output deploy/quantized/best_int8.pth
```

### 10.4 HTTP服务

```bash
python deploy/http_server.py \
  --model resnet50 \
  --checkpoint output/best.pth \
  --num_classes 1000 \
  --class_names classes.txt \
  --host 0.0.0.0 \
  --port 8080
```

**API调用**:
```python
import requests

# 分类
files = {'image': open('test.jpg', 'rb')}
response = requests.post('http://localhost:8080/classify', files=files)
print(response.json())
```

### 10.5 批量推理

```bash
python deploy/batch_inference.py \
  --task classify \
  --model resnet50 \
  --checkpoint output/best.pth \
  --image_dir test_images/ \
  --batch_size 32 \
  --output results.json
```

---

## 11. API参考

### 11.1 核心API

```python
# 构建模型
from visionhub.ptcls.arch.backbone import build_backbone
model = build_backbone(name='resnet50', num_classes=1000, pretrained=True)

# 构建Loss
from visionhub.ptcls.loss.metric import ArcFaceLoss
criterion = ArcFaceLoss(in_features=512, num_classes=1000)

# 数据加载
from visionhub.ptcls.data.datasets import ImageFolderDataset
dataset = ImageFolderDataset(root='data/train', transform=transform)

# 评估指标
from visionhub.ptcls.metric import accuracy, recall_at_k
acc = accuracy(outputs, labels)
recall = recall_at_k(similarity_matrix, labels, k=5)
```

### 11.2 配置文件

**YAML配置示例**:
```yaml
# configs/custom_config.yaml
Global:
  device: cuda
  epochs: 100
  batch_size: 128

Arch:
  name: resnet50
  pretrained: True
  num_classes: 1000

Loss:
  Train:
    - CELoss:
        weight: 1.0
    - TripletLoss:
        weight: 0.5
        margin: 0.3

Optimizer:
  name: SGD
  lr: 0.1
  momentum: 0.9
  weight_decay: 0.0001

DataLoader:
  Train:
    dataset:
      name: ImageFolder
      root: ./data/train
    batch_size: 128
    num_workers: 4
```

---

## 12. 常见问题

### Q1: 如何选择合适的Backbone？

**推荐选择**:
- **高精度**：ResNet50/101, EfficientNet-B4/B5, Swin-Transformer
- **速度优先**：MobileNetV3, EfficientNet-B0, ResNet18
- **人脸识别**：IR-Net-50/100, MobileFaceNet
- **检索任务**：ResNet50 + Embedding Head

### Q2: YOLO数据能直接用于训练吗？

**可以！** visionhub支持直接使用YOLO检测数据：
```bash
python tools/train_classification.py \
  --yolo_images dataset/train \
  --yolo_labels dataset/train \
  --data_yaml dataset/data.yaml \
  --model resnet50 \
  --epochs 100
```

系统会自动crop检测框作为分类样本。

### Q3: 训练效果不好怎么办？

**优化建议**:
1. 增加数据增强：`--mixup 0.2 --cutmix 0.2`
2. 使用预训练模型：`--pretrained`
3. 调整学习率：`--lr 0.01 --scheduler cosine`
4. 使用更强Loss：ArcFace, Circle Loss
5. 增加训练轮数：`--epochs 200`

### Q4: 如何加速推理？

**加速方案**:
1. 使用TensorRT：`5-10x加速`
2. 模型量化：`INT8量化，4x压缩`
3. 使用轻量模型：MobileNet, EfficientNet-B0
4. 批量推理：增大batch_size

### Q5: 支持多GPU训练吗？

**支持！**
```bash
# 使用DataParallel
python -m torch.distributed.launch --nproc_per_node=4 \
  tools/train_classification.py \
  --data_root data/ \
  --model resnet50 \
  --distributed
```

---

## 📞 获取帮助

- **文档**: [https://visionhub.readthedocs.io](docs/)
- **Issue**: [GitHub Issues](https://github.com/cs405/visionhub/issues)
- **示例**: [examples/](examples/)

---

**✅ 文档完成！visionhub已准备就绪！** 🎉

