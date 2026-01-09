# 数据增强与训练指南

## 数据增强

### 可用的增强方式

在 YAML 配置文件的 `RecPreProcess.transform_ops` 中添加：

```yaml
RecPreProcess:
  transform_ops:
  - ResizeImage:
      resize_short: 256
  - RandomHorizontalFlip:  # 水平翻转
      p: 0.5
  - ColorJitter:  # 颜色抖动
      brightness: 0.4
      contrast: 0.4
      saturation: 0.4
      hue: 0.1
      p: 0.8
  - RandomRotation:  # 旋转
      degrees: 15
      p: 0.3
  - RandomGaussianBlur:  # 模糊
      kernel_size: 5
      sigma: [0.1, 2.0]
      p: 0.2
  - CropImage:
      size: 224
  - NormalizeImage:
      scale: 1.0/255.0
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
      order: chw
  - ToTensor: null
```

### 增强类型说明

| 类名 | 参数 | 说明 |
|------|------|------|
| RandomHorizontalFlip | p=0.5 | 随机水平翻转 |
| RandomVerticalFlip | p=0.5 | 随机垂直翻转 |
| ColorJitter | brightness, contrast, saturation, hue | 颜色抖动 |
| RandomRotation | degrees=15, p=1.0 | 随机旋转 |
| RandomGaussianBlur | kernel_size=5, sigma=(0.1, 2.0) | 高斯模糊 |
| RandomGrayscale | p=0.1 | 随机转灰度 |
| RandomErasing | p=0.5, scale=(0.02, 0.33) | 随机擦除 |
| RandomResizedCrop | size=224, scale=(0.08, 1.0) | 随机缩放裁剪 |

---

## 训练流程

### 完整训练（带数据增强+评估+分析）

```powershell
python run_pipeline.py all -c visionhub/configs/shitu/rec_faiss_augment.yaml `
  --data_yaml dataset/data.yaml `
  --yolo_train_images dataset/train --yolo_train_labels dataset/train `
  --val_gallery_images dataset/val --val_gallery_labels dataset/val `
  --val_query_images dataset/test --val_query_labels dataset/test `
  --save_dir visionhub/output/training/my_exp `
  --epochs 50 --batch_size 16 --device cuda --amp `
  --use_pk --P 8 --K 4 `
  --w_triplet 1.0 --w_circle 0.2 `
  --teacher_torchvision --student_pretrained `
  --eval_gallery_images dataset/val --eval_gallery_labels dataset/val `
  --eval_query_images dataset/test --eval_query_labels dataset/test `
  --strict_image_split --exclude_same_image `
  --save_eval_dir visionhub/output/eval/my_exp
```

### 分析 TopK 结果

```powershell
python visionhub/tools/analyze_topk.py `
  --topk_json visionhub/output/eval/my_exp/topk.json `
  --data_yaml dataset/data.yaml `
  --save_dir visionhub/output/eval/my_exp/analysis
```

查看报告：
```powershell
type visionhub\output\eval\my_exp\analysis\analysis_report.txt
```

---

## 调参建议

### Loss 组合

| 配置 | 参数 | 说明 |
|------|------|------|
| 纯 Triplet | `--w_triplet 1.0 --w_circle 0.0` | 最稳定 |
| Triplet+Circle | `--w_triplet 1.0 --w_circle 0.2` | 推荐 |
| 强 Circle | `--w_triplet 0.5 --w_circle 0.5` | 激进 |

### 数据增强强度

根据数据集大小选择：
- **小数据集（< 1000）**：降低 p 值，减少增强种类
- **中数据集（1000-5000）**：使用默认配置（rec_faiss_augment.yaml）
- **大数据集（> 5000）**：增加增强种类和强度

---

## 常见问题

**Q: 训练时 loss=nan 怎么办？**  
A: 使用 warmup 和梯度裁剪：`--warmup_epochs 5 --grad_clip 1.0`

**Q: 如何提升召回率低的类别？**  
A: 查看 analysis_report.txt，针对性增加该类别的训练样本

**Q: 类别混淆严重怎么办？**  
A: 增大 margin 参数，或使用更强的 loss 组合

