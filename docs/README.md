# visionhub 文档中心

欢迎来到visionhub文档中心！这里包含了所有的使用文档、API参考和迁移报告。

---

## 📚 核心文档

### 🚀 [完整使用指南 (USER_GUIDE.md)](USER_GUIDE.md)
**最重要的文档！** 包含：
- 安装指南
- 快速开始（5/10/15分钟教程）
- 所有功能详解（分类/检索/人脸识别/YOLO）
- Backbone模型库完整列表
- Loss函数库完整列表
- 数据增强详解
- 训练指南
- 部署指南
- API参考
- 常见问题

👉 **新手必读，功能全覆盖！**

---

## 📊 迁移报告

### [项目状态 (PROJECT_STATUS.md)](PROJECT_STATUS.md)
- 整体移植进度：**90%**
- 各模块完成度统计
- 已完成功能清单
- 待完成功能列表

### [Backbone迁移报告 (BACKBONE_MIGRATION_REPORT.md)](BACKBONE_MIGRATION_REPORT.md)
- **85+个Backbone模型**详细说明
- CNN系列（60+）：ResNet, EfficientNet, MobileFaceNet, IR-Net等
- Transformer系列（25+）：ViT, Swin, CSWin, LeViT等
- 性能对比和使用示例

### [数据增强&Loss报告 (AUGMENTATION_LOSS_REPORT.md)](AUGMENTATION_LOSS_REPORT.md)
- **16种数据增强**详解（100%完成）
- **50+个Loss函数**详解（85%完成）
- 度量学习Loss（28个）
- 蒸馏Loss（16个）
- 使用示例

---

## 🎯 专项指南

### [部署指南 (DEPLOYMENT_GUIDE.md)](DEPLOYMENT_GUIDE.md)
- 模型导出（ONNX, TensorRT, TorchScript等）
- 推理引擎（ONNX Runtime, TensorRT）
- 模型量化（INT8, 4x压缩）
- HTTP服务部署
- 批量推理
- 性能优化

### [分类指南 (CLASSIFICATION_GUIDE.md)](CLASSIFICATION_GUIDE.md)
- 标准分类训练
- YOLO-CLS集成
- YOLO检测转分类
- 数据格式说明
- 最佳实践

---

## 🗂️ 文档导航

### 按功能查找

| 功能 | 主要文档 | 章节 |
|------|---------|------|
| **图像分类** | [USER_GUIDE](USER_GUIDE.md#41-图像分类) | 4.1 |
| **图像检索** | [USER_GUIDE](USER_GUIDE.md#42-图像检索) | 4.2 |
| **人脸识别** | [USER_GUIDE](USER_GUIDE.md#43-人脸识别) | 4.3 |
| **YOLO集成** | [USER_GUIDE](USER_GUIDE.md#9-yolo集成) | 9 |
| **知识蒸馏** | [USER_GUIDE](USER_GUIDE.md#44-知识蒸馏) | 4.4 |
| **模型部署** | [DEPLOYMENT_GUIDE](DEPLOYMENT_GUIDE.md) | 全文 |
| **数据增强** | [AUGMENTATION_LOSS_REPORT](AUGMENTATION_LOSS_REPORT.md) | 完整 |

### 按角色查找

**🔰 新手入门**:
1. [README](../README.md) - 项目概览
2. [USER_GUIDE - 快速开始](USER_GUIDE.md#3-快速开始)
3. [USER_GUIDE - 图像分类](USER_GUIDE.md#41-图像分类)

**🎓 进阶用户**:
1. [USER_GUIDE - 图像检索](USER_GUIDE.md#42-图像检索)
2. [USER_GUIDE - YOLO集成](USER_GUIDE.md#9-yolo集成)
3. [AUGMENTATION_LOSS_REPORT](AUGMENTATION_LOSS_REPORT.md)

**🚀 生产部署**:
1. [DEPLOYMENT_GUIDE](DEPLOYMENT_GUIDE.md)
2. [USER_GUIDE - 部署指南](USER_GUIDE.md#10-部署指南)
3. [USER_GUIDE - API参考](USER_GUIDE.md#11-api参考)

**👨‍💻 开发者**:
1. [PROJECT_STATUS](PROJECT_STATUS.md)
2. [BACKBONE_MIGRATION_REPORT](BACKBONE_MIGRATION_REPORT.md)
3. [MIGRATION_STATUS](MIGRATION_STATUS.md)

---

## 📖 快速索引

### 常见任务

| 任务 | 查看章节 |
|------|---------|
| 安装visionhub | [USER_GUIDE - 安装](USER_GUIDE.md#2-安装指南) |
| 训练分类模型 | [USER_GUIDE - 分类训练](USER_GUIDE.md#81-标准分类训练) |
| 训练检索模型 | [USER_GUIDE - 检索训练](USER_GUIDE.md#82-检索模型训练) |
| 训练人脸模型 | [USER_GUIDE - 人脸训练](USER_GUIDE.md#83-人脸识别训练) |
| YOLO数据训练 | [USER_GUIDE - YOLO集成](USER_GUIDE.md#9-yolo集成) |
| 导出ONNX | [DEPLOYMENT_GUIDE - ONNX](DEPLOYMENT_GUIDE.md#1-模型导出) |
| TensorRT加速 | [DEPLOYMENT_GUIDE - TensorRT](DEPLOYMENT_GUIDE.md#build-engine-from-onnx) |
| 模型量化 | [DEPLOYMENT_GUIDE - 量化](DEPLOYMENT_GUIDE.md#3-模型量化) |
| 选择Backbone | [BACKBONE_MIGRATION_REPORT](BACKBONE_MIGRATION_REPORT.md) |
| 选择Loss | [AUGMENTATION_LOSS_REPORT](AUGMENTATION_LOSS_REPORT.md) |

### 数据格式

| 格式 | 查看章节 |
|------|---------|
| ImageFolder格式 | [USER_GUIDE - 标准分类](USER_GUIDE.md#标准分类训练) |
| YOLO检测格式 | [USER_GUIDE - YOLO数据](USER_GUIDE.md#93-数据格式yolo) |
| 人脸识别格式 | [USER_GUIDE - 人脸数据](USER_GUIDE.md#训练人脸模型) |
| 检索数据格式 | [USER_GUIDE - 检索训练](USER_GUIDE.md#检索模型训练) |

---

## 🔍 文档搜索建议

### 如何找到想要的内容？

1. **先看目录**：每个文档开头都有详细目录
2. **使用搜索**：Ctrl+F 搜索关键词
3. **查看索引**：本文件提供快速索引
4. **看示例代码**：每个功能都有完整示例

### 常用搜索关键词

```
# 安装相关
"安装", "install", "requirements", "环境"

# 训练相关
"train", "训练", "数据集", "dataset", "格式"

# 模型相关
"backbone", "模型", "resnet", "efficientnet"

# Loss相关
"loss", "arcface", "triplet", "蒸馏"

# 部署相关
"onnx", "tensorrt", "量化", "部署", "推理"

# YOLO相关
"yolo", "检测", "data.yaml"
```

---

## ⚡ 快速链接

### 必读文档
- 📘 [完整使用指南](USER_GUIDE.md) ⭐⭐⭐⭐⭐
- 📗 [项目状态](PROJECT_STATUS.md)
- 📙 [部署指南](DEPLOYMENT_GUIDE.md)

### 参考文档
- 📕 [Backbone报告](BACKBONE_MIGRATION_REPORT.md)
- 📔 [数据增强&Loss](AUGMENTATION_LOSS_REPORT.md)
- 📓 [分类指南](CLASSIFICATION_GUIDE.md)

---

## 📞 获取帮助

如果文档没有解决您的问题：

1. **查看常见问题**: [USER_GUIDE - FAQ](USER_GUIDE.md#12-常见问题)
2. **提交Issue**: [GitHub Issues](https://github.com/cs405/visionhub/issues)
3. **查看示例**: [examples/](../examples/)

---

## 📝 文档维护

- **最后更新**: 2026-01-09
- **文档版本**: v1.0.0
- **对应代码版本**: visionhub v1.0.0

---

**✅ 开始使用 → [完整使用指南 (USER_GUIDE.md)](USER_GUIDE.md)** 🎉

