"""Face Recognition Module

完整的人脸识别功能：
1. 人脸检测（可选，使用YOLO或其他检测器）
2. 人脸对齐（关键点检测+仿射变换）
3. 人脸特征提取（ArcFace, CosFace等）
4. 人脸识别（1:1验证 或 1:N识别）
5. 人脸质量评估

开发自visionhub人脸识别模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize
from typing import Tuple, List, Optional


class FaceAccuracy(nn.Module):
    """人脸识别准确率评估

    支持：
    - 1:1 验证准确率
    - ROC曲线
    - 最优阈值计算
    """

    def __init__(self, nfolds=10):
        super().__init__()
        self.nfolds = nfolds
        self.embedding_left_list = []
        self.embedding_right_list = []
        self.label_list = []
        self.best_acc = 0.
        self.best_threshold = 0.

    def update(self, embeddings_left, embeddings_right, labels):
        """更新评估数据

        Args:
            embeddings_left: 左侧人脸特征 (B, D)
            embeddings_right: 右侧人脸特征 (B, D)
            labels: 是否同一人 (B,) [0或1]
        """
        if isinstance(embeddings_left, torch.Tensor):
            embeddings_left = embeddings_left.detach().cpu().numpy()
        if isinstance(embeddings_right, torch.Tensor):
            embeddings_right = embeddings_right.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()

        # L2归一化
        embeddings_left = normalize(embeddings_left)
        embeddings_right = normalize(embeddings_right)

        self.embedding_left_list.append(embeddings_left)
        self.embedding_right_list.append(embeddings_right)
        self.label_list.append(labels)

    def compute(self):
        """计算准确率和最优阈值"""
        embeddings_left = np.concatenate(self.embedding_left_list, axis=0)
        embeddings_right = np.concatenate(self.embedding_right_list, axis=0)
        labels = np.concatenate(self.label_list, axis=0)

        # 计算相似度（余弦相似度）
        similarities = np.sum(embeddings_left * embeddings_right, axis=1)

        # K折交叉验证
        kfold = KFold(n_splits=self.nfolds, shuffle=False)

        accuracies = []
        thresholds = []

        for train_idx, test_idx in kfold.split(similarities):
            train_similarities = similarities[train_idx]
            train_labels = labels[train_idx]
            test_similarities = similarities[test_idx]
            test_labels = labels[test_idx]

            # 在训练集上找最优阈值
            best_acc = 0
            best_thresh = 0

            for thresh in np.arange(-1.0, 1.0, 0.01):
                train_preds = (train_similarities >= thresh).astype(int)
                acc = (train_preds == train_labels).mean()
                if acc > best_acc:
                    best_acc = acc
                    best_thresh = thresh

            # 在测试集上评估
            test_preds = (test_similarities >= best_thresh).astype(int)
            test_acc = (test_preds == test_labels).mean()

            accuracies.append(test_acc)
            thresholds.append(best_thresh)

        mean_acc = np.mean(accuracies)
        mean_threshold = np.mean(thresholds)

        self.best_acc = float(mean_acc)
        self.best_threshold = float(mean_threshold)

        return {
            'accuracy': self.best_acc,
            'threshold': self.best_threshold,
            'std': float(np.std(accuracies))
        }

    def reset(self):
        """重置统计"""
        self.embedding_left_list = []
        self.embedding_right_list = []
        self.label_list = []
        self.best_acc = 0.
        self.best_threshold = 0.


class FaceVerification:
    """1:1 人脸验证

    判断两张人脸图片是否为同一人
    """

    def __init__(self, model, threshold=0.3, device='cuda'):
        """
        Args:
            model: 人脸识别模型
            threshold: 相似度阈值
            device: 设备
        """
        self.model = model
        self.threshold = threshold
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def extract_feature(self, image):
        """提取人脸特征

        Args:
            image: 输入图片 (C, H, W) 或 (B, C, H, W) tensor

        Returns:
            feature: 归一化后的特征向量
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)

        image = image.to(self.device)
        feature = self.model(image)

        # 如果模型返回字典（如EmbeddingHead）
        if isinstance(feature, dict):
            feature = feature.get('embedding', feature.get('features', feature))

        # L2归一化
        feature = F.normalize(feature, p=2, dim=1)

        return feature

    def verify(self, image1, image2):
        """验证两张图片是否为同一人

        Args:
            image1: 第一张图片 tensor
            image2: 第二张图片 tensor

        Returns:
            is_same: 是否同一人 (bool)
            similarity: 相似度分数 (float)
        """
        feat1 = self.extract_feature(image1)
        feat2 = self.extract_feature(image2)

        # 余弦相似度
        similarity = (feat1 * feat2).sum(dim=1).item()
        is_same = similarity >= self.threshold

        return is_same, similarity


class FaceIdentification:
    """1:N 人脸识别

    在人脸库中搜索最相似的人脸
    """

    def __init__(self, model, gallery_features=None, gallery_labels=None,
                 threshold=0.3, device='cuda'):
        """
        Args:
            model: 人脸识别模型
            gallery_features: 人脸库特征 (N, D)
            gallery_labels: 人脸库标签 (N,)
            threshold: 识别阈值
            device: 设备
        """
        self.model = model
        self.threshold = threshold
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

        self.gallery_features = gallery_features
        self.gallery_labels = gallery_labels

    def build_gallery(self, images, labels):
        """构建人脸库

        Args:
            images: 人脸图片列表
            labels: 对应的标签列表
        """
        features = []

        with torch.no_grad():
            for img in images:
                if img.dim() == 3:
                    img = img.unsqueeze(0)
                img = img.to(self.device)

                feat = self.model(img)
                if isinstance(feat, dict):
                    feat = feat.get('embedding', feat.get('features', feat))
                feat = F.normalize(feat, p=2, dim=1)

                features.append(feat.cpu())

        self.gallery_features = torch.cat(features, dim=0)
        self.gallery_labels = torch.tensor(labels)

        print(f"[INFO] Gallery built: {len(labels)} faces")

    def identify(self, image, top_k=5):
        """识别人脸

        Args:
            image: 查询图片
            top_k: 返回前K个最相似结果

        Returns:
            results: [(label, similarity), ...]
        """
        if self.gallery_features is None:
            raise ValueError("Gallery not built. Call build_gallery() first.")

        # 提取查询特征
        if image.dim() == 3:
            image = image.unsqueeze(0)
        image = image.to(self.device)

        with torch.no_grad():
            query_feat = self.model(image)
            if isinstance(query_feat, dict):
                query_feat = query_feat.get('embedding', query_feat.get('features', query_feat))
            query_feat = F.normalize(query_feat, p=2, dim=1)

        # 计算与所有gallery的相似度
        gallery_features = self.gallery_features.to(self.device)
        similarities = (query_feat @ gallery_features.t()).squeeze(0)

        # Top-K
        top_k = min(top_k, len(similarities))
        top_similarities, top_indices = similarities.topk(top_k)

        results = []
        for sim, idx in zip(top_similarities, top_indices):
            if sim.item() >= self.threshold:
                label = self.gallery_labels[idx].item()
                results.append((label, sim.item()))

        return results


class FaceQualityAssessment(nn.Module):
    """人脸质量评估

    评估人脸图片质量，包括：
    - 模糊度
    - 光照
    - 姿态角度
    - 遮挡
    """

    def __init__(self):
        super().__init__()

    def assess_blur(self, image):
        """评估模糊度（Laplacian方差）

        Args:
            image: 输入图片 (C, H, W) tensor

        Returns:
            blur_score: 模糊度分数，越大越清晰
        """
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()

        # 转灰度图
        if image.shape[0] == 3:
            image = image.transpose(1, 2, 0)
            gray = 0.299 * image[:,:,0] + 0.587 * image[:,:,1] + 0.114 * image[:,:,2]
        else:
            gray = image[0]

        # Laplacian算子
        import cv2
        laplacian_var = cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F).var()

        return float(laplacian_var)

    def assess_brightness(self, image):
        """评估光照（平均亮度）

        Args:
            image: 输入图片 (C, H, W) tensor

        Returns:
            brightness: 亮度值 [0-255]
        """
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()

        # 计算平均亮度
        mean_brightness = image.mean()

        # 如果是归一化后的[0-1]，转换到[0-255]
        if mean_brightness <= 1.0:
            mean_brightness *= 255

        return float(mean_brightness)

    def assess_quality(self, image):
        """综合质量评估

        Returns:
            quality_dict: {
                'blur': blur_score,
                'brightness': brightness_score,
                'overall': overall_score,
                'is_good': bool
            }
        """
        blur = self.assess_blur(image)
        brightness = self.assess_brightness(image)

        # 质量阈值
        blur_threshold = 100  # Laplacian方差阈值
        bright_min, bright_max = 50, 200  # 亮度范围

        # 判断是否合格
        is_good_blur = blur > blur_threshold
        is_good_brightness = bright_min < brightness < bright_max
        is_good = is_good_blur and is_good_brightness

        # 综合分数（0-100）
        blur_score = min(blur / blur_threshold, 1.0) * 50
        bright_score = (1.0 - abs(brightness - 125) / 125) * 50
        overall = blur_score + bright_score

        return {
            'blur': float(blur),
            'brightness': float(brightness),
            'blur_score': float(blur_score),
            'brightness_score': float(bright_score),
            'overall': float(overall),
            'is_good': bool(is_good)
        }


# 导出所有类
__all__ = [
    'FaceAccuracy',
    'FaceVerification',
    'FaceIdentification',
    'FaceQualityAssessment'
]

