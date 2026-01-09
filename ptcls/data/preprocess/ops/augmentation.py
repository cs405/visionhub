"""数据增强模块 - 用于检索训练的图像增强

支持的增强方式：
- RandomHorizontalFlip: 随机水平翻转
- RandomVerticalFlip: 随机垂直翻转
- ColorJitter: 颜色抖动（亮度、对比度、饱和度、色调）
- RandomRotation: 随机旋转
- RandomGrayscale: 随机转灰度图
- RandomGaussianBlur: 随机高斯模糊
- RandomErasing: 随机擦除（Cutout）

使用方式：
在配置文件中添加到 RecPreProcess.transform_ops 中：
```yaml
RecPreProcess:
  transform_ops:
    - ResizeImage:
        resize_short: 256
    - RandomHorizontalFlip:
        p: 0.5
    - ColorJitter:
        brightness: 0.4
        contrast: 0.4
        saturation: 0.4
        hue: 0.1
    - CropImage:
        size: 224
    - NormalizeImage:
        scale: 1.0/255.0
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
        order: chw
    - ToTensor: null
```
"""

import cv2
import numpy as np
import random
from typing import Optional, Tuple


class RandomHorizontalFlip:
    """随机水平翻转

    Args:
        p (float): 翻转概率，默认 0.5
    """
    def __init__(self, p=0.5):
        assert 0 <= p <= 1.0, f"概率必须在 [0, 1] 之间，得到 {p}"
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return cv2.flip(img, 1)  # 1 表示水平翻转
        return img


class RandomVerticalFlip:
    """随机垂直翻转

    Args:
        p (float): 翻转概率，默认 0.5
    """
    def __init__(self, p=0.5):
        assert 0 <= p <= 1.0, f"概率必须在 [0, 1] 之间，得到 {p}"
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return cv2.flip(img, 0)  # 0 表示垂直翻转
        return img


class ColorJitter:
    """颜色抖动 - 随机改变图像的亮度、对比度、饱和度、色调

    Args:
        brightness (float): 亮度变化范围 [max(0, 1-brightness), 1+brightness]
        contrast (float): 对比度变化范围
        saturation (float): 饱和度变化范围
        hue (float): 色调变化范围（注意：hue 的范围通常较小，如 0.1）
        p (float): 应用概率，默认 1.0（总是应用）
    """
    def __init__(self, brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0, p=1.0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img

        # 转换到 HSV 用于调整饱和度和色调
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)

        # 亮度调整（在 HSV 的 V 通道）
        if self.brightness > 0:
            alpha = 1.0 + random.uniform(-self.brightness, self.brightness)
            img_hsv[:, :, 2] = np.clip(img_hsv[:, :, 2] * alpha, 0, 255)

        # 饱和度调整（在 HSV 的 S 通道）
        if self.saturation > 0:
            alpha = 1.0 + random.uniform(-self.saturation, self.saturation)
            img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1] * alpha, 0, 255)

        # 色调调整（在 HSV 的 H 通道，范围 0-180）
        if self.hue > 0:
            delta = random.uniform(-self.hue, self.hue) * 180
            img_hsv[:, :, 0] = (img_hsv[:, :, 0] + delta) % 180

        # 转回 RGB
        img = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        # 对比度调整（在 RGB 上）
        if self.contrast > 0:
            alpha = 1.0 + random.uniform(-self.contrast, self.contrast)
            img = np.clip(img.astype(np.float32) * alpha, 0, 255).astype(np.uint8)

        return img


class RandomRotation:
    """随机旋转

    Args:
        degrees (float or tuple): 旋转角度范围
            - 如果是单个数字，范围是 [-degrees, +degrees]
            - 如果是 tuple (min, max)，范围是 [min, max]
        p (float): 应用概率，默认 1.0
        fill (tuple): 填充颜色，默认 (0, 0, 0) 黑色
    """
    def __init__(self, degrees=10, p=1.0, fill=(0, 0, 0)):
        if isinstance(degrees, (int, float)):
            self.degrees = (-degrees, degrees)
        else:
            assert len(degrees) == 2
            self.degrees = degrees
        self.p = p
        self.fill = fill

    def __call__(self, img):
        if random.random() > self.p:
            return img

        angle = random.uniform(self.degrees[0], self.degrees[1])
        h, w = img.shape[:2]
        center = (w // 2, h // 2)

        # 获取旋转矩阵
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # 执行旋转
        img = cv2.warpAffine(
            img, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=self.fill
        )
        return img


class RandomGrayscale:
    """随机转灰度图（保持 3 通道）

    Args:
        p (float): 转换概率，默认 0.1
    """
    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            # 转回 3 通道
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        return img


class RandomGaussianBlur:
    """随机高斯模糊

    Args:
        kernel_size (int or tuple): 高斯核大小，必须是奇数
        sigma (float or tuple): 高斯核标准差范围
        p (float): 应用概率，默认 0.5
    """
    def __init__(self, kernel_size=5, sigma=(0.1, 2.0), p=0.5):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        assert kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1, "kernel_size 必须是奇数"

        self.kernel_size = kernel_size
        if isinstance(sigma, (int, float)):
            self.sigma = (sigma, sigma)
        else:
            self.sigma = sigma
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.uniform(self.sigma[0], self.sigma[1])
            return cv2.GaussianBlur(img, self.kernel_size, sigma)
        return img


class RandomErasing:
    """随机擦除（Cutout）- 在 Normalize 之后、ToTensor 之后应用

    注意：这个需要在 numpy array (CHW format) 上操作

    Args:
        p (float): 应用概率
        scale (tuple): 擦除区域占图像的比例范围，默认 (0.02, 0.33)
        ratio (tuple): 擦除区域的宽高比范围，默认 (0.3, 3.3)
        value (float or tuple): 填充值，默认 0（可以是单个值或每通道的值）
    """
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value

    def __call__(self, img):
        """
        Args:
            img: numpy array, shape (C, H, W) after NormalizeImage with order='chw'
        """
        if random.random() > self.p:
            return img

        if len(img.shape) == 3:
            c, h, w = img.shape
        else:
            h, w = img.shape
            c = 1

        # 计算擦除区域面积
        area = h * w
        target_area = random.uniform(self.scale[0], self.scale[1]) * area
        aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])

        # 计算擦除区域的高和宽
        erase_h = int(round(np.sqrt(target_area * aspect_ratio)))
        erase_w = int(round(np.sqrt(target_area / aspect_ratio)))

        if erase_h < h and erase_w < w:
            # 随机选择擦除区域的位置
            top = random.randint(0, h - erase_h)
            left = random.randint(0, w - erase_w)

            # 执行擦除
            if c == 1:
                img[top:top+erase_h, left:left+erase_w] = self.value
            else:
                if isinstance(self.value, (int, float)):
                    img[:, top:top+erase_h, left:left+erase_w] = self.value
                else:
                    for i in range(c):
                        img[i, top:top+erase_h, left:left+erase_w] = self.value[i]

        return img


class RandomResizedCrop:
    """随机缩放裁剪 - 类似 torchvision.transforms.RandomResizedCrop

    Args:
        size (int or tuple): 目标大小
        scale (tuple): 缩放范围，默认 (0.08, 1.0)
        ratio (tuple): 宽高比范围，默认 (3/4, 4/3)
        interpolation: cv2 插值方法，默认 cv2.INTER_LINEAR
    """
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3./4., 4./3.), interpolation=cv2.INTER_LINEAR):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation

    def __call__(self, img):
        h, w = img.shape[:2]
        area = h * w

        for _ in range(10):  # 最多尝试 10 次
            target_area = random.uniform(self.scale[0], self.scale[1]) * area
            aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])

            crop_w = int(round(np.sqrt(target_area * aspect_ratio)))
            crop_h = int(round(np.sqrt(target_area / aspect_ratio)))

            if crop_w <= w and crop_h <= h:
                top = random.randint(0, h - crop_h)
                left = random.randint(0, w - crop_w)

                cropped = img[top:top+crop_h, left:left+crop_w]
                return cv2.resize(cropped, self.size, interpolation=self.interpolation)

        # 如果 10 次都失败，则使用中心裁剪
        in_ratio = w / h
        if in_ratio < min(self.ratio):
            crop_w = w
            crop_h = int(round(crop_w / min(self.ratio)))
        elif in_ratio > max(self.ratio):
            crop_h = h
            crop_w = int(round(crop_h * max(self.ratio)))
        else:
            crop_w = w
            crop_h = h

        top = (h - crop_h) // 2
        left = (w - crop_w) // 2
        cropped = img[top:top+crop_h, left:left+crop_w]
        return cv2.resize(cropped, self.size, interpolation=self.interpolation)

