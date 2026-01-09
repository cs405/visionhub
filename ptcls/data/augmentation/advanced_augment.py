"""Advanced Data Augmentation

完整的数据增强策略，对齐visionhub:
- AutoAugment (自动增强策略)
- RandAugment (随机增强)
- FMix (Fourier Mix)
- HideAndSeek (隐藏并寻找)
- GridMask (网格遮罩)
- Mixup (混合增强)
- Cutmix (裁剪混合)
- CutOut (随机擦除)
- RandomErasing (随机擦除v2)
"""

import torch
import numpy as np
import random
import math
from PIL import Image, ImageEnhance, ImageOps
import torch.nn.functional as F


# ============ AutoAugment ============
class AutoAugment:
    """AutoAugment data augmentation policy

    Paper: AutoAugment: Learning Augmentation Strategies from Data
    """

    def __init__(self, policy='imagenet'):
        self.policy = policy
        self.policies = self._get_policies()

    def _get_policies(self):
        """返回增强策略"""
        if self.policy == 'imagenet':
            return [
                [('Posterize', 0.4, 8), ('Rotate', 0.6, 9)],
                [('Solarize', 0.6, 5), ('AutoContrast', 0.6, 5)],
                [('Equalize', 0.8, 8), ('Equalize', 0.6, 3)],
                [('Posterize', 0.6, 7), ('Posterize', 0.6, 6)],
                [('Equalize', 0.4, 7), ('Solarize', 0.2, 4)],
                [('Equalize', 0.4, 4), ('Rotate', 0.8, 8)],
                [('Solarize', 0.6, 3), ('Equalize', 0.6, 7)],
                [('Posterize', 0.8, 5), ('Equalize', 1.0, 2)],
                [('Rotate', 0.2, 3), ('Solarize', 0.6, 8)],
                [('Equalize', 0.6, 8), ('Posterize', 0.4, 6)],
                [('Rotate', 0.8, 8), ('Color', 0.4, 0)],
                [('Rotate', 0.4, 9), ('Equalize', 0.6, 2)],
                [('Equalize', 0.0, 7), ('Equalize', 0.8, 8)],
                [('Invert', 0.6, 4), ('Equalize', 1.0, 8)],
                [('Color', 0.6, 4), ('Contrast', 1.0, 8)],
                [('Rotate', 0.8, 8), ('Color', 1.0, 2)],
                [('Color', 0.8, 8), ('Solarize', 0.8, 7)],
                [('Sharpness', 0.4, 7), ('Invert', 0.6, 8)],
                [('ShearX', 0.6, 5), ('Equalize', 1.0, 9)],
                [('Color', 0.4, 0), ('Equalize', 0.6, 3)],
                [('Equalize', 0.4, 7), ('Solarize', 0.2, 4)],
                [('Solarize', 0.6, 5), ('AutoContrast', 0.6, 5)],
                [('Invert', 0.6, 4), ('Equalize', 1.0, 8)],
                [('Color', 0.6, 4), ('Contrast', 1.0, 8)],
                [('Equalize', 0.8, 8), ('Equalize', 0.6, 3)],
            ]
        else:
            raise ValueError(f"Unknown policy: {self.policy}")

    def __call__(self, img):
        """应用随机策略"""
        policy = random.choice(self.policies)
        for op_name, prob, magnitude in policy:
            if random.random() < prob:
                img = self._apply_op(img, op_name, magnitude)
        return img

    def _apply_op(self, img, op_name, magnitude):
        """应用具体操作"""
        if op_name == 'Rotate':
            return img.rotate(magnitude * 3)
        elif op_name == 'ShearX':
            return img.transform(img.size, Image.AFFINE, (1, magnitude/10, 0, 0, 1, 0))
        elif op_name == 'ShearY':
            return img.transform(img.size, Image.AFFINE, (1, 0, 0, magnitude/10, 1, 0))
        elif op_name == 'Color':
            return ImageEnhance.Color(img).enhance(1 + magnitude/10)
        elif op_name == 'Contrast':
            return ImageEnhance.Contrast(img).enhance(1 + magnitude/10)
        elif op_name == 'Brightness':
            return ImageEnhance.Brightness(img).enhance(1 + magnitude/10)
        elif op_name == 'Sharpness':
            return ImageEnhance.Sharpness(img).enhance(1 + magnitude/10)
        elif op_name == 'Posterize':
            return ImageOps.posterize(img, int(magnitude))
        elif op_name == 'Solarize':
            return ImageOps.solarize(img, int(magnitude * 25.6))
        elif op_name == 'AutoContrast':
            return ImageOps.autocontrast(img)
        elif op_name == 'Equalize':
            return ImageOps.equalize(img)
        elif op_name == 'Invert':
            return ImageOps.invert(img)
        return img


# ============ RandAugment ============
class RandAugment:
    """RandAugment: Practical automated data augmentation

    Paper: RandAugment: Practical automated data augmentation with a reduced search space
    """

    def __init__(self, n=2, m=10):
        """
        Args:
            n: 每张图片应用n个操作
            m: 操作的幅度 (0-10)
        """
        self.n = n
        self.m = m
        self.augment_list = [
            'AutoContrast', 'Equalize', 'Invert', 'Rotate', 'Posterize',
            'Solarize', 'Color', 'Contrast', 'Brightness', 'Sharpness',
            'ShearX', 'ShearY', 'TranslateX', 'TranslateY'
        ]

    def __call__(self, img):
        """应用n个随机操作"""
        ops = random.choices(self.augment_list, k=self.n)
        for op in ops:
            img = self._apply_op(img, op, self.m)
        return img

    def _apply_op(self, img, op_name, magnitude):
        """应用操作"""
        magnitude = float(magnitude) / 10.0

        if op_name == 'Rotate':
            degrees = magnitude * 30
            return img.rotate(degrees)
        elif op_name == 'ShearX':
            return img.transform(img.size, Image.AFFINE, (1, magnitude*0.3, 0, 0, 1, 0))
        elif op_name == 'ShearY':
            return img.transform(img.size, Image.AFFINE, (1, 0, 0, magnitude*0.3, 1, 0))
        elif op_name == 'TranslateX':
            pixels = magnitude * img.size[0] * 0.3
            return img.transform(img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0))
        elif op_name == 'TranslateY':
            pixels = magnitude * img.size[1] * 0.3
            return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels))
        elif op_name == 'Color':
            return ImageEnhance.Color(img).enhance(1 + magnitude)
        elif op_name == 'Contrast':
            return ImageEnhance.Contrast(img).enhance(1 + magnitude)
        elif op_name == 'Brightness':
            return ImageEnhance.Brightness(img).enhance(1 + magnitude)
        elif op_name == 'Sharpness':
            return ImageEnhance.Sharpness(img).enhance(1 + magnitude)
        elif op_name == 'Posterize':
            bits = int(4 + magnitude * 4)
            return ImageOps.posterize(img, bits)
        elif op_name == 'Solarize':
            threshold = int(magnitude * 256)
            return ImageOps.solarize(img, threshold)
        elif op_name == 'AutoContrast':
            return ImageOps.autocontrast(img)
        elif op_name == 'Equalize':
            return ImageOps.equalize(img)
        elif op_name == 'Invert':
            return ImageOps.invert(img)
        return img


# ============ FMix ============
def fmix(images, labels, alpha=1.0, decay_power=3, shape=(224, 224)):
    """FMix: Enhancing Mixed Sample Data Augmentation

    Paper: FMix: Enhancing Mixed Sample Data Augmentation
    """
    batch_size = images.size(0)

    # 生成频域mask
    lam = np.random.beta(alpha, alpha)

    # 傅里叶空间采样
    h, w = shape
    mask = fmix_sample_mask(lam, decay_power, shape)
    mask = torch.from_numpy(mask).float().to(images.device)

    # 随机打乱
    index = torch.randperm(batch_size).to(images.device)

    # 混合
    mixed_images = images * mask + images[index] * (1 - mask)

    # 混合标签
    y_a, y_b = labels, labels[index]

    return mixed_images, y_a, y_b, lam


def fmix_sample_mask(lam, decay_power, shape):
    """生成FMix mask"""
    h, w = shape

    # 在频域生成mask
    fx = np.fft.fftfreq(h)[:, None]
    fy = np.fft.fftfreq(w)[None, :]
    freq = np.sqrt(fx**2 + fy**2)

    # 功率谱衰减
    spectrum = (freq + 1e-8) ** (-decay_power)

    # 随机相位
    phase = np.random.uniform(0, 2*np.pi, (h, w))

    # 生成mask
    mask = np.real(np.fft.ifft2(spectrum * np.exp(1j * phase)))
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    mask = (mask < lam).astype(np.float32)

    return mask[:, :, None]


# ============ GridMask ============
class GridMask:
    """GridMask Data Augmentation

    Paper: GridMask Data Augmentation
    """

    def __init__(self, d1=96, d2=224, rotate=1, ratio=0.6, mode=1, prob=0.8):
        self.d1 = d1
        self.d2 = d2
        self.rotate = rotate
        self.ratio = ratio
        self.mode = mode
        self.prob = prob

    def __call__(self, img):
        """应用GridMask"""
        if random.random() > self.prob:
            return img

        h, w = img.size(1), img.size(2)

        # 随机网格大小
        d = random.randint(self.d1, self.d2)

        # 生成mask
        mask = self._generate_mask(h, w, d)
        mask = torch.from_numpy(mask).float().to(img.device)

        # 应用mask
        if self.mode == 1:
            img = img * mask
        else:
            img = img * (1 - mask)

        return img

    def _generate_mask(self, h, w, d):
        """生成网格mask"""
        mask = np.ones((h, w), dtype=np.float32)

        # 网格参数
        l = int(d * self.ratio)

        # 随机起始点
        st_h = random.randint(0, d)
        st_w = random.randint(0, d)

        # 生成网格
        for i in range(h // d + 1):
            s = d * i + st_h
            t = min(s + l, h)
            mask[s:t, :] = 0

        for i in range(w // d + 1):
            s = d * i + st_w
            t = min(s + l, w)
            mask[:, s:t] = 0

        # 随机旋转
        if self.rotate:
            mask = Image.fromarray((mask * 255).astype(np.uint8))
            angle = random.randint(0, 360)
            mask = mask.rotate(angle)
            mask = np.array(mask) / 255.0

        return mask


# ============ HideAndSeek ============
class HideAndSeek:
    """Hide-and-Seek: A Data Augmentation Technique

    Paper: Hide-and-Seek: Forcing a Network to be Meticulous for Weakly-supervised Object and Action Localization
    """

    def __init__(self, grid_size=16, ratio=0.5, prob=0.5):
        self.grid_size = grid_size
        self.ratio = ratio
        self.prob = prob

    def __call__(self, img):
        """应用HideAndSeek"""
        if random.random() > self.prob:
            return img

        c, h, w = img.size()

        # 计算网格
        grid_h = h // self.grid_size
        grid_w = w // self.grid_size

        # 随机隐藏
        for i in range(grid_h):
            for j in range(grid_w):
                if random.random() < self.ratio:
                    h_start = i * self.grid_size
                    h_end = (i + 1) * self.grid_size
                    w_start = j * self.grid_size
                    w_end = (j + 1) * self.grid_size

                    img[:, h_start:h_end, w_start:w_end] = 0

        return img


# ============ Enhanced Mixup ============
def mixup_data_enhanced(x, y, alpha=1.0, use_cuda=True):
    """Enhanced Mixup with more strategies"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


# ============ Enhanced CutMix ============
def cutmix_data_enhanced(x, y, alpha=1.0, use_cuda=True):
    """Enhanced CutMix"""
    if alpha <= 0:
        return x, y, y, 1.0

    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]

    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    # Adjust lambda
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam


def rand_bbox(size, lam):
    """生成随机bbox"""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


# ============ CutOut / Random Erasing ============
class RandomErasing:
    """Random Erasing Data Augmentation

    Paper: Random Erasing Data Augmentation
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.485, 0.456, 0.406]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)

                if img.size()[0] == 3:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                else:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                return img

        return img


__all__ = [
    'AutoAugment',
    'RandAugment',
    'fmix',
    'GridMask',
    'HideAndSeek',
    'mixup_data_enhanced',
    'cutmix_data_enhanced',
    'RandomErasing'
]

