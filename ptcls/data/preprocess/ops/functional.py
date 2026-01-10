# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Migrated to PyTorch for visionhub project.

import numpy as np
from PIL import Image, ImageOps, ImageEnhance


def int_parameter(level, maxval):
    """缩放整数参数

    Args:
        level: 操作级别 [0, 10]
        maxval: 最大值

    Returns:
        缩放后的整数
    """
    return int(level * maxval / 10)


def float_parameter(level, maxval):
    """缩放浮点参数

    Args:
        level: 操作级别 [0, 10]
        maxval: 最大值

    Returns:
        缩放后的浮点数
    """
    return float(level) * maxval / 10.


def sample_level(n):
    """采样级别"""
    return np.random.uniform(low=0.1, high=n)


def autocontrast(pil_img, *args):
    """自动对比度"""
    return ImageOps.autocontrast(pil_img)


def equalize(pil_img, *args):
    """直方图均衡化"""
    return ImageOps.equalize(pil_img)


def posterize(pil_img, level, *args):
    """色调分离"""
    level = int_parameter(sample_level(level), 4)
    return ImageOps.posterize(pil_img, 4 - level)


def rotate(pil_img, level, *args):
    """旋转"""
    degrees = int_parameter(sample_level(level), 30)
    if np.random.uniform() > 0.5:
        degrees = -degrees
    return pil_img.rotate(degrees, resample=Image.BILINEAR)


def solarize(pil_img, level, *args):
    """曝光"""
    level = int_parameter(sample_level(level), 256)
    return ImageOps.solarize(pil_img, 256 - level)


def shear_x(pil_img, level):
    """X轴剪切"""
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform(
        pil_img.size,
        Image.AFFINE, (1, level, 0, 0, 1, 0),
        resample=Image.BILINEAR
    )


def shear_y(pil_img, level):
    """Y轴剪切"""
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform(
        pil_img.size,
        Image.AFFINE, (1, 0, 0, level, 1, 0),
        resample=Image.BILINEAR
    )


def translate_x(pil_img, level):
    """X轴平移"""
    level = int_parameter(sample_level(level), pil_img.size[0] / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform(
        pil_img.size,
        Image.AFFINE, (1, 0, level, 0, 1, 0),
        resample=Image.BILINEAR
    )


def translate_y(pil_img, level):
    """Y轴平移"""
    level = int_parameter(sample_level(level), pil_img.size[1] / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform(
        pil_img.size,
        Image.AFFINE, (1, 0, 0, 0, 1, level),
        resample=Image.BILINEAR
    )


def color(pil_img, level, *args):
    """颜色调整"""
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(level)


def contrast(pil_img, level, *args):
    """对比度调整"""
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(level)


def brightness(pil_img, level, *args):
    """亮度调整"""
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(level)


def sharpness(pil_img, level, *args):
    """锐度调整"""
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(level)


# 所有增强操作列表
augmentations = [
    autocontrast, equalize, posterize, rotate, solarize,
    shear_x, shear_y, translate_x, translate_y
]

augmentations_all = augmentations + [color, contrast, brightness, sharpness]

