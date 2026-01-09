"""Backbone Registry System

统一的 Backbone 注册和加载机制
"""

import torch.nn as nn
from typing import Dict, Type, Any

# 全局注册表
BACKBONE_REGISTRY: Dict[str, Type[nn.Module]] = {}


def register_backbone(name: str):
    """
    Backbone 注册装饰器

    使用方法:
        @register_backbone('mobilenet_v2')
        class MobileNetV2(nn.Module):
            ...
    """
    def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        if name in BACKBONE_REGISTRY:
            raise ValueError(f"Backbone '{name}' already registered!")
        BACKBONE_REGISTRY[name] = cls
        return cls
    return decorator


def build_backbone(name: str, **kwargs) -> nn.Module:
    """
    根据名称构建 Backbone

    Args:
        name: Backbone 名称 (如 'resnet18', 'mobilenet_v2')
        **kwargs: 传递给 Backbone 的参数

    Returns:
        Backbone 实例
    """
    if name not in BACKBONE_REGISTRY:
        raise ValueError(
            f"Backbone '{name}' not found in registry. "
            f"Available: {list(BACKBONE_REGISTRY.keys())}"
        )

    return BACKBONE_REGISTRY[name](**kwargs)


def list_backbones() -> list:
    """列出所有已注册的 Backbone"""
    return sorted(BACKBONE_REGISTRY.keys())


# 导出
__all__ = [
    'register_backbone',
    'build_backbone',
    'list_backbones',
    'BACKBONE_REGISTRY',
]

