"""Backbone networks for various architectures"""

from .registry import register_backbone, build_backbone, list_backbones, BACKBONE_REGISTRY
from .resnet import ResNet18, ResNet50

# Import all CNN backbones
from . import cnn

# Import all Transformer backbones
from . import transformer

__all__ = [
    'register_backbone',
    'build_backbone',
    'list_backbones',
    'BACKBONE_REGISTRY',
    'ResNet18',
    'ResNet50',
]
