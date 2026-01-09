"""
visionhub - PyTorch Image Classification Toolkit

A comprehensive PyTorch-based image classification toolkit, developed by visionhub.
Supports classification, retrieval, face recognition, and YOLO integration.

Author: JKDCPPZzz
Version: 1.0.0
License: Apache-2.0
"""

__version__ = '1.0.0'
__author__ = 'JKDCPPZzz'

# Import main components
from .ptcls.arch.backbone import build_backbone, list_backbones
from .ptcls.loss import *
from .ptcls.data import *
from .ptcls.metric import *

__all__ = [
    'build_backbone',
    'list_backbones',
    '__version__',
    '__author__'
]

