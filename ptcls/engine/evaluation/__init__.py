# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Migrated to PyTorch for visionhub project.

from .classification import classification_eval
from .retrieval import retrieval_eval

__all__ = [
    'classification_eval',
    'retrieval_eval'
]

