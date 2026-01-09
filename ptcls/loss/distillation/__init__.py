"""Distillation Losses

All knowledge distillation loss functions
"""

from .distillation_losses import *

__all__ = [
    'KLDivLoss',
    'DKDLoss',
    'RKDLoss',
    'SKDLoss',
    'AttentionTransferLoss',
    'FactorTransferLoss',
    'SimilarityPreservingLoss',
]

