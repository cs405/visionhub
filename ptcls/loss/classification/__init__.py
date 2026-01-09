"""Classification Losses

All classification loss functions
"""

from .additional_losses import *

__all__ = [
    'MultiLabelLoss',
    'AsymmetricLoss',
    'DeepHashLoss',
    'PairwiseRankingLoss',
    'AngularSoftmaxLoss',
    'LargeMarginCosineLoss',
    'ProxyNCALoss',
    'LiftedStructureLoss',
    'HistogramLoss',
]

