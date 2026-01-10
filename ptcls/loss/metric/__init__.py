"""Metric Learning Losses

All metric learning loss functions
"""

from .metric_losses import *
from .advanced_metric_loss import *
from .eml_loss import EMLLoss as EMLLossFull
from .trihard_loss import TriHardLoss
from .supcon_loss import SupConLoss as SupConLossFull

# Try to import from parent if available
try:
    from ..triplet import BatchHardTripletLoss
    from ..circle_loss import CircleLoss
except:
    BatchHardTripletLoss = None
    CircleLoss = None

__all__ = [
    # metric_losses.py
    'ArcFaceLoss',
    'CosFaceLoss',
    'SphereFaceLoss',
    'CenterLoss',
    'ContrastiveLoss',
    'NPairsLoss',
    'TripletAngularMarginLoss',
    'FocalLoss',
    'LabelSmoothingCrossEntropy',
    'SoftTargetCrossEntropy',
    # advanced_metric_loss.py
    'MSMLoss',
    'MetaBinLoss',
    'XBMLoss',
    'PairwiseCosFaceLoss',
    'EMLLoss',
    'SoftTripleLoss',
    'AngularLoss',
    'RankedListLoss',
    # 新迁移的PaddleClas损失
    'EMLLossFull',
    'TriHardLoss',
    'SupConLossFull',
]

if BatchHardTripletLoss:
    __all__.append('BatchHardTripletLoss')
if CircleLoss:
    __all__.append('CircleLoss')


