"""Metric Learning Losses

All metric learning loss functions
"""

from .metric_losses import *

# Try to import from parent if available
try:
    from ..triplet import BatchHardTripletLoss
    from ..circle_loss import CircleLoss
except:
    BatchHardTripletLoss = None
    CircleLoss = None

__all__ = [
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
]

if BatchHardTripletLoss:
    __all__.append('BatchHardTripletLoss')
if CircleLoss:
    __all__.append('CircleLoss')


