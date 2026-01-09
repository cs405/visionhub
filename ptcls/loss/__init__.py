"""ptcls.loss

Loss modules for classification / retrieval.

This package is intentionally small and grows as we port more visionhub loss heads.
"""

from .celoss import CELoss
from .circle_loss import CircleLoss
from .arcface import ArcFaceHead

__all__ = [
    "CELoss",
    "CircleLoss",
    "ArcFaceHead",
]
