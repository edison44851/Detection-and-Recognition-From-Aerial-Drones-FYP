"""Loss functions for detection and counting tasks."""

from .ot_loss import OT_Loss
from .LRD import RDLoss, CL1

__all__ = [
    'OT_Loss',
    'RDLoss',
    'CL1',
]
