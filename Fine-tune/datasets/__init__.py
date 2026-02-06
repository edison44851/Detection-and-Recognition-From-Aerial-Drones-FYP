"""Dataset loaders for crowd counting and detection tasks."""

from .crowd import Crowd
from .dm_crowd import Crowd as Crowd_DM
from .dm_detection import DetectionDataset

__all__ = [
    'Crowd',
    'Crowd_DM',
    'DetectionDataset',
]
