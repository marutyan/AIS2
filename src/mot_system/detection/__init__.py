"""
物体検出モジュール

RT-DETRv2を使用した物体検出機能を提供
将来的に他の検出モデルも追加可能な設計
"""

from .rt_detr_detector import RTDETRDetector
from .base_detector import BaseDetector

__all__ = ["RTDETRDetector", "BaseDetector"]
