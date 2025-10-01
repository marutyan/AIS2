"""
MOT (Multiple Object Tracking) System

拡張可能な物体検出・追跡システム
RT-DETRv2を基盤とし、将来的にセグメンテーションなどのモデル追加にも対応
"""

__version__ = "0.1.0"
__author__ = "CVLAB Team"

from .detection import RTDETRDetector, BaseDetector
from .tracking import SimpleTracker, BaseTracker
from .utils import VideoProcessor, Visualizer, Config

__all__ = ["RTDETRDetector", "BaseDetector", "SimpleTracker", "BaseTracker", "VideoProcessor", "Visualizer", "Config"]
