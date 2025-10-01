"""
物体検出の基底クラス

全ての物体検出モデルが継承すべき抽象基底クラス
将来的な拡張性を考慮した統一インターフェースを提供
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Union, Optional
import numpy as np
import cv2


class Detection:
    """
    単一の検出結果を表すデータクラス
    
    Attributes:
        bbox: バウンディングボックス [x1, y1, x2, y2]
        confidence: 信頼度スコア
        class_id: クラスID
        class_name: クラス名
        mask: セグメンテーションマスク（オプション）
    """
    
    def __init__(
        self,
        bbox: Tuple[float, float, float, float],
        confidence: float,
        class_id: int,
        class_name: str,
        mask: Optional[np.ndarray] = None
    ):
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name
        self.mask = mask  # 将来のセグメンテーション対応
    
    @property
    def center(self) -> Tuple[float, float]:
        """バウンディングボックスの中心座標を返す"""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    @property
    def area(self) -> float:
        """バウンディングボックスの面積を返す"""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)


class BaseDetector(ABC):
    """
    物体検出モデルの抽象基底クラス
    
    全ての検出モデルはこのクラスを継承し、統一されたインターフェースを実装する
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        """
        検出器の初期化
        
        Args:
            model_path: モデルファイルのパス
            device: 実行デバイス ("cpu", "cuda", "mps")
        """
        self.model_path = model_path
        self.device = device
        self.model = None
        self.class_names = []
        self.is_loaded = False
    
    @abstractmethod
    def load_model(self) -> None:
        """
        モデルをロードする
        
        継承クラスで実装必須のメソッド
        """
        pass
    
    @abstractmethod
    def detect(
        self, 
        image: np.ndarray, 
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4
    ) -> List[Detection]:
        """
        画像から物体を検出する
        
        Args:
            image: 入力画像 (BGR format)
            confidence_threshold: 信頼度の閾値
            nms_threshold: NMSの閾値
            
        Returns:
            検出結果のリスト
        """
        pass
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        画像の前処理（サブクラスでオーバーライド可能）
        
        Args:
            image: 入力画像
            
        Returns:
            前処理済み画像
        """
        return image
    
    def postprocess_detections(
        self, 
        detections: List[Detection],
        confidence_threshold: float = 0.5
    ) -> List[Detection]:
        """
        検出結果の後処理（フィルタリングなど）
        
        Args:
            detections: 検出結果のリスト
            confidence_threshold: 信頼度の閾値
            
        Returns:
            フィルタリング済みの検出結果
        """
        return [det for det in detections if det.confidence >= confidence_threshold]
    
    def get_class_names(self) -> List[str]:
        """
        クラス名のリストを取得
        
        Returns:
            クラス名のリスト
        """
        return self.class_names
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(device={self.device}, loaded={self.is_loaded})"
