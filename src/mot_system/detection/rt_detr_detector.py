"""
RT-DETRv2を使用した物体検出器

Real-Time Detection Transformer v2を使用した高速物体検出
公式リポジトリ: https://github.com/lyuwenyu/RT-DETR
"""

import os
import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import List, Optional, Tuple
from urllib.parse import urlparse
import requests
from pathlib import Path

from .base_detector import BaseDetector, Detection


class RTDETRDetector(BaseDetector):
    """
    RT-DETRv2を使用した物体検出器
    
    COCOデータセットで事前学習されたモデルを使用
    リアルタイム検出に最適化されたTransformerベースの検出器
    """
    
    # COCOデータセットのクラス名
    COCO_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
        'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
        'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
        'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush'
    ]
    
    # 事前学習済みモデルのURL
    MODEL_URLS = {
        'rtdetrv2_r18vd': 'https://github.com/lyuwenyu/RT-DETR/releases/download/v0.1.0/rtdetrv2_r18vd_coco_from_paddle.pth',
        'rtdetrv2_r34vd': 'https://github.com/lyuwenyu/RT-DETR/releases/download/v0.1.0/rtdetrv2_r34vd_coco_from_paddle.pth',
        'rtdetrv2_r50vd': 'https://github.com/lyuwenyu/RT-DETR/releases/download/v0.1.0/rtdetrv2_r50vd_coco_from_paddle.pth',
    }
    
    def __init__(
        self, 
        model_name: str = "rtdetrv2_r50vd",
        model_path: Optional[str] = None,
        device: str = "auto",
        input_size: Tuple[int, int] = (640, 640)
    ):
        """
        RT-DETRv2検出器の初期化
        
        Args:
            model_name: 使用するモデル名 ("rtdetrv2_r18vd", "rtdetrv2_r34vd", "rtdetrv2_r50vd")
            model_path: カスタムモデルファイルのパス
            device: 実行デバイス ("auto", "cpu", "cuda", "mps")
            input_size: 入力画像サイズ (height, width)
        """
        # デバイスの自動選択
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        super().__init__(model_path, device)
        
        self.model_name = model_name
        self.input_size = input_size
        self.class_names = self.COCO_CLASSES
        
        # モデルファイルのパスを決定
        if model_path is None:
            self.model_path = self._get_default_model_path()
        
        print(f"RT-DETRv2検出器を初期化: {model_name}, デバイス: {device}")
    
    def _get_default_model_path(self) -> str:
        """
        デフォルトのモデルファイルパスを取得
        
        Returns:
            モデルファイルのパス
        """
        models_dir = Path(__file__).parent.parent / "models" / "weights"
        models_dir.mkdir(parents=True, exist_ok=True)
        return str(models_dir / f"{self.model_name}.pth")
    
    def _download_model(self, url: str, save_path: str) -> None:
        """
        モデルファイルをダウンロード
        
        Args:
            url: ダウンロードURL
            save_path: 保存先パス
        """
        print(f"モデルをダウンロード中: {self.model_name}")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    
                    if total_size > 0:
                        progress = (downloaded_size / total_size) * 100
                        print(f"\\rダウンロード進行状況: {progress:.1f}%", end="", flush=True)
        
        print(f"\\nモデルのダウンロード完了: {save_path}")
    
    def load_model(self) -> None:
        """
        RT-DETRv2モデルをロード
        """
        try:
            # モデルファイルが存在しない場合はダウンロード
            if not os.path.exists(self.model_path):
                if self.model_name in self.MODEL_URLS:
                    self._download_model(self.MODEL_URLS[self.model_name], self.model_path)
                else:
                    raise FileNotFoundError(f"モデルファイルが見つかりません: {self.model_path}")
            
            # 現時点ではプレースホルダー実装
            # 実際のRT-DETRv2の実装が必要な場合は、公式リポジトリからコードを統合
            print(f"RT-DETRv2モデルをロード中: {self.model_path}")
            
            # TODO: 実際のRT-DETRv2モデルのロード実装
            # この部分は公式リポジトリのコードを統合する必要があります
            self.model = self._create_dummy_model()  # プレースホルダー
            
            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            
            print(f"モデルのロード完了: {self.device}デバイス")
            
        except Exception as e:
            print(f"モデルのロードに失敗: {e}")
            raise
    
    def _create_dummy_model(self) -> nn.Module:
        """
        RT-DETRv2モデルの作成
        
        公式リポジトリのRT-DETRv2実装を使用
        """
        try:
            # RT-DETRv2の公式実装をインポート
            import sys
            import os
            
            # RT-DETRリポジトリのパスを追加
            rt_detr_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "RT-DETR", "rtdetrv2_pytorch")
            if os.path.exists(rt_detr_path):
                sys.path.insert(0, rt_detr_path)
                sys.path.insert(0, os.path.join(rt_detr_path, "src"))
            
            # RT-DETRv2の設定ファイルを読み込み
            config_path = os.path.join(rt_detr_path, "configs", "rtdetrv2", f"{self.model_name}.yaml")
            if not os.path.exists(config_path):
                # デフォルト設定を使用
                config_path = os.path.join(rt_detr_path, "configs", "rtdetrv2", "rtdetrv2_r50vd.yaml")
            
            print(f"RT-DETRv2設定ファイルを使用: {config_path}")
            
            # RT-DETRv2のモデルを初期化
            from src.core import build_model
            
            # 設定を読み込み
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # モデルをビルド
            model = build_model(config['model'])
            
            # 事前学習済み重みをロード（利用可能な場合）
            if os.path.exists(self.model_path):
                print(f"事前学習済み重みをロード: {self.model_path}")
                checkpoint = torch.load(self.model_path, map_location='cpu')
                if 'model' in checkpoint:
                    model.load_state_dict(checkpoint['model'])
                else:
                    model.load_state_dict(checkpoint)
            
            return model
            
        except ImportError as e:
            print(f"RT-DETRv2のインポートに失敗: {e}")
            print("RT-DETRリポジトリの依存関係をインストールしてください:")
            print("cd RT-DETR/rtdetrv2_pytorch && pip install -r requirements.txt")
            
            # フォールバック: 基本的なTransformerベースの検出器
            return self._create_fallback_model()
        except Exception as e:
            print(f"RT-DETRv2モデルの作成に失敗: {e}")
            return self._create_fallback_model()
    
    def _create_fallback_model(self) -> nn.Module:
        """
        フォールバックモデルの作成
        
        RT-DETRv2が利用できない場合の代替実装
        """
        print("警告: RT-DETRv2が利用できません。フォールバックモデルを使用します")
        
        # 基本的なTransformerベースの検出器を実装
        class FallbackDetector(nn.Module):
            def __init__(self, num_classes=80):
                super().__init__()
                self.num_classes = num_classes
                # 簡単なCNNベースの検出器
                self.backbone = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1))
                )
                self.classifier = nn.Linear(128, num_classes)
                self.bbox_regressor = nn.Linear(128, 4)
            
            def forward(self, x):
                features = self.backbone(x)
                features = features.view(features.size(0), -1)
                class_logits = self.classifier(features)
                bbox_coords = self.bbox_regressor(features)
                return class_logits, bbox_coords
        
        return FallbackDetector()
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        RT-DETR用の画像前処理
        
        Args:
            image: 入力画像 (BGR format)
            
        Returns:
            前処理済み画像
        """
        # リサイズ
        height, width = image.shape[:2]
        target_h, target_w = self.input_size
        
        # アスペクト比を保持してリサイズ
        scale = min(target_w / width, target_h / height)
        new_w, new_h = int(width * scale), int(height * scale)
        
        resized = cv2.resize(image, (new_w, new_h))
        
        # パディング
        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        padded[:new_h, :new_w] = resized
        
        return padded
    
    def detect(
        self, 
        image: np.ndarray, 
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4
    ) -> List[Detection]:
        """
        画像から物体を検出
        
        Args:
            image: 入力画像 (BGR format)
            confidence_threshold: 信頼度の閾値
            nms_threshold: NMSの閾値
            
        Returns:
            検出結果のリスト
        """
        if not self.is_loaded:
            self.load_model()
        
        # 前処理
        processed_image = self.preprocess_image(image)
        
        try:
            # ダミー実装（YOLOv8を使用）
            if hasattr(self.model, 'predict'):
                results = self.model.predict(
                    processed_image, 
                    conf=confidence_threshold,
                    iou=nms_threshold,
                    verbose=False
                )
                
                detections = []
                if results and len(results) > 0:
                    result = results[0]
                    if result.boxes is not None:
                        boxes = result.boxes.xyxy.cpu().numpy()
                        scores = result.boxes.conf.cpu().numpy()
                        class_ids = result.boxes.cls.cpu().numpy().astype(int)
                        
                        # 元の画像サイズにスケール調整
                        original_h, original_w = image.shape[:2]
                        scale_x = original_w / self.input_size[1]
                        scale_y = original_h / self.input_size[0]
                        
                        for box, score, class_id in zip(boxes, scores, class_ids):
                            x1, y1, x2, y2 = box
                            
                            # 座標を元の画像サイズに調整
                            x1 *= scale_x
                            y1 *= scale_y
                            x2 *= scale_x
                            y2 *= scale_y
                            
                            if class_id < len(self.class_names):
                                detection = Detection(
                                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                                    confidence=float(score),
                                    class_id=int(class_id),
                                    class_name=self.class_names[class_id]
                                )
                                detections.append(detection)
                
                return self.postprocess_detections(detections, confidence_threshold)
            
        except Exception as e:
            print(f"検出処理でエラー: {e}")
            return []
        
        return []
    
    def __str__(self) -> str:
        return f"RTDETRDetector({self.model_name}, device={self.device}, loaded={self.is_loaded})"
