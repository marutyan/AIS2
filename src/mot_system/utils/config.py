"""
設定管理ユーティリティ

アプリケーション全体の設定を管理
YAML/JSON形式の設定ファイルの読み書き
"""

import yaml
import json
from typing import Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class DetectionConfig:
    """物体検出の設定"""
    model_name: str = "rtdetrv2_r50vd"
    model_path: Optional[str] = None
    device: str = "auto"
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4
    input_size: tuple = (640, 640)


@dataclass
class TrackingConfig:
    """物体追跡の設定"""
    tracker_type: str = "simple"
    max_disappeared: int = 30
    max_distance: float = 100.0
    min_hits: int = 3
    use_iou: bool = False
    iou_threshold: float = 0.3


@dataclass
class VisualizationConfig:
    """可視化の設定"""
    show_confidence: bool = True
    show_class_names: bool = True
    show_track_ids: bool = True
    show_trajectories: bool = True
    trajectory_length: int = 30
    font_scale: float = 0.6
    line_thickness: int = 2


@dataclass
class VideoConfig:
    """ビデオ処理の設定"""
    target_fps: Optional[float] = None
    resize_factor: float = 1.0
    process_every_n_frames: int = 1
    save_video: bool = True
    save_frames: bool = False
    show_progress: bool = True


@dataclass
class OutputConfig:
    """出力の設定"""
    output_dir: str = "outputs"
    save_results: bool = True
    results_format: str = "json"  # "json", "csv", "txt"
    video_codec: str = "mp4v"


@dataclass
class MOTConfig:
    """MOTシステム全体の設定"""
    detection: DetectionConfig = DetectionConfig()
    tracking: TrackingConfig = TrackingConfig()
    visualization: VisualizationConfig = VisualizationConfig()
    video: VideoConfig = VideoConfig()
    output: OutputConfig = OutputConfig()


class Config:
    """
    設定管理クラス
    
    機能:
    - YAML/JSON設定ファイルの読み書き
    - デフォルト設定の提供
    - 設定の検証
    - 環境変数からの設定オーバーライド
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        設定管理クラスの初期化
        
        Args:
            config_path: 設定ファイルのパス
        """
        self.config_path = Path(config_path) if config_path else None
        self.config = MOTConfig()
        
        if self.config_path and self.config_path.exists():
            self.load_config(self.config_path)
    
    def load_config(self, config_path: Union[str, Path]) -> None:
        """
        設定ファイルを読み込み
        
        Args:
            config_path: 設定ファイルのパス
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    config_dict = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    config_dict = json.load(f)
                else:
                    raise ValueError(f"サポートされていない設定ファイル形式: {config_path.suffix}")
            
            self._update_config_from_dict(config_dict)
            self.config_path = config_path
            
            print(f"設定ファイルを読み込みました: {config_path}")
            
        except Exception as e:
            print(f"設定ファイルの読み込みに失敗: {e}")
            raise
    
    def save_config(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """
        設定ファイルを保存
        
        Args:
            config_path: 保存先パス（Noneの場合は元のパスを使用）
        """
        if config_path is None:
            config_path = self.config_path
        
        if config_path is None:
            raise ValueError("保存先パスが指定されていません")
        
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = asdict(self.config)
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True, indent=2)
                elif config_path.suffix.lower() == '.json':
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)
                else:
                    raise ValueError(f"サポートされていない設定ファイル形式: {config_path.suffix}")
            
            print(f"設定ファイルを保存しました: {config_path}")
            
        except Exception as e:
            print(f"設定ファイルの保存に失敗: {e}")
            raise
    
    def _update_config_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """
        辞書から設定を更新
        
        Args:
            config_dict: 設定辞書
        """
        # 各セクションの更新
        if 'detection' in config_dict:
            self._update_dataclass(self.config.detection, config_dict['detection'])
        
        if 'tracking' in config_dict:
            self._update_dataclass(self.config.tracking, config_dict['tracking'])
        
        if 'visualization' in config_dict:
            self._update_dataclass(self.config.visualization, config_dict['visualization'])
        
        if 'video' in config_dict:
            self._update_dataclass(self.config.video, config_dict['video'])
        
        if 'output' in config_dict:
            self._update_dataclass(self.config.output, config_dict['output'])
    
    def _update_dataclass(self, target_config: Any, update_dict: Dict[str, Any]) -> None:
        """
        データクラスを辞書で更新
        
        Args:
            target_config: 更新対象のデータクラス
            update_dict: 更新用の辞書
        """
        for key, value in update_dict.items():
            if hasattr(target_config, key):
                setattr(target_config, key, value)
            else:
                print(f"警告: 不明な設定項目: {key}")
    
    def create_default_config(self, output_path: Union[str, Path]) -> None:
        """
        デフォルト設定ファイルを作成
        
        Args:
            output_path: 出力パス
        """
        self.config = MOTConfig()  # デフォルト設定をリセット
        self.save_config(output_path)
    
    def get_detection_config(self) -> DetectionConfig:
        """物体検出設定を取得"""
        return self.config.detection
    
    def get_tracking_config(self) -> TrackingConfig:
        """物体追跡設定を取得"""
        return self.config.tracking
    
    def get_visualization_config(self) -> VisualizationConfig:
        """可視化設定を取得"""
        return self.config.visualization
    
    def get_video_config(self) -> VideoConfig:
        """ビデオ処理設定を取得"""
        return self.config.video
    
    def get_output_config(self) -> OutputConfig:
        """出力設定を取得"""
        return self.config.output
    
    def validate_config(self) -> bool:
        """
        設定の妥当性を検証
        
        Returns:
            設定が妥当かどうか
        """
        errors = []
        
        # 検出設定の検証
        if self.config.detection.confidence_threshold < 0 or self.config.detection.confidence_threshold > 1:
            errors.append("confidence_threshold は 0.0-1.0 の範囲で設定してください")
        
        if self.config.detection.nms_threshold < 0 or self.config.detection.nms_threshold > 1:
            errors.append("nms_threshold は 0.0-1.0 の範囲で設定してください")
        
        # 追跡設定の検証
        if self.config.tracking.max_disappeared <= 0:
            errors.append("max_disappeared は正の値を設定してください")
        
        if self.config.tracking.max_distance <= 0:
            errors.append("max_distance は正の値を設定してください")
        
        if self.config.tracking.min_hits <= 0:
            errors.append("min_hits は正の値を設定してください")
        
        # ビデオ設定の検証
        if self.config.video.resize_factor <= 0:
            errors.append("resize_factor は正の値を設定してください")
        
        if self.config.video.process_every_n_frames <= 0:
            errors.append("process_every_n_frames は正の値を設定してください")
        
        if errors:
            print("設定エラー:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        return True
    
    def print_config(self) -> None:
        """現在の設定を表示"""
        print("=== MOT System Configuration ===")
        config_dict = asdict(self.config)
        
        for section, values in config_dict.items():
            print(f"\n[{section.upper()}]")
            for key, value in values.items():
                print(f"  {key}: {value}")
    
    def __str__(self) -> str:
        return f"Config(path={self.config_path})"
