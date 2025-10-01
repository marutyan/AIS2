"""
モデル管理ユーティリティ

各種モデルファイルのダウンロード、管理、キャッシュ機能を提供
RT-DETRv2およびその他のモデルの自動取得
"""

import os
import hashlib
import requests
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import json
from urllib.parse import urlparse
import shutil


class ModelManager:
    """
    モデルファイルの管理クラス
    
    機能:
    - モデルファイルの自動ダウンロード
    - ハッシュ値による整合性チェック
    - キャッシュ管理
    - モデル情報の管理
    """
    
    # サポートされているモデルの定義
    SUPPORTED_MODELS = {
        'rtdetrv2_r18vd': {
            'url': 'https://github.com/lyuwenyu/RT-DETR/releases/download/v0.1.0/rtdetrv2_r18vd_coco_from_paddle.pth',
            'filename': 'rtdetrv2_r18vd_coco.pth',
            'description': 'RT-DETRv2 with ResNet-18 backbone',
            'size_mb': 85,
            'sha256': None  # 実際のハッシュ値は公式リポジトリから取得
        },
        'rtdetrv2_r34vd': {
            'url': 'https://github.com/lyuwenyu/RT-DETR/releases/download/v0.1.0/rtdetrv2_r34vd_coco_from_paddle.pth',
            'filename': 'rtdetrv2_r34vd_coco.pth',
            'description': 'RT-DETRv2 with ResNet-34 backbone',
            'size_mb': 120,
            'sha256': None
        },
        'rtdetrv2_r50vd': {
            'url': 'https://github.com/lyuwenyu/RT-DETR/releases/download/v0.1.0/rtdetrv2_r50vd_coco_from_paddle.pth',
            'filename': 'rtdetrv2_r50vd_coco.pth',
            'description': 'RT-DETRv2 with ResNet-50 backbone',
            'size_mb': 150,
            'sha256': None
        },
        'yolov8n': {
            'url': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt',
            'filename': 'yolov8n.pt',
            'description': 'YOLOv8 Nano - fallback model',
            'size_mb': 6,
            'sha256': None
        },
        'yolov8s': {
            'url': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt',
            'filename': 'yolov8s.pt',
            'description': 'YOLOv8 Small - fallback model',
            'size_mb': 22,
            'sha256': None
        }
    }
    
    def __init__(self, models_dir: Optional[str] = None):
        """
        モデル管理クラスの初期化
        
        Args:
            models_dir: モデルファイルを保存するディレクトリ
        """
        if models_dir is None:
            # デフォルトのモデルディレクトリを設定
            models_dir = Path(__file__).parent / "weights"
        
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # モデル情報ファイル
        self.info_file = self.models_dir / "models_info.json"
        self.model_info = self._load_model_info()
        
        print(f"モデル管理クラスを初期化: {self.models_dir}")
    
    def _load_model_info(self) -> Dict:
        """
        モデル情報ファイルを読み込み
        
        Returns:
            モデル情報の辞書
        """
        if self.info_file.exists():
            try:
                with open(self.info_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"モデル情報ファイルの読み込みに失敗: {e}")
        
        return {}
    
    def _save_model_info(self) -> None:
        """
        モデル情報ファイルを保存
        """
        try:
            with open(self.info_file, 'w', encoding='utf-8') as f:
                json.dump(self.model_info, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"モデル情報ファイルの保存に失敗: {e}")
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """
        ファイルのSHA256ハッシュを計算
        
        Args:
            file_path: ファイルパス
            
        Returns:
            SHA256ハッシュ値
        """
        hash_sha256 = hashlib.sha256()
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        
        return hash_sha256.hexdigest()
    
    def download_model(
        self,
        model_name: str,
        force_download: bool = False,
        show_progress: bool = True
    ) -> str:
        """
        モデルファイルをダウンロード
        
        Args:
            model_name: モデル名
            force_download: 強制的に再ダウンロードするかどうか
            show_progress: ダウンロード進捗を表示するかどうか
            
        Returns:
            ダウンロードされたモデルファイルのパス
        """
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"サポートされていないモデル: {model_name}")
        
        model_config = self.SUPPORTED_MODELS[model_name]
        model_path = self.models_dir / model_config['filename']
        
        # ファイルが既に存在し、強制ダウンロードが無効の場合はスキップ
        if model_path.exists() and not force_download:
            if show_progress:
                print(f"モデルは既に存在します: {model_path}")
            return str(model_path)
        
        # ダウンロード実行
        print(f"モデルをダウンロード中: {model_name}")
        print(f"  URL: {model_config['url']}")
        print(f"  サイズ: 約{model_config['size_mb']}MB")
        
        try:
            response = requests.get(model_config['url'], stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0
            
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        
                        if show_progress and total_size > 0:
                            progress = (downloaded_size / total_size) * 100
                            print(f"\\rダウンロード進行状況: {progress:.1f}%", end="", flush=True)
            
            if show_progress:
                print(f"\\nダウンロード完了: {model_path}")
            
            # モデル情報を更新
            self._update_model_info(model_name, model_path)
            
            return str(model_path)
            
        except Exception as e:
            # ダウンロードに失敗した場合、部分ファイルを削除
            if model_path.exists():
                model_path.unlink()
            
            print(f"\\nモデルのダウンロードに失敗: {e}")
            raise
    
    def _update_model_info(self, model_name: str, model_path: Path) -> None:
        """
        モデル情報を更新
        
        Args:
            model_name: モデル名
            model_path: モデルファイルのパス
        """
        file_size = model_path.stat().st_size
        file_hash = self._calculate_file_hash(model_path)
        
        self.model_info[model_name] = {
            'filename': model_path.name,
            'file_size': file_size,
            'sha256': file_hash,
            'download_time': str(model_path.stat().st_mtime),
            'description': self.SUPPORTED_MODELS[model_name]['description']
        }
        
        self._save_model_info()
    
    def verify_model(self, model_name: str) -> bool:
        """
        モデルファイルの整合性を検証
        
        Args:
            model_name: モデル名
            
        Returns:
            整合性が正しいかどうか
        """
        if model_name not in self.SUPPORTED_MODELS:
            return False
        
        model_config = self.SUPPORTED_MODELS[model_name]
        model_path = self.models_dir / model_config['filename']
        
        if not model_path.exists():
            return False
        
        # ハッシュ値が設定されている場合のみ検証
        expected_hash = model_config.get('sha256')
        if expected_hash:
            actual_hash = self._calculate_file_hash(model_path)
            return actual_hash == expected_hash
        
        # ハッシュ値が設定されていない場合はファイルの存在のみ確認
        return True
    
    def get_model_path(self, model_name: str, auto_download: bool = True) -> str:
        """
        モデルファイルのパスを取得
        
        Args:
            model_name: モデル名
            auto_download: 存在しない場合に自動ダウンロードするかどうか
            
        Returns:
            モデルファイルのパス
        """
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"サポートされていないモデル: {model_name}")
        
        model_config = self.SUPPORTED_MODELS[model_name]
        model_path = self.models_dir / model_config['filename']
        
        if not model_path.exists() and auto_download:
            return self.download_model(model_name)
        
        return str(model_path)
    
    def list_available_models(self) -> List[str]:
        """
        利用可能なモデルのリストを取得
        
        Returns:
            モデル名のリスト
        """
        return list(self.SUPPORTED_MODELS.keys())
    
    def list_downloaded_models(self) -> List[Tuple[str, Path, int]]:
        """
        ダウンロード済みモデルのリストを取得
        
        Returns:
            (モデル名, ファイルパス, ファイルサイズ) のリスト
        """
        downloaded = []
        
        for model_name, config in self.SUPPORTED_MODELS.items():
            model_path = self.models_dir / config['filename']
            if model_path.exists():
                file_size = model_path.stat().st_size
                downloaded.append((model_name, model_path, file_size))
        
        return downloaded
    
    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """
        モデルの詳細情報を取得
        
        Args:
            model_name: モデル名
            
        Returns:
            モデル情報の辞書
        """
        if model_name not in self.SUPPORTED_MODELS:
            return None
        
        config = self.SUPPORTED_MODELS[model_name].copy()
        
        # ローカルファイル情報を追加
        model_path = self.models_dir / config['filename']
        config['local_path'] = str(model_path)
        config['exists'] = model_path.exists()
        
        if model_path.exists():
            config['actual_size'] = model_path.stat().st_size
            config['download_time'] = model_path.stat().st_mtime
        
        return config
    
    def cleanup_old_models(self, keep_latest: int = 2) -> None:
        """
        古いモデルファイルをクリーンアップ
        
        Args:
            keep_latest: 保持する最新ファイル数
        """
        print(f"古いモデルファイルをクリーンアップ中...")
        
        # 各モデルの複数バージョンがある場合の処理
        # 現在の実装では単一バージョンのみなのでスキップ
        
        print("クリーンアップ完了")
    
    def get_storage_usage(self) -> Dict[str, int]:
        """
        ストレージ使用量を取得
        
        Returns:
            使用量情報の辞書
        """
        total_size = 0
        model_sizes = {}
        
        for model_name, config in self.SUPPORTED_MODELS.items():
            model_path = self.models_dir / config['filename']
            if model_path.exists():
                size = model_path.stat().st_size
                model_sizes[model_name] = size
                total_size += size
        
        return {
            'total_size': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'model_sizes': model_sizes,
            'models_count': len(model_sizes)
        }
    
    def __str__(self) -> str:
        downloaded = len(self.list_downloaded_models())
        available = len(self.list_available_models())
        return f"ModelManager(downloaded={downloaded}/{available}, dir={self.models_dir})"
