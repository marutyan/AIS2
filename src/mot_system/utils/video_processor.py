"""
ビデオ処理ユーティリティ

MP4ファイルの読み込み、フレーム処理、結果保存などの機能を提供
リアルタイム処理とバッチ処理の両方に対応
"""

import cv2
import os
import time
from typing import Generator, Tuple, Optional, Callable, Any, List
from pathlib import Path
import numpy as np

from ..detection.base_detector import Detection
from ..tracking.base_tracker import Track


class VideoProcessor:
    """
    ビデオファイルの処理を行うクラス
    
    機能:
    - MP4ファイルの読み込み
    - フレームごとの処理
    - 結果の保存
    - 進捗表示
    - フレームレート制御
    """
    
    def __init__(
        self,
        video_path: str,
        output_dir: Optional[str] = None,
        target_fps: Optional[float] = None,
        resize_factor: float = 1.0
    ):
        """
        ビデオプロセッサーの初期化
        
        Args:
            video_path: 入力ビデオファイルのパス
            output_dir: 出力ディレクトリ
            target_fps: 目標フレームレート（制限用）
            resize_factor: リサイズ倍率
        """
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir) if output_dir else self.video_path.parent / "outputs"
        self.target_fps = target_fps
        self.resize_factor = resize_factor
        
        # ビデオファイルの存在確認
        if not self.video_path.exists():
            raise FileNotFoundError(f"ビデオファイルが見つかりません: {video_path}")
        
        # 出力ディレクトリの作成
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # ビデオキャプチャーの初期化
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise ValueError(f"ビデオファイルを開けません: {video_path}")
        
        # ビデオ情報の取得
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.frame_count / self.fps
        
        # リサイズ後のサイズ
        self.output_width = int(self.width * resize_factor)
        self.output_height = int(self.height * resize_factor)
        
        print(f"ビデオ情報:")
        print(f"  ファイル: {self.video_path.name}")
        print(f"  解像度: {self.width}x{self.height} -> {self.output_width}x{self.output_height}")
        print(f"  FPS: {self.fps:.2f}")
        print(f"  フレーム数: {self.frame_count}")
        print(f"  時間: {self.duration:.2f}秒")
    
    def get_frame_generator(
        self, 
        start_frame: int = 0, 
        end_frame: Optional[int] = None
    ) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        フレームを順次取得するジェネレーター
        
        Args:
            start_frame: 開始フレーム番号
            end_frame: 終了フレーム番号（Noneの場合は最後まで）
            
        Yields:
            (frame_number, frame_image) のタプル
        """
        if end_frame is None:
            end_frame = self.frame_count
        
        # 開始フレームにシーク
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frame_number = start_frame
        while frame_number < end_frame:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # リサイズ処理
            if self.resize_factor != 1.0:
                frame = cv2.resize(frame, (self.output_width, self.output_height))
            
            yield frame_number, frame
            frame_number += 1
    
    def process_video(
        self,
        detector,
        tracker,
        visualizer=None,
        save_video: bool = True,
        save_frames: bool = False,
        show_progress: bool = True,
        process_every_n_frames: int = 1
    ) -> List[Tuple[int, List[Detection], List[Track]]]:
        """
        ビデオ全体を処理
        
        Args:
            detector: 物体検出器
            tracker: 物体追跡器
            visualizer: 可視化器（オプション）
            save_video: 結果ビデオを保存するかどうか
            save_frames: フレーム画像を保存するかどうか
            show_progress: 進捗を表示するかどうか
            process_every_n_frames: N フレームごとに処理（スキップ処理）
            
        Returns:
            (frame_number, detections, tracks) のリスト
        """
        results = []
        
        # ビデオライターの初期化
        video_writer = None
        if save_video:
            output_video_path = self.output_dir / f"{self.video_path.stem}_result.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                str(output_video_path),
                fourcc,
                self.fps,
                (self.output_width, self.output_height)
            )
        
        # フレーム保存ディレクトリ
        if save_frames:
            frames_dir = self.output_dir / "frames"
            frames_dir.mkdir(exist_ok=True)
        
        # 処理時間の計測
        start_time = time.time()
        processed_frames = 0
        
        try:
            for frame_number, frame in self.get_frame_generator():
                # フレームスキップ処理
                if frame_number % process_every_n_frames != 0:
                    continue
                
                # FPS制限
                if self.target_fps and processed_frames > 0:
                    elapsed = time.time() - start_time
                    expected_time = processed_frames / self.target_fps
                    if elapsed < expected_time:
                        time.sleep(expected_time - elapsed)
                
                # 物体検出
                detections = detector.detect(frame)
                
                # 物体追跡
                tracks = tracker.update(detections)
                
                # 結果を保存
                results.append((frame_number, detections, tracks))
                
                # 可視化
                output_frame = frame.copy()
                if visualizer:
                    output_frame = visualizer.draw_detections_and_tracks(
                        output_frame, detections, tracks
                    )
                
                # ビデオ保存
                if video_writer:
                    video_writer.write(output_frame)
                
                # フレーム画像保存
                if save_frames:
                    frame_path = frames_dir / f"frame_{frame_number:06d}.jpg"
                    cv2.imwrite(str(frame_path), output_frame)
                
                processed_frames += 1
                
                # 進捗表示
                if show_progress and processed_frames % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = processed_frames / elapsed if elapsed > 0 else 0
                    progress = frame_number / self.frame_count * 100
                    
                    print(f"\\r処理進捗: {progress:.1f}% "
                          f"({frame_number}/{self.frame_count}フレーム) "
                          f"FPS: {fps:.1f}", end="", flush=True)
        
        finally:
            # リソースのクリーンアップ
            if video_writer:
                video_writer.release()
            
            # 処理完了の表示
            if show_progress:
                elapsed = time.time() - start_time
                avg_fps = processed_frames / elapsed if elapsed > 0 else 0
                print(f"\\n処理完了: {processed_frames}フレーム, "
                      f"平均FPS: {avg_fps:.1f}, 処理時間: {elapsed:.2f}秒")
        
        return results
    
    def process_single_frame(
        self,
        frame_number: int,
        detector,
        tracker=None
    ) -> Tuple[List[Detection], List[Track]]:
        """
        単一フレームを処理
        
        Args:
            frame_number: 処理するフレーム番号
            detector: 物体検出器
            tracker: 物体追跡器（オプション）
            
        Returns:
            (detections, tracks) のタプル
        """
        # 指定フレームにシーク
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        
        if not ret:
            raise ValueError(f"フレーム {frame_number} を読み込めません")
        
        # リサイズ処理
        if self.resize_factor != 1.0:
            frame = cv2.resize(frame, (self.output_width, self.output_height))
        
        # 物体検出
        detections = detector.detect(frame)
        
        # 物体追跡
        tracks = []
        if tracker:
            tracks = tracker.update(detections)
        
        return detections, tracks
    
    def get_video_info(self) -> dict:
        """
        ビデオの詳細情報を取得
        
        Returns:
            ビデオ情報の辞書
        """
        return {
            "file_path": str(self.video_path),
            "file_name": self.video_path.name,
            "file_size_mb": self.video_path.stat().st_size / (1024 * 1024),
            "width": self.width,
            "height": self.height,
            "output_width": self.output_width,
            "output_height": self.output_height,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "duration_seconds": self.duration,
            "resize_factor": self.resize_factor
        }
    
    def extract_frames(
        self,
        output_dir: Optional[str] = None,
        interval: int = 30,
        format: str = "jpg"
    ) -> List[str]:
        """
        指定間隔でフレームを抽出
        
        Args:
            output_dir: 出力ディレクトリ
            interval: 抽出間隔（フレーム数）
            format: 保存形式
            
        Returns:
            保存されたファイルパスのリスト
        """
        if output_dir is None:
            output_dir = self.output_dir / "extracted_frames"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        
        for frame_number, frame in self.get_frame_generator():
            if frame_number % interval == 0:
                filename = f"frame_{frame_number:06d}.{format}"
                filepath = output_dir / filename
                
                if cv2.imwrite(str(filepath), frame):
                    saved_files.append(str(filepath))
        
        print(f"{len(saved_files)}枚のフレームを抽出しました: {output_dir}")
        return saved_files
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cap:
            self.cap.release()
    
    def __del__(self):
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
