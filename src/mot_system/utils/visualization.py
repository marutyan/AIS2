"""
可視化ユーティリティ

検出結果と追跡結果の可視化機能を提供
バウンディングボックス、トラックID、軌跡などの描画
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import colorsys
import random

from ..detection.base_detector import Detection
from ..tracking.base_tracker import Track


class Visualizer:
    """
    検出・追跡結果の可視化クラス
    
    機能:
    - バウンディングボックスの描画
    - トラックIDの表示
    - 軌跡の描画
    - 統計情報の表示
    - カスタマイズ可能な色設定
    """
    
    def __init__(
        self,
        show_confidence: bool = True,
        show_class_names: bool = True,
        show_track_ids: bool = True,
        show_trajectories: bool = True,
        trajectory_length: int = 30,
        font_scale: float = 0.6,
        line_thickness: int = 2
    ):
        """
        可視化器の初期化
        
        Args:
            show_confidence: 信頼度を表示するかどうか
            show_class_names: クラス名を表示するかどうか
            show_track_ids: トラックIDを表示するかどうか
            show_trajectories: 軌跡を表示するかどうか
            trajectory_length: 軌跡の長さ（フレーム数）
            font_scale: フォントサイズ
            line_thickness: 線の太さ
        """
        self.show_confidence = show_confidence
        self.show_class_names = show_class_names
        self.show_track_ids = show_track_ids
        self.show_trajectories = show_trajectories
        self.trajectory_length = trajectory_length
        self.font_scale = font_scale
        self.line_thickness = line_thickness
        
        # 色の設定
        self.colors = self._generate_colors(100)  # 100種類の色を生成
        self.detection_color = (0, 255, 0)  # 緑色（検出用）
        self.track_color_map = {}  # トラックIDごとの色マップ
        
        # 軌跡の履歴
        self.trajectories: Dict[int, List[Tuple[float, float]]] = {}
    
    def _generate_colors(self, num_colors: int) -> List[Tuple[int, int, int]]:
        """
        指定数の色を生成
        
        Args:
            num_colors: 生成する色の数
            
        Returns:
            BGR色のリスト
        """
        colors = []
        for i in range(num_colors):
            hue = i / num_colors
            saturation = 0.8 + (i % 3) * 0.1  # 0.8-1.0の範囲
            value = 0.8 + (i % 2) * 0.2       # 0.8-1.0の範囲
            
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
            colors.append(bgr)
        
        return colors
    
    def get_track_color(self, track_id: int) -> Tuple[int, int, int]:
        """
        トラックIDに対応する色を取得
        
        Args:
            track_id: トラックID
            
        Returns:
            BGR色のタプル
        """
        if track_id not in self.track_color_map:
            color_index = track_id % len(self.colors)
            self.track_color_map[track_id] = self.colors[color_index]
        
        return self.track_color_map[track_id]
    
    def draw_detection(
        self,
        image: np.ndarray,
        detection: Detection,
        color: Optional[Tuple[int, int, int]] = None
    ) -> np.ndarray:
        """
        単一の検出結果を描画
        
        Args:
            image: 入力画像
            detection: 検出結果
            color: 描画色（Noneの場合はデフォルト色）
            
        Returns:
            描画済み画像
        """
        if color is None:
            color = self.detection_color
        
        x1, y1, x2, y2 = map(int, detection.bbox)
        
        # バウンディングボックスの描画
        cv2.rectangle(image, (x1, y1), (x2, y2), color, self.line_thickness)
        
        # ラベルテキストの作成
        label_parts = []
        if self.show_class_names:
            label_parts.append(detection.class_name)
        if self.show_confidence:
            label_parts.append(f"{detection.confidence:.2f}")
        
        if label_parts:
            label = " ".join(label_parts)
            
            # テキストサイズの計算
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 1
            )
            
            # ラベル背景の描画
            cv2.rectangle(
                image,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                color,
                -1
            )
            
            # テキストの描画
            cv2.putText(
                image,
                label,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                (255, 255, 255),
                1
            )
        
        return image
    
    def draw_track(
        self,
        image: np.ndarray,
        track: Track
    ) -> np.ndarray:
        """
        単一の追跡結果を描画
        
        Args:
            image: 入力画像
            track: 追跡結果
            
        Returns:
            描画済み画像
        """
        if not track.current_detection:
            return image
        
        color = self.get_track_color(track.track_id)
        
        # 検出結果の描画
        self.draw_detection(image, track.current_detection, color)
        
        # トラックIDの表示
        if self.show_track_ids:
            x1, y1, x2, y2 = map(int, track.current_detection.bbox)
            track_label = f"ID:{track.track_id}"
            
            cv2.putText(
                image,
                track_label,
                (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                color,
                2
            )
        
        # 軌跡の更新と描画
        if self.show_trajectories:
            center = track.current_detection.center
            
            # 軌跡履歴の更新
            if track.track_id not in self.trajectories:
                self.trajectories[track.track_id] = []
            
            self.trajectories[track.track_id].append(center)
            
            # 軌跡の長さを制限
            if len(self.trajectories[track.track_id]) > self.trajectory_length:
                self.trajectories[track.track_id].pop(0)
            
            # 軌跡の描画
            self._draw_trajectory(image, track.track_id, color)
        
        return image
    
    def _draw_trajectory(
        self,
        image: np.ndarray,
        track_id: int,
        color: Tuple[int, int, int]
    ) -> None:
        """
        軌跡を描画
        
        Args:
            image: 入力画像
            track_id: トラックID
            color: 描画色
        """
        if track_id not in self.trajectories:
            return
        
        trajectory = self.trajectories[track_id]
        if len(trajectory) < 2:
            return
        
        # 軌跡の線を描画
        for i in range(1, len(trajectory)):
            pt1 = (int(trajectory[i-1][0]), int(trajectory[i-1][1]))
            pt2 = (int(trajectory[i][0]), int(trajectory[i][1]))
            
            # 透明度を距離に応じて調整
            alpha = i / len(trajectory)
            thickness = max(1, int(self.line_thickness * alpha))
            
            cv2.line(image, pt1, pt2, color, thickness)
        
        # 最新の位置に円を描画
        if trajectory:
            center = (int(trajectory[-1][0]), int(trajectory[-1][1]))
            cv2.circle(image, center, 3, color, -1)
    
    def draw_detections_and_tracks(
        self,
        image: np.ndarray,
        detections: List[Detection],
        tracks: List[Track]
    ) -> np.ndarray:
        """
        検出結果と追跡結果をまとめて描画
        
        Args:
            image: 入力画像
            detections: 検出結果のリスト
            tracks: 追跡結果のリスト
            
        Returns:
            描画済み画像
        """
        output_image = image.copy()
        
        # 追跡結果の描画（優先）
        for track in tracks:
            if track.state == "active":
                output_image = self.draw_track(output_image, track)
        
        # マッチしていない検出結果の描画
        tracked_detections = {id(track.current_detection) for track in tracks if track.current_detection}
        
        for detection in detections:
            if id(detection) not in tracked_detections:
                output_image = self.draw_detection(output_image, detection)
        
        return output_image
    
    def draw_statistics(
        self,
        image: np.ndarray,
        detections: List[Detection],
        tracks: List[Track],
        frame_number: int,
        fps: Optional[float] = None
    ) -> np.ndarray:
        """
        統計情報を画像に描画
        
        Args:
            image: 入力画像
            detections: 検出結果のリスト
            tracks: 追跡結果のリスト
            frame_number: フレーム番号
            fps: フレームレート
            
        Returns:
            統計情報付きの画像
        """
        # 統計情報の計算
        active_tracks = len([t for t in tracks if t.state == "active"])
        total_detections = len(detections)
        
        # クラス別の検出数
        class_counts = {}
        for detection in detections:
            class_name = detection.class_name
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # 統計テキストの作成
        stats_lines = [
            f"Frame: {frame_number}",
            f"Detections: {total_detections}",
            f"Active Tracks: {active_tracks}",
        ]
        
        if fps is not None:
            stats_lines.append(f"FPS: {fps:.1f}")
        
        # クラス別統計の追加
        for class_name, count in sorted(class_counts.items()):
            stats_lines.append(f"{class_name}: {count}")
        
        # 統計情報の描画
        y_offset = 30
        for line in stats_lines:
            cv2.putText(
                image,
                line,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                (255, 255, 255),
                2
            )
            # 黒い縁取りを追加
            cv2.putText(
                image,
                line,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                (0, 0, 0),
                1
            )
            y_offset += 25
        
        return image
    
    def cleanup_old_trajectories(self, active_track_ids: List[int]) -> None:
        """
        非アクティブな追跡の軌跡をクリーンアップ
        
        Args:
            active_track_ids: アクティブなトラックIDのリスト
        """
        inactive_ids = [
            track_id for track_id in self.trajectories.keys()
            if track_id not in active_track_ids
        ]
        
        for track_id in inactive_ids:
            del self.trajectories[track_id]
    
    def reset(self) -> None:
        """
        可視化器をリセット
        """
        self.trajectories.clear()
        self.track_color_map.clear()
    
    def save_frame_with_annotations(
        self,
        image: np.ndarray,
        detections: List[Detection],
        tracks: List[Track],
        output_path: str,
        frame_number: int,
        fps: Optional[float] = None
    ) -> None:
        """
        注釈付きフレームを保存
        
        Args:
            image: 入力画像
            detections: 検出結果のリスト
            tracks: 追跡結果のリスト
            output_path: 出力パス
            frame_number: フレーム番号
            fps: フレームレート
        """
        annotated_image = self.draw_detections_and_tracks(image, detections, tracks)
        annotated_image = self.draw_statistics(annotated_image, detections, tracks, frame_number, fps)
        
        cv2.imwrite(output_path, annotated_image)
    
    def __str__(self) -> str:
        return (
            f"Visualizer("
            f"confidence={self.show_confidence}, "
            f"tracks={self.show_track_ids}, "
            f"trajectories={self.show_trajectories}"
            f")"
        )
