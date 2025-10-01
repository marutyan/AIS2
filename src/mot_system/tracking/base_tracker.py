"""
物体追跡の基底クラス

全ての追跡アルゴリズムが継承すべき抽象基底クラス
Multiple Object Tracking (MOT) の統一インターフェースを提供
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
import numpy as np
from dataclasses import dataclass

from ..detection.base_detector import Detection


@dataclass
class Track:
    """
    追跡中の物体を表すデータクラス
    
    Attributes:
        track_id: 一意の追跡ID
        detections: 過去の検出結果の履歴
        state: 追跡状態 ("active", "lost", "deleted")
        age: 追跡開始からのフレーム数
        hits: 検出された回数
        time_since_update: 最後に更新されてからのフレーム数
        velocity: 移動速度 (x, y)
        predicted_bbox: 予測されたバウンディングボックス
    """
    
    track_id: int
    detections: List[Detection]
    state: str = "active"  # "active", "lost", "deleted"
    age: int = 0
    hits: int = 0
    time_since_update: int = 0
    velocity: Tuple[float, float] = (0.0, 0.0)
    predicted_bbox: Optional[Tuple[float, float, float, float]] = None
    
    @property
    def current_detection(self) -> Optional[Detection]:
        """最新の検出結果を取得"""
        return self.detections[-1] if self.detections else None
    
    @property
    def current_bbox(self) -> Optional[Tuple[float, float, float, float]]:
        """現在のバウンディングボックスを取得"""
        if self.current_detection:
            return self.current_detection.bbox
        return self.predicted_bbox
    
    @property
    def current_center(self) -> Optional[Tuple[float, float]]:
        """現在の中心座標を取得"""
        bbox = self.current_bbox
        if bbox:
            x1, y1, x2, y2 = bbox
            return ((x1 + x2) / 2, (y1 + y2) / 2)
        return None
    
    def update_velocity(self) -> None:
        """移動速度を更新"""
        if len(self.detections) >= 2:
            current_center = self.detections[-1].center
            prev_center = self.detections[-2].center
            
            self.velocity = (
                current_center[0] - prev_center[0],
                current_center[1] - prev_center[1]
            )
    
    def predict_next_position(self) -> Tuple[float, float]:
        """次のフレームでの位置を予測"""
        current_center = self.current_center
        if current_center:
            return (
                current_center[0] + self.velocity[0],
                current_center[1] + self.velocity[1]
            )
        return (0.0, 0.0)


class BaseTracker(ABC):
    """
    物体追跡アルゴリズムの抽象基底クラス
    
    全ての追跡アルゴリズムはこのクラスを継承し、統一されたインターフェースを実装する
    """
    
    def __init__(
        self,
        max_disappeared: int = 30,
        max_distance: float = 50.0,
        min_hits: int = 3
    ):
        """
        追跡器の初期化
        
        Args:
            max_disappeared: 物体が消失したと判断するフレーム数
            max_distance: 同一物体と判断する最大距離
            min_hits: 確定追跡に必要な最小検出回数
        """
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.min_hits = min_hits
        
        self.tracks: Dict[int, Track] = {}
        self.next_track_id = 0
        self.frame_count = 0
    
    @abstractmethod
    def update(self, detections: List[Detection]) -> List[Track]:
        """
        新しいフレームの検出結果で追跡を更新
        
        Args:
            detections: 現在のフレームの検出結果
            
        Returns:
            更新された追跡結果のリスト
        """
        pass
    
    def create_new_track(self, detection: Detection) -> Track:
        """
        新しい追跡を開始
        
        Args:
            detection: 初期検出結果
            
        Returns:
            新しい追跡オブジェクト
        """
        track = Track(
            track_id=self.next_track_id,
            detections=[detection],
            age=1,
            hits=1,
            time_since_update=0
        )
        
        self.tracks[self.next_track_id] = track
        self.next_track_id += 1
        
        return track
    
    def update_track(self, track: Track, detection: Detection) -> None:
        """
        既存の追跡を更新
        
        Args:
            track: 更新する追跡オブジェクト
            detection: 新しい検出結果
        """
        track.detections.append(detection)
        track.hits += 1
        track.time_since_update = 0
        track.age += 1
        track.state = "active"
        
        # 速度を更新
        track.update_velocity()
    
    def predict_tracks(self) -> None:
        """
        全ての追跡の次の位置を予測
        """
        for track in self.tracks.values():
            if track.state == "active":
                # 簡単な線形予測
                predicted_center = track.predict_next_position()
                
                if track.current_bbox:
                    x1, y1, x2, y2 = track.current_bbox
                    width = x2 - x1
                    height = y2 - y1
                    
                    track.predicted_bbox = (
                        predicted_center[0] - width / 2,
                        predicted_center[1] - height / 2,
                        predicted_center[0] + width / 2,
                        predicted_center[1] + height / 2
                    )
    
    def calculate_distance(
        self, 
        detection: Detection, 
        track: Track
    ) -> float:
        """
        検出結果と追跡の間の距離を計算
        
        Args:
            detection: 検出結果
            track: 追跡オブジェクト
            
        Returns:
            ユークリッド距離
        """
        det_center = detection.center
        track_center = track.current_center
        
        if track_center is None:
            return float('inf')
        
        return np.sqrt(
            (det_center[0] - track_center[0]) ** 2 + 
            (det_center[1] - track_center[1]) ** 2
        )
    
    def calculate_iou(
        self, 
        bbox1: Tuple[float, float, float, float], 
        bbox2: Tuple[float, float, float, float]
    ) -> float:
        """
        2つのバウンディングボックス間のIoUを計算
        
        Args:
            bbox1: バウンディングボックス1 (x1, y1, x2, y2)
            bbox2: バウンディングボックス2 (x1, y1, x2, y2)
            
        Returns:
            IoU値 (0.0-1.0)
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # 交差領域の計算
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # 各バウンディングボックスの面積
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def cleanup_tracks(self) -> None:
        """
        古い追跡や消失した追跡をクリーンアップ
        """
        tracks_to_delete = []
        
        for track_id, track in self.tracks.items():
            track.time_since_update += 1
            track.age += 1
            
            # 長時間更新されていない追跡を削除
            if track.time_since_update > self.max_disappeared:
                track.state = "deleted"
                tracks_to_delete.append(track_id)
            elif track.time_since_update > self.max_disappeared // 2:
                track.state = "lost"
        
        # 削除対象の追跡を除去
        for track_id in tracks_to_delete:
            del self.tracks[track_id]
    
    def get_active_tracks(self) -> List[Track]:
        """
        アクティブな追跡のみを取得
        
        Returns:
            アクティブな追跡のリスト
        """
        return [
            track for track in self.tracks.values() 
            if track.state == "active" and track.hits >= self.min_hits
        ]
    
    def get_all_tracks(self) -> List[Track]:
        """
        全ての追跡を取得
        
        Returns:
            全ての追跡のリスト
        """
        return list(self.tracks.values())
    
    def reset(self) -> None:
        """
        追跡器をリセット
        """
        self.tracks.clear()
        self.next_track_id = 0
        self.frame_count = 0
    
    def __len__(self) -> int:
        return len(self.tracks)
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(tracks={len(self.tracks)}, frame={self.frame_count})"
