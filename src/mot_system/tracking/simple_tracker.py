"""
シンプルな物体追跡器

距離ベースの単純な追跡アルゴリズム
ハンガリアンアルゴリズムを使用した最適割り当て
"""

import numpy as np
from typing import List, Tuple, Dict
from scipy.optimize import linear_sum_assignment

from .base_tracker import BaseTracker, Track
from ..detection.base_detector import Detection


class SimpleTracker(BaseTracker):
    """
    距離ベースのシンプルな物体追跡器
    
    特徴:
    - ユークリッド距離による類似度計算
    - ハンガリアンアルゴリズムによる最適割り当て
    - 線形予測による位置推定
    - シンプルで高速な実装
    """
    
    def __init__(
        self,
        max_disappeared: int = 30,
        max_distance: float = 100.0,
        min_hits: int = 3,
        use_iou: bool = False,
        iou_threshold: float = 0.3
    ):
        """
        シンプル追跡器の初期化
        
        Args:
            max_disappeared: 物体が消失したと判断するフレーム数
            max_distance: 同一物体と判断する最大距離
            min_hits: 確定追跡に必要な最小検出回数
            use_iou: IoUを距離計算に使用するかどうか
            iou_threshold: IoUの閾値
        """
        super().__init__(max_disappeared, max_distance, min_hits)
        
        self.use_iou = use_iou
        self.iou_threshold = iou_threshold
        
        print(f"SimpleTracker初期化: max_distance={max_distance}, use_iou={use_iou}")
    
    def update(self, detections: List[Detection]) -> List[Track]:
        """
        新しいフレームの検出結果で追跡を更新
        
        Args:
            detections: 現在のフレームの検出結果
            
        Returns:
            更新された追跡結果のリスト
        """
        self.frame_count += 1
        
        # 追跡の予測を実行
        self.predict_tracks()
        
        # 現在のアクティブな追跡を取得
        active_tracks = [
            track for track in self.tracks.values() 
            if track.state in ["active", "lost"]
        ]
        
        if len(active_tracks) == 0:
            # 追跡が存在しない場合、全ての検出を新しい追跡として開始
            for detection in detections:
                self.create_new_track(detection)
        
        elif len(detections) == 0:
            # 検出が存在しない場合、追跡の状態を更新
            self.cleanup_tracks()
        
        else:
            # 検出と追跡のマッチングを実行
            self._match_detections_to_tracks(detections, active_tracks)
        
        # 古い追跡をクリーンアップ
        self.cleanup_tracks()
        
        return self.get_active_tracks()
    
    def _match_detections_to_tracks(
        self, 
        detections: List[Detection], 
        tracks: List[Track]
    ) -> None:
        """
        検出結果と追跡のマッチングを実行
        
        Args:
            detections: 検出結果のリスト
            tracks: 追跡のリスト
        """
        if len(tracks) == 0 or len(detections) == 0:
            return
        
        # コスト行列を計算
        cost_matrix = self._calculate_cost_matrix(detections, tracks)
        
        # ハンガリアンアルゴリズムで最適割り当て
        detection_indices, track_indices = linear_sum_assignment(cost_matrix)
        
        # マッチング結果を処理
        matched_detections = set()
        matched_tracks = set()
        
        for det_idx, track_idx in zip(detection_indices, track_indices):
            cost = cost_matrix[det_idx, track_idx]
            
            # コストが閾値以下の場合のみマッチングとして採用
            if cost < self._get_cost_threshold():
                detection = detections[det_idx]
                track = tracks[track_idx]
                
                self.update_track(track, detection)
                matched_detections.add(det_idx)
                matched_tracks.add(track_idx)
        
        # マッチしなかった検出から新しい追跡を作成
        for det_idx, detection in enumerate(detections):
            if det_idx not in matched_detections:
                self.create_new_track(detection)
        
        # マッチしなかった追跡の状態を更新
        for track_idx, track in enumerate(tracks):
            if track_idx not in matched_tracks:
                track.time_since_update += 1
                if track.time_since_update > self.max_disappeared // 2:
                    track.state = "lost"
    
    def _calculate_cost_matrix(
        self, 
        detections: List[Detection], 
        tracks: List[Track]
    ) -> np.ndarray:
        """
        検出と追跡間のコスト行列を計算
        
        Args:
            detections: 検出結果のリスト
            tracks: 追跡のリスト
            
        Returns:
            コスト行列 (num_detections x num_tracks)
        """
        cost_matrix = np.zeros((len(detections), len(tracks)))
        
        for det_idx, detection in enumerate(detections):
            for track_idx, track in enumerate(tracks):
                if self.use_iou and track.current_bbox:
                    # IoUベースのコスト計算
                    iou = self.calculate_iou(detection.bbox, track.current_bbox)
                    cost_matrix[det_idx, track_idx] = 1.0 - iou  # IoUが高いほどコストが低い
                else:
                    # 距離ベースのコスト計算
                    distance = self.calculate_distance(detection, track)
                    cost_matrix[det_idx, track_idx] = distance
        
        return cost_matrix
    
    def _get_cost_threshold(self) -> float:
        """
        マッチングのコスト閾値を取得
        
        Returns:
            コスト閾値
        """
        if self.use_iou:
            return 1.0 - self.iou_threshold  # IoUの場合
        else:
            return self.max_distance  # 距離の場合
    
    def calculate_distance(self, detection: Detection, track: Track) -> float:
        """
        検出と追跡間の距離を計算（拡張版）
        
        Args:
            detection: 検出結果
            track: 追跡オブジェクト
            
        Returns:
            重み付き距離
        """
        base_distance = super().calculate_distance(detection, track)
        
        # 追跡の信頼性に基づく重み付け
        reliability_weight = 1.0
        if track.hits < self.min_hits:
            reliability_weight = 2.0  # 新しい追跡はペナルティ
        elif track.time_since_update > 0:
            reliability_weight = 1.5  # 更新されていない追跡もペナルティ
        
        # クラスの一致性チェック
        class_penalty = 0.0
        if track.current_detection and track.current_detection.class_id != detection.class_id:
            class_penalty = 50.0  # クラスが異なる場合は大きなペナルティ
        
        return base_distance * reliability_weight + class_penalty
    
    def get_track_statistics(self) -> Dict[str, int]:
        """
        追跡の統計情報を取得
        
        Returns:
            統計情報の辞書
        """
        stats = {
            "total_tracks": len(self.tracks),
            "active_tracks": len([t for t in self.tracks.values() if t.state == "active"]),
            "lost_tracks": len([t for t in self.tracks.values() if t.state == "lost"]),
            "confirmed_tracks": len([t for t in self.tracks.values() if t.hits >= self.min_hits]),
            "frame_count": self.frame_count
        }
        return stats
    
    def __str__(self) -> str:
        stats = self.get_track_statistics()
        return (
            f"SimpleTracker("
            f"frame={stats['frame_count']}, "
            f"active={stats['active_tracks']}, "
            f"total={stats['total_tracks']}"
            f")"
        )
