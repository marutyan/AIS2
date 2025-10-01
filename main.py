"""
MOT (Multiple Object Tracking) システムのメインスクリプト

RT-DETRv2を使用した物体検出とシンプルな追跡アルゴリズムによる
Multiple Object Trackingシステムのエントリーポイント

使用例:
    python main.py --video path/to/video.mp4
    python main.py --video path/to/video.mp4 --config config.yaml
    python main.py --video path/to/video.mp4 --output outputs/ --save-video
"""

import argparse
import sys
import time
from pathlib import Path

# プロジェクトのsrcディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.mot_system.detection.rt_detr_detector import RTDETRDetector
from src.mot_system.tracking.simple_tracker import SimpleTracker
from src.mot_system.utils.video_processor import VideoProcessor
from src.mot_system.utils.visualization import Visualizer
from src.mot_system.utils.config import Config
from src.mot_system.models.model_manager import ModelManager


def parse_arguments():
    """
    コマンドライン引数の解析
    
    Returns:
        解析済み引数のNamespace
    """
    parser = argparse.ArgumentParser(
        description="MOT (Multiple Object Tracking) システム",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python main.py --video video.mp4
  python main.py --video video.mp4 --config config.yaml --output results/
  python main.py --video video.mp4 --model rtdetrv2_r18vd --confidence 0.6
  python main.py --list-models  # 利用可能なモデルを表示
        """
    )
    
    # 必須引数
    parser.add_argument(
        '--video', '-v',
        type=str,
        help='入力ビデオファイルのパス (.mp4, .avi, .mov など)'
    )
    
    # オプション引数
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='設定ファイルのパス (.yaml または .json)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='outputs',
        help='出力ディレクトリ (デフォルト: outputs)'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='rtdetrv2_r50vd',
        choices=['rtdetrv2_r18vd', 'rtdetrv2_r34vd', 'rtdetrv2_r50vd'],
        help='使用するモデル (デフォルト: rtdetrv2_r50vd)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda', 'mps'],
        help='実行デバイス (デフォルト: auto)'
    )
    
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.5,
        help='検出の信頼度閾値 (デフォルト: 0.5)'
    )
    
    parser.add_argument(
        '--nms-threshold',
        type=float,
        default=0.4,
        help='NMSの閾値 (デフォルト: 0.4)'
    )
    
    parser.add_argument(
        '--max-distance',
        type=float,
        default=100.0,
        help='追跡の最大距離 (デフォルト: 100.0)'
    )
    
    parser.add_argument(
        '--resize-factor',
        type=float,
        default=1.0,
        help='ビデオのリサイズ倍率 (デフォルト: 1.0)'
    )
    
    parser.add_argument(
        '--save-video',
        action='store_true',
        help='結果ビデオを保存'
    )
    
    parser.add_argument(
        '--save-frames',
        action='store_true',
        help='個別フレームを保存'
    )
    
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='可視化を無効化'
    )
    
    parser.add_argument(
        '--skip-frames',
        type=int,
        default=1,
        help='処理するフレーム間隔 (デフォルト: 1)'
    )
    
    # 情報表示用
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='利用可能なモデルを表示'
    )
    
    parser.add_argument(
        '--create-config',
        type=str,
        help='デフォルト設定ファイルを作成'
    )
    
    return parser.parse_args()


def list_available_models():
    """利用可能なモデルを表示"""
    print("=== 利用可能なモデル ===")
    
    model_manager = ModelManager()
    
    print("\\n[サポートされているモデル]")
    for model_name in model_manager.list_available_models():
        info = model_manager.get_model_info(model_name)
        status = "✓ ダウンロード済み" if info['exists'] else "- 未ダウンロード"
        print(f"  {model_name}: {info['description']} ({info['size_mb']}MB) {status}")
    
    print("\\n[ダウンロード済みモデル]")
    downloaded = model_manager.list_downloaded_models()
    if downloaded:
        for model_name, path, size in downloaded:
            size_mb = size / (1024 * 1024)
            print(f"  {model_name}: {path} ({size_mb:.1f}MB)")
    else:
        print("  なし")
    
    storage = model_manager.get_storage_usage()
    print(f"\\n[ストレージ使用量]")
    print(f"  合計: {storage['total_size_mb']:.1f}MB ({storage['models_count']}モデル)")


def create_default_config(config_path: str):
    """デフォルト設定ファイルを作成"""
    config = Config()
    config.create_default_config(config_path)
    print(f"デフォルト設定ファイルを作成しました: {config_path}")


def main():
    """メイン処理"""
    args = parse_arguments()
    
    # 情報表示のみの場合
    if args.list_models:
        list_available_models()
        return
    
    if args.create_config:
        create_default_config(args.create_config)
        return
    
    # ビデオファイルが指定されていない場合はエラー
    if not args.video:
        print("エラー: ビデオファイルが指定されていません")
        print("使用方法: python main.py --video path/to/video.mp4")
        return
    
    # ビデオファイルの存在確認
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"エラー: ビデオファイルが見つかりません: {args.video}")
        return
    
    print("=== MOT システム開始 ===")
    print(f"入力ビデオ: {video_path}")
    print(f"出力ディレクトリ: {args.output}")
    print(f"使用モデル: {args.model}")
    print(f"実行デバイス: {args.device}")
    
    try:
        # 設定の読み込み
        config = Config(args.config) if args.config else Config()
        
        # コマンドライン引数で設定をオーバーライド
        config.config.detection.model_name = args.model
        config.config.detection.device = args.device
        config.config.detection.confidence_threshold = args.confidence
        config.config.detection.nms_threshold = args.nms_threshold
        config.config.tracking.max_distance = args.max_distance
        config.config.video.resize_factor = args.resize_factor
        config.config.video.process_every_n_frames = args.skip_frames
        config.config.video.save_video = args.save_video
        config.config.video.save_frames = args.save_frames
        config.config.output.output_dir = args.output
        
        # 設定の妥当性チェック
        if not config.validate_config():
            print("設定エラーが発生しました。処理を中断します。")
            return
        
        # モデル管理の初期化
        print("\\n=== モデル準備 ===")
        model_manager = ModelManager()
        model_path = None
        try:
            model_path = model_manager.download_model(
                config.config.detection.model_name,
                force_download=False,
                show_progress=True,
            )
            print(f"モデル重みを準備しました: {model_path}")
        except Exception as exc:
            print(f"警告: モデル重みの事前ダウンロードに失敗しました: {exc}")

        # 物体検出器の初期化
        print("物体検出器を初期化中...")
        detector = RTDETRDetector(
            model_name=config.config.detection.model_name,
            model_path=model_path,
            device=config.config.detection.device,
            input_size=tuple(config.config.detection.input_size)
        )
        
        # 物体追跡器の初期化
        print("物体追跡器を初期化中...")
        tracker = SimpleTracker(
            max_disappeared=config.config.tracking.max_disappeared,
            max_distance=config.config.tracking.max_distance,
            min_hits=config.config.tracking.min_hits,
            use_iou=config.config.tracking.use_iou,
            iou_threshold=config.config.tracking.iou_threshold
        )
        
        # 可視化器の初期化
        visualizer = None
        if not args.no_display:
            print("可視化器を初期化中...")
            visualizer = Visualizer(
                show_confidence=config.config.visualization.show_confidence,
                show_class_names=config.config.visualization.show_class_names,
                show_track_ids=config.config.visualization.show_track_ids,
                show_trajectories=config.config.visualization.show_trajectories,
                trajectory_length=config.config.visualization.trajectory_length,
                font_scale=config.config.visualization.font_scale,
                line_thickness=config.config.visualization.line_thickness
            )
        
        # ビデオプロセッサーの初期化
        print("\\n=== ビデオ処理開始 ===")
        with VideoProcessor(
            video_path=str(video_path),
            output_dir=config.config.output.output_dir,
            resize_factor=config.config.video.resize_factor
        ) as video_processor:
            
            # ビデオ情報の表示
            video_info = video_processor.get_video_info()
            print(f"解像度: {video_info['width']}x{video_info['height']} -> "
                  f"{video_info['output_width']}x{video_info['output_height']}")
            print(f"FPS: {video_info['fps']:.2f}")
            print(f"フレーム数: {video_info['frame_count']}")
            print(f"時間: {video_info['duration_seconds']:.2f}秒")
            
            # 処理開始時間の記録
            start_time = time.time()
            
            # ビデオ処理の実行
            results = video_processor.process_video(
                detector=detector,
                tracker=tracker,
                visualizer=visualizer,
                save_video=config.config.video.save_video,
                save_frames=config.config.video.save_frames,
                show_progress=config.config.video.show_progress,
                process_every_n_frames=config.config.video.process_every_n_frames
            )
            
            # 処理時間の計算
            processing_time = time.time() - start_time
            processed_frames = len(results)
            avg_fps = processed_frames / processing_time if processing_time > 0 else 0
            
            print(f"\\n=== 処理完了 ===")
            print(f"処理フレーム数: {processed_frames}")
            print(f"処理時間: {processing_time:.2f}秒")
            print(f"平均FPS: {avg_fps:.2f}")
            
            # 追跡統計の表示
            if hasattr(tracker, 'get_track_statistics'):
                stats = tracker.get_track_statistics()
                print(f"\\n=== 追跡統計 ===")
                print(f"総追跡数: {stats['total_tracks']}")
                print(f"確定追跡数: {stats['confirmed_tracks']}")
                print(f"最終アクティブ追跡数: {stats['active_tracks']}")
            
            print(f"\\n結果は以下に保存されました: {config.config.output.output_dir}")
    
    except KeyboardInterrupt:
        print("\\n\\n処理が中断されました。")
    
    except Exception as e:
        print(f"\\nエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
    
    print("\\n=== MOT システム終了 ===")


if __name__ == "__main__":
    main()
