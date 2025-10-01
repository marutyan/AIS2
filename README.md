# MOT (Multiple Object Tracking) システム

RT-DETRv2を使用した物体検出とシンプルな追跡アルゴリズムによるMultiple Object Trackingシステムです。

## 特徴

- **RT-DETRv2物体検出**: 高速かつ高精度なTransformerベースの物体検出
- **拡張可能アーキテクチャ**: 将来的にセグメンテーションなどのモデル追加が容易
- **リアルタイム処理**: MP4ビデオファイルの効率的な処理
- **可視化機能**: バウンディングボックス、追跡ID、軌跡の表示
- **設定管理**: YAML/JSON形式の柔軟な設定システム

## システム要件

- Python 3.8以上
- CUDA対応GPU（推奨、CPUでも動作可能）
- uv（Python仮想環境管理）

## インストール

### 1. リポジトリのクローン

```bash
git clone <repository-url>
cd AIS2
```

### 2. 仮想環境の作成と依存関係のインストール

```bash
# uvを使用した仮想環境の作成
uv venv

# 仮想環境のアクティベート
source .venv/bin/activate  # Linux/macOS
# または
.venv\\Scripts\\activate  # Windows

# 依存関係のインストール
uv pip install -e .
```

## 使用方法

### 基本的な使用方法

```bash
# ビデオファイルを処理
python main.py --video path/to/your/video.mp4

# 結果ビデオを保存
python main.py --video path/to/your/video.mp4 --save-video

# 出力ディレクトリを指定
python main.py --video path/to/your/video.mp4 --output results/
```

### 高度な設定

```bash
# 特定のモデルを使用
python main.py --video video.mp4 --model rtdetrv2_r18vd

# 信頼度閾値を調整
python main.py --video video.mp4 --confidence 0.6

# GPU使用を強制
python main.py --video video.mp4 --device cuda

# フレームをスキップして高速処理
python main.py --video video.mp4 --skip-frames 2
```

### 設定ファイルを使用

```bash
# デフォルト設定ファイルを作成
python main.py --create-config config.yaml

# 設定ファイルを使用
python main.py --video video.mp4 --config config.yaml
```

### 利用可能なモデルの確認

```bash
python main.py --list-models
```

## プロジェクト構造

```
AIS2/
├── main.py                 # メインスクリプト
├── pyproject.toml          # プロジェクト設定
├── README.md              # このファイル
├── .gitignore             # Git除外設定
├── src/
│   └── mot_system/        # メインパッケージ
│       ├── __init__.py
│       ├── detection/     # 物体検出モジュール
│       │   ├── base_detector.py      # 検出器の基底クラス
│       │   └── rt_detr_detector.py   # RT-DETRv2実装
│       ├── tracking/      # 物体追跡モジュール
│       │   ├── base_tracker.py       # 追跡器の基底クラス
│       │   └── simple_tracker.py     # シンプル追跡器
│       ├── utils/         # ユーティリティ
│       │   ├── video_processor.py    # ビデオ処理
│       │   ├── visualization.py      # 可視化
│       │   └── config.py            # 設定管理
│       └── models/        # モデル管理
│           └── model_manager.py      # モデルダウンロード・管理
├── data/                  # データファイル（.gitignoreに含まれる）
├── videos/               # 入力ビデオ（.gitignoreに含まれる）
└── outputs/              # 出力結果（.gitignoreに含まれる）
```

## アーキテクチャ

### 物体検出 (Detection)

- **BaseDetector**: 全ての検出器の基底クラス
- **RTDETRDetector**: RT-DETRv2を使用した検出器実装
- 将来的に他の検出モデル（YOLO、Faster R-CNN等）の追加が容易

### 物体追跡 (Tracking)

- **BaseTracker**: 全ての追跡器の基底クラス
- **SimpleTracker**: 距離ベースのシンプルな追跡アルゴリズム
- ハンガリアンアルゴリズムによる最適割り当て
- 将来的にDeepSORTやByteTrackなどの追加が可能

### ユーティリティ

- **VideoProcessor**: MP4ファイルの読み込み・処理・保存
- **Visualizer**: 検出・追跡結果の可視化
- **Config**: 設定ファイルの管理
- **ModelManager**: モデルファイルの自動ダウンロード・管理

## 設定項目

### 物体検出設定

```yaml
detection:
  model_name: "rtdetrv2_r50vd"    # 使用するモデル
  device: "auto"                  # 実行デバイス
  confidence_threshold: 0.5       # 信頼度閾値
  nms_threshold: 0.4             # NMS閾値
  input_size: [640, 640]         # 入力サイズ
```

### 物体追跡設定

```yaml
tracking:
  tracker_type: "simple"          # 追跡アルゴリズム
  max_disappeared: 30             # 消失判定フレーム数
  max_distance: 100.0            # 最大追跡距離
  min_hits: 3                    # 確定追跡に必要な検出回数
  use_iou: false                 # IoUベース追跡の使用
  iou_threshold: 0.3             # IoU閾値
```

### 可視化設定

```yaml
visualization:
  show_confidence: true           # 信頼度表示
  show_class_names: true         # クラス名表示
  show_track_ids: true           # 追跡ID表示
  show_trajectories: true        # 軌跡表示
  trajectory_length: 30          # 軌跡の長さ
  font_scale: 0.6               # フォントサイズ
  line_thickness: 2             # 線の太さ
```

## サポートされているモデル

| モデル名 | バックボーン | サイズ | 説明 |
|---------|-------------|--------|------|
| rtdetrv2_r18vd | ResNet-18 | ~85MB | 軽量・高速 |
| rtdetrv2_r34vd | ResNet-34 | ~120MB | バランス型 |
| rtdetrv2_r50vd | ResNet-50 | ~150MB | 高精度（デフォルト） |

## 出力形式

### ビデオ出力

- 検出されたオブジェクトのバウンディングボックス
- 追跡ID付きの軌跡表示
- リアルタイム統計情報

### フレーム出力

- 個別フレーム画像（オプション）
- 注釈付きフレーム保存

## 拡張性

### 新しい検出モデルの追加

1. `src/mot_system/detection/`に新しい検出器クラスを作成
2. `BaseDetector`を継承
3. `detect()`メソッドを実装

### 新しい追跡アルゴリズムの追加

1. `src/mot_system/tracking/`に新しい追跡器クラスを作成
2. `BaseTracker`を継承
3. `update()`メソッドを実装

## トラブルシューティング

### よくある問題

1. **CUDA out of memory**: `--device cpu`を使用するか、`--resize-factor`を小さくしてください
2. **モデルダウンロードエラー**: インターネット接続を確認し、再実行してください
3. **ビデオが開けない**: OpenCVがサポートする形式か確認してください

### ログとデバッグ

```bash
# 詳細な出力で実行
python main.py --video video.mp4 --save-video --save-frames

# 設定を確認
python main.py --create-config debug_config.yaml
```

## 開発者向け情報

### テスト実行

```bash
# 単体テストの実行
python -m pytest tests/

# 型チェック
mypy src/

# コードフォーマット
black src/
```

### 開発環境セットアップ

```bash
# 開発用依存関係のインストール
uv pip install -e ".[dev]"

# pre-commitフックの設定
pre-commit install
```

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 参考文献

- [RT-DETR: DETRs Beat YOLOs on Real-time Object Detection](https://arxiv.org/abs/2304.08069)
- [RT-DETRv2: Improved Baseline with Bag-of-Freebies](https://arxiv.org/abs/2407.17140)
- [公式RT-DETRリポジトリ](https://github.com/lyuwenyu/RT-DETR)

## 貢献

プルリクエストやイシューの報告を歓迎します。開発に参加される場合は、以下の手順に従ってください：

1. リポジトリをフォーク
2. 機能ブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを開く
