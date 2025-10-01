# uvを使用したMOTシステム実行ガイド

## 🚀 クイックスタート

### 1. 初回セットアップ

```bash
# プロジェクトディレクトリに移動
cd /Users/marutyan/CVLAB/AIS2

# 仮想環境をアクティベート
source .venv/bin/activate

# 依存関係をインストール（既に完了済み）
uv pip install -e .
```

### 2. 基本的な実行

```bash
# 利用可能なモデルを確認
python main.py --list-models

# ビデオファイルを処理（結果ビデオを保存）
python main.py --video path/to/your/video.mp4 --save-video

# 出力ディレクトリを指定
python main.py --video video.mp4 --output results/
```

## 📋 詳細な使用方法

### モデル選択

```bash
# 軽量モデル（高速）
python main.py --video video.mp4 --model rtdetrv2_r18vd

# バランス型モデル
python main.py --video video.mp4 --model rtdetrv2_r34vd

# 高精度モデル（デフォルト）
python main.py --video video.mp4 --model rtdetrv2_r50vd
```

### デバイス設定

```bash
# 自動選択（推奨）
python main.py --video video.mp4 --device auto

# CPU強制使用
python main.py --video video.mp4 --device cpu

# GPU使用（CUDA）
python main.py --video video.mp4 --device cuda

# Apple Silicon GPU
python main.py --video video.mp4 --device mps
```

### 検出パラメータ調整

```bash
# 信頼度閾値を調整
python main.py --video video.mp4 --confidence 0.6

# NMS閾値を調整
python main.py --video video.mp4 --nms-threshold 0.3

# 追跡距離を調整
python main.py --video video.mp4 --max-distance 150.0
```

### 処理速度の最適化

```bash
# フレームをスキップして高速処理
python main.py --video video.mp4 --skip-frames 2

# ビデオをリサイズして高速処理
python main.py --video video.mp4 --resize-factor 0.5

# 個別フレームも保存
python main.py --video video.mp4 --save-frames
```

### 設定ファイルの使用

```bash
# デフォルト設定ファイルを作成
python main.py --create-config my_config.yaml

# 設定ファイルを使用
python main.py --video video.mp4 --config my_config.yaml
```

## 🔧 uv特有のコマンド

### 依存関係管理

```bash
# プロジェクトの依存関係を再インストール
uv pip install -e .

# 新しいパッケージを追加
uv pip install package_name

# パッケージを削除
uv pip uninstall package_name

# 依存関係を更新
uv pip install -e . --upgrade
```

### 仮想環境管理

```bash
# 仮想環境を再作成
rm -rf .venv
uv venv
source .venv/bin/activate
uv pip install -e .

# 仮想環境の情報を表示
uv pip list

# 仮想環境を無効化
deactivate
```

## 📁 プロジェクト構造

```
AIS2/
├── main.py                 # メインスクリプト
├── pyproject.toml          # プロジェクト設定
├── README.md              # プロジェクト説明
├── config_example.yaml    # 設定ファイル例
├── UV_USAGE.md           # このファイル
├── .venv/                # uv仮想環境
├── src/mot_system/       # メインパッケージ
│   ├── detection/        # 物体検出
│   ├── tracking/         # 物体追跡
│   ├── utils/           # ユーティリティ
│   └── models/          # モデル管理
├── data/                # データファイル
├── videos/             # 入力ビデオ
└── outputs/            # 出力結果
```

## 🐛 トラブルシューティング

### よくある問題

1. **仮想環境がアクティベートされていない**
   ```bash
   source .venv/bin/activate
   ```

2. **依存関係がインストールされていない**
   ```bash
   uv pip install -e .
   ```

3. **uvが見つからない**
   ```bash
   # uvを再インストール
   curl -LsSf https://astral.sh/uv/install.sh | sh
   source $HOME/.local/bin/env
   ```

4. **CUDA out of memory**
   ```bash
   # CPU使用またはリサイズ
   python main.py --video video.mp4 --device cpu
   python main.py --video video.mp4 --resize-factor 0.5
   ```

### ログとデバッグ

```bash
# 詳細な出力で実行
python main.py --video video.mp4 --save-video --save-frames

# 設定を確認
python main.py --create-config debug_config.yaml
cat debug_config.yaml
```

## 🚀 パフォーマンス最適化

### 高速処理の設定

```bash
# 軽量モデル + フレームスキップ + リサイズ
python main.py --video video.mp4 \
  --model rtdetrv2_r18vd \
  --skip-frames 2 \
  --resize-factor 0.7 \
  --save-video
```

### 高精度処理の設定

```bash
# 高精度モデル + 高信頼度
python main.py --video video.mp4 \
  --model rtdetrv2_r50vd \
  --confidence 0.7 \
  --max-distance 80.0 \
  --save-video
```

## 📊 出力結果

### ビデオ出力
- `outputs/your_video_result.mp4`: 検出・追跡結果付きビデオ
- バウンディングボックス、追跡ID、軌跡が表示

### フレーム出力（オプション）
- `outputs/frames/`: 個別フレーム画像
- 注釈付きフレームが保存

### 統計情報
- 処理フレーム数
- 平均FPS
- 追跡統計（総追跡数、確定追跡数など）

## 🔄 継続的開発

### 新しい機能の追加

```bash
# 機能ブランチを作成
git checkout -b feature/new-feature

# 開発作業
# ... コード変更 ...

# 変更をコミット
git add .
git commit -m "新しい機能を追加"

# ブランチをプッシュ
git push origin feature/new-feature
```

### 依存関係の更新

```bash
# pyproject.tomlを編集後
uv pip install -e .

# 変更をコミット
git add pyproject.toml
git commit -m "依存関係を更新"
```

このガイドに従って、uvを使用したMOTシステムの開発・実行が可能です！
