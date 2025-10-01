"""RT-DETRv2ベースの物体検出器."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    import torch
    from torch import nn
    from torchvision import transforms as T
    from torchvision.ops import nms
except Exception as exc:  # pragma: no cover - missing optional dependency
    raise ImportError(
        "RT-DETRv2 detector requires PyTorch and torchvision. "
        "Please install them before using this detector."
    ) from exc

from PIL import Image

from .base_detector import BaseDetector, Detection
from ..models.model_manager import ModelManager


class RTDETRDetector(BaseDetector):
    """RT-DETRv2を使用した物体検出器."""

    COCO_CLASSES = [
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]

    _AVAILABLE_MODELS = {
        name: info for name, info in ModelManager.SUPPORTED_MODELS.items() if name.startswith("rtdetrv2_")
    }

    def __init__(
        self,
        model_name: str = "rtdetrv2_r50vd",
        model_path: Optional[str] = None,
        device: str = "auto",
        input_size: Tuple[int, int] = (640, 640),
    ) -> None:
        if model_name not in self._AVAILABLE_MODELS:
            available = ", ".join(sorted(self._AVAILABLE_MODELS.keys()))
            raise ValueError(f"未対応のモデル名です: {model_name}. 使用可能: {available}")

        resolved_device = self._resolve_device(device)
        super().__init__(model_path, resolved_device)

        self.model_name = model_name
        self.input_size = input_size
        self.class_names = self.COCO_CLASSES
        self.device_obj = torch.device(self.device)
        self.transform = T.Compose([T.Resize(self.input_size), T.ToTensor()])
        self.model: Optional[nn.Module] = None

        self._model_info = self._AVAILABLE_MODELS[model_name]
        self._model_manager: Optional[ModelManager] = None

        if self.model_path is None:
            default_dir = Path(__file__).parent.parent / "models" / "weights"
            default_dir.mkdir(parents=True, exist_ok=True)
            self.model_path = str(default_dir / self._model_info["filename"])

        print(f"RT-DETRv2検出器を初期化: {model_name}, デバイス: {self.device}")

    def _resolve_device(self, device: str) -> str:
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"

        if device == "cuda" and not torch.cuda.is_available():
            print("警告: CUDAが利用できません。CPUにフォールバックします。")
            return "cpu"

        if device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            print("警告: MPSが利用できません。CPUにフォールバックします。")
            return "cpu"

        return device

    def _ensure_model_manager(self) -> ModelManager:
        if self._model_manager is None:
            self._model_manager = ModelManager()
        return self._model_manager

    def _ensure_local_weights(self) -> Optional[str]:
        if self.model_path and Path(self.model_path).exists():
            return self.model_path

        manager = self._ensure_model_manager()
        try:
            downloaded = manager.download_model(self.model_name, force_download=False)
            self.model_path = downloaded
            return downloaded
        except Exception as exc:
            print(f"ローカル重みのダウンロードに失敗しました: {exc}")
            return None

    def load_model(self) -> None:
        """RT-DETRv2モデルをロードする."""

        if self.is_loaded:
            return

        print(f"RT-DETRv2モデルを初期化中: {self.model_name}")

        # まずTorch Hubからモデルを取得
        try:
            self.model = torch.hub.load(
                "lyuwenyu/RT-DETR",
                self.model_name,
                pretrained=True,
                trust_repo=True,
            )
            print("Torch HubからRT-DETRv2モデルをロードしました。")
        except Exception as exc:
            raise RuntimeError(f"Torch Hubからモデルをロードできません: {exc}") from exc

        # 可能であればローカル重みで上書き
        local_checkpoint = self._ensure_local_weights()
        if local_checkpoint:
            try:
                checkpoint = torch.load(local_checkpoint, map_location="cpu")
                if isinstance(checkpoint, dict):
                    if "ema" in checkpoint and isinstance(checkpoint["ema"], dict):
                        state = checkpoint["ema"].get("module", checkpoint["ema"])
                    elif "model" in checkpoint:
                        state = checkpoint["model"]
                    else:
                        state = checkpoint

                    try:
                        if hasattr(self.model, "model") and isinstance(self.model.model, nn.Module):
                            self.model.model.load_state_dict(state, strict=False)
                            print(f"ローカルチェックポイントをロードしました: {local_checkpoint}")
                        else:
                            print(
                                "警告: Torch Hubモデルの構造が想定と異なるため、"
                                "ローカルチェックポイントを適用できません。"
                            )
                    except Exception as load_exc:
                        print(
                            "警告: ローカルチェックポイントの読み込みに失敗しました。"
                            f" Torch Hubの重みを使用します: {load_exc}"
                        )
                else:
                    print("警告: 不正なチェックポイント形式です。Torch Hubの重みを使用します。")
            except Exception as exc:
                print(f"ローカルチェックポイントのロードに失敗しました: {exc}")

        self.model = self.model.to(self.device_obj)
        self.model.eval()
        self.is_loaded = True
        print(f"RT-DETRv2モデルのロード完了: デバイス {self.device}")

    def _prepare_inputs(self, image: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device_obj)
        orig_size = (
            torch.tensor([[image.shape[1], image.shape[0]]], dtype=torch.float32)
            .to(self.device_obj)
        )
        return input_tensor, orig_size

    def detect(
        self,
        image: np.ndarray,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.5,
    ) -> List[Detection]:
        if not self.is_loaded:
            self.load_model()

        if self.model is None:
            raise RuntimeError("RT-DETRv2モデルがロードされていません。")

        inputs, orig_size = self._prepare_inputs(image)

        with torch.no_grad():
            outputs = self.model(inputs, orig_size)

        detections = self._convert_outputs_to_detections(
            outputs, confidence_threshold, nms_threshold
        )

        return self.postprocess_detections(detections, confidence_threshold)

    def _convert_outputs_to_detections(
        self,
        outputs,
        confidence_threshold: float,
        nms_threshold: float,
    ) -> List[Detection]:
        if not isinstance(outputs, (list, tuple)) or len(outputs) != 3:
            raise ValueError("RT-DETRv2の出力形式が想定外です。")

        labels, boxes, scores = outputs

        if boxes.ndim != 3:
            raise ValueError("RT-DETRv2の出力に予期しない次元があります。")

        labels = labels[0].to("cpu")
        boxes = boxes[0].to("cpu")
        scores = scores[0].to("cpu")

        mask = scores >= confidence_threshold
        boxes = boxes[mask]
        labels = labels[mask]
        scores = scores[mask]

        if boxes.numel() == 0:
            return []

        if 0.0 < nms_threshold < 1.0:
            keep = nms(boxes, scores, nms_threshold)
            boxes = boxes[keep]
            labels = labels[keep]
            scores = scores[keep]

        detections: List[Detection] = []
        for box, label, score in zip(boxes, labels, scores):
            class_id = int(label.item())
            if class_id >= len(self.class_names):
                continue

            x1, y1, x2, y2 = box.tolist()
            detections.append(
                Detection(
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    confidence=float(score.item()),
                    class_id=class_id,
                    class_name=self.class_names[class_id],
                )
            )

        return detections

    def __str__(self) -> str:  # pragma: no cover - representation only
        return (
            f"RTDETRDetector(model={self.model_name}, "
            f"device={self.device}, loaded={self.is_loaded})"
        )
