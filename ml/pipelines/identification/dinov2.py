"""DINOv2-ViT-B image embedder for production card identification.

Loads a fine-tuned DINOv2-ViT-B model (or the public checkpoint as a baseline)
via torch + transformers. The CLS-token embedding (dim 768) is used directly
as the card embedding, L2-normalized so cosine distance can be computed via
inner product.

Construction is cheap: the model is loaded lazily on first `encode()`. This
keeps Celery worker cold-start fast.

Training of the fine-tuned weights lives in
`ml/training/trainers/identification.py` (not yet implemented; comes with
the data ingestion task)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from pipelines.identification.embedding import EMBEDDING_DIM


@dataclass
class DinoV2Embedder:
    """Lazy DINOv2 wrapper. Defaults to the public facebook/dinov2-base
    checkpoint when no fine-tuned weights are configured, so identification
    works out of the box (with reduced accuracy) before fine-tuning lands."""

    weights_path: str | None = None
    base_model: str = "facebook/dinov2-base"
    device: str = "cuda:0"
    target_size: int = 224
    _model: Any = field(default=None, init=False, repr=False)
    _processor: Any = field(default=None, init=False, repr=False)
    _torch: Any = field(default=None, init=False, repr=False)

    @property
    def dim(self) -> int:
        return EMBEDDING_DIM

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as e:
            raise RuntimeError(
                "torch + transformers required for DinoV2Embedder; "
                "install via `uv sync --extra training` from ml/, "
                "or set GRADER_EMBEDDER=simple"
            ) from e

        self._torch = torch
        try:
            self._processor = AutoImageProcessor.from_pretrained(self.base_model)
            model = AutoModel.from_pretrained(self.base_model)
            if self.weights_path:
                state = torch.load(self.weights_path, map_location="cpu")
                state = state.get("model", state) if isinstance(state, dict) else state
                model.load_state_dict(state, strict=False)
            device = self.device if torch.cuda.is_available() else "cpu"
            model = model.to(device).eval()
            self._model = model
            self.device = device
        except Exception as e:
            raise RuntimeError(f"failed to load DINOv2 model: {e}") from e

    def encode(self, image: NDArray[np.uint8]) -> NDArray[np.float32]:
        if image.dtype != np.uint8:
            raise ValueError(f"expected uint8 image, got {image.dtype}")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"expected 3-channel BGR image, got shape {image.shape}")

        self._ensure_loaded()
        torch = self._torch

        # Convert BGR (cv2) → RGB (transformers expects RGB).
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        inputs = self._processor(images=rgb, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
        # Use CLS token (first position of last_hidden_state).
        cls = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy().astype(np.float32)
        if cls.shape[0] != EMBEDDING_DIM:
            raise RuntimeError(
                f"unexpected embedding dim {cls.shape[0]}, expected {EMBEDDING_DIM}"
            )
        norm = float(np.linalg.norm(cls))
        if norm > 0:
            cls = cls / norm
        return cls
