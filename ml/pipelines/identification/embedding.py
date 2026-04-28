"""Image embedder Protocol + a deterministic CPU fallback.

The production embedder is DINOv2-ViT-B fine-tuned on card crops (see
`ml/training/trainers/identification.py`). For unit tests, CI, and
local dev where torch may not be installed, `SimpleEmbedder` produces
a deterministic 768-d vector from the canonical image using cv2-only
operations: an L2-normalized concatenation of HSV histogram bins +
spatial color block means + a Sobel gradient magnitude histogram.

SimpleEmbedder is good enough to:
  - Distinguish dramatically different cards in tests.
  - Stably round-trip a card image through encode → vector → cosine.
  - Satisfy the ImageEmbedder Protocol so downstream code is identical
    in test and production.

It is NOT good enough for real card identification — production swaps
in DinoV2Embedder via the registry."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

EMBEDDING_DIM = 768


class ImageEmbedder(Protocol):
    @property
    def dim(self) -> int: ...
    def encode(self, image: NDArray[np.uint8]) -> NDArray[np.float32]: ...


# -----------------------------
# SimpleEmbedder (no-torch fallback)
# -----------------------------


@dataclass(frozen=True)
class SimpleEmbedder:
    """Deterministic 768-d image embedding using only cv2 + numpy.

    The vector is a concatenation of:
      - 3 x 32-bin H/S/V histograms      (96 dims)
      - 8x8 spatial mean of HSV channels  (192 dims)
      - 32-bin gradient-magnitude histogram (32 dims)
      - 16x16 spatial mean of grayscale   (256 dims)
      - zero-padding to EMBEDDING_DIM     (192 dims)

    Total: 768. L2-normalized so cosine distance is bounded in [0, 2]."""

    target_size: int = 224

    @property
    def dim(self) -> int:
        return EMBEDDING_DIM

    def encode(self, image: NDArray[np.uint8]) -> NDArray[np.float32]:
        if image.dtype != np.uint8:
            raise ValueError(f"expected uint8 image, got {image.dtype}")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"expected 3-channel BGR image, got shape {image.shape}")

        resized = cv2.resize(image, (self.target_size, self.target_size), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # HSV histograms (96 dims)
        h_hist = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
        s_hist = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()
        v_hist = cv2.calcHist([hsv], [2], None, [32], [0, 256]).flatten()
        hsv_hists = np.concatenate([h_hist, s_hist, v_hist])

        # Spatial 8x8 HSV means (192 dims)
        spatial_hsv = cv2.resize(hsv, (8, 8), interpolation=cv2.INTER_AREA).astype(np.float32)
        spatial_hsv_flat = spatial_hsv.reshape(-1)  # 8*8*3 = 192

        # Gradient magnitude histogram (32 dims)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy).astype(np.float32)
        mag_hist, _ = np.histogram(mag.flatten(), bins=32, range=(0.0, 1024.0))
        mag_hist = mag_hist.astype(np.float32)

        # Spatial 16x16 grayscale (256 dims)
        spatial_gray = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32).reshape(-1)

        parts = [hsv_hists, spatial_hsv_flat, mag_hist, spatial_gray]
        used = sum(p.size for p in parts)
        pad = np.zeros(EMBEDDING_DIM - used, dtype=np.float32) if used < EMBEDDING_DIM else np.empty(0, dtype=np.float32)
        if used > EMBEDDING_DIM:
            raise RuntimeError(f"embedding parts sum to {used} > {EMBEDDING_DIM}")

        vec = np.concatenate([*parts, pad]).astype(np.float32)
        # L2 normalize
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        return vec


# -----------------------------
# Registry
# -----------------------------


def get_embedder() -> ImageEmbedder:
    """Pick an embedder based on env config.

    Selection:
      - GRADER_EMBEDDER=dinov2 with GRADER_DINOV2_WEIGHTS pointing at a
        torch checkpoint: load DinoV2Embedder. Falls back to SimpleEmbedder
        on import or load failure.
      - Otherwise: SimpleEmbedder.
    """
    backend = os.environ.get("GRADER_EMBEDDER", "simple").lower()
    if backend == "dinov2":
        try:
            from pipelines.identification.dinov2 import DinoV2Embedder

            weights = os.environ.get("GRADER_DINOV2_WEIGHTS")
            return DinoV2Embedder(weights_path=weights)
        except Exception:
            return SimpleEmbedder()
    return SimpleEmbedder()
