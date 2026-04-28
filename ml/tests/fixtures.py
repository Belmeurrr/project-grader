"""Synthetic card fixture generation for unit tests.

We generate a canonical 750x1050 card-shaped image with a configurable inner
printed border offset. This lets us test centering measurement against a known
ground truth without depending on any real card images.

Additional helpers below build degraded variants (blurry, glary,
perspective-distorted, framed in a larger background) for the quality-pipeline
and capture-time gating tests.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

CARD_W = 750
CARD_H = 1050


def synth_card(
    left_border_px: int = 40,
    right_border_px: int = 40,
    top_border_px: int = 40,
    bottom_border_px: int = 40,
    image_color: tuple[int, int, int] = (200, 50, 50),
    # Real card whites under typical phone lighting fall in the V≈220-235 range
    # (camera response curves, JPEG quant, ambient cast). Pure white would
    # correctly be flagged as glare by the HSV detector — match reality.
    border_color: tuple[int, int, int] = (228, 228, 230),
    background_color: tuple[int, int, int] = (10, 10, 10),
    width: int = CARD_W,
    height: int = CARD_H,
) -> NDArray[np.uint8]:
    """Make a synthetic dewarped card.

    Layout (left → right):
      [ outer cut edge ] [ white border ] [ printed image (image_color) ]

    The "outer cut edge" is the image boundary itself (background color is
    just used for any padding outside the card; default cards fill the image).
    """
    canvas = np.full((height, width, 3), background_color, dtype=np.uint8)
    canvas[:, :] = border_color

    inner_x0 = left_border_px
    inner_y0 = top_border_px
    inner_x1 = width - right_border_px
    inner_y1 = height - bottom_border_px

    if inner_x0 >= inner_x1 or inner_y0 >= inner_y1:
        raise ValueError("borders too thick — inner image would have nonpositive size")

    canvas[inner_y0:inner_y1, inner_x0:inner_x1] = image_color
    return canvas


def synth_card_with_pattern(
    seed: int,
    n_shapes: int = 12,
    border_color: tuple[int, int, int] = (228, 228, 230),
    width: int = CARD_W,
    height: int = CARD_H,
    border_px: int = 40,
) -> NDArray[np.uint8]:
    """Card with deterministic random rectangles in the inner area.

    Different seeds produce visually distinct cards — useful for hashing /
    identification tests where uniform-color inner regions all hash the
    same (pHash is brightness-invariant by design)."""
    rng = np.random.default_rng(seed)
    canvas = np.full((height, width, 3), border_color, dtype=np.uint8)
    inner_x0, inner_y0 = border_px, border_px
    inner_x1, inner_y1 = width - border_px, height - border_px
    canvas[inner_y0:inner_y1, inner_x0:inner_x1] = (40, 40, 40)

    for _ in range(n_shapes):
        x0 = int(rng.integers(inner_x0, inner_x1 - 30))
        y0 = int(rng.integers(inner_y0, inner_y1 - 30))
        w = int(rng.integers(20, 120))
        h = int(rng.integers(20, 120))
        x1 = min(inner_x1, x0 + w)
        y1 = min(inner_y1, y0 + h)
        color = tuple(int(c) for c in rng.integers(40, 230, size=3))
        canvas[y0:y1, x0:x1] = color
    return canvas


def card_in_scene(
    card: NDArray[np.uint8] | None = None,
    scene_w: int = 1500,
    scene_h: int = 2000,
    bg_color: tuple[int, int, int] = (35, 35, 35),
    fill: float = 0.55,
    perspective_skew_px: int = 0,
) -> NDArray[np.uint8]:
    """Place a card on a contrasting background. Used for quality-pipeline tests.

    `fill` is the target *area* ratio of card to scene (matches what
    QualityThresholds.min_fill_ratio measures). perspective_skew_px tilts the
    right side of the card toward the camera by that many pixels — a non-zero
    value should make perspective_deg go up."""
    if card is None:
        card = synth_card()
    # Solve for target_w such that target_w * target_h / scene_area == fill,
    # where target_h = target_w * (CARD_H / CARD_W).
    aspect_h_over_w = CARD_H / CARD_W
    target_w = int((fill * scene_w * scene_h / aspect_h_over_w) ** 0.5)
    target_h = int(target_w * aspect_h_over_w)
    resized = cv2.resize(card, (target_w, target_h), interpolation=cv2.INTER_AREA)

    canvas = np.full((scene_h, scene_w, 3), bg_color, dtype=np.uint8)
    cx = (scene_w - target_w) // 2
    cy = (scene_h - target_h) // 2

    if perspective_skew_px == 0:
        canvas[cy : cy + target_h, cx : cx + target_w] = resized
        return canvas

    src = np.float32(
        [
            [0, 0],
            [target_w, 0],
            [target_w, target_h],
            [0, target_h],
        ]
    )
    dst = np.float32(
        [
            [cx, cy + perspective_skew_px],
            [cx + target_w, cy],
            [cx + target_w, cy + target_h],
            [cx, cy + target_h - perspective_skew_px],
        ]
    )
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(
        resized, M, (scene_w, scene_h), borderMode=cv2.BORDER_TRANSPARENT
    )
    mask = (warped.sum(axis=2) > 0)[..., None]
    canvas = np.where(mask, warped, canvas).astype(np.uint8)
    return canvas


def blurry(image: NDArray[np.uint8], k: int = 25) -> NDArray[np.uint8]:
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(image, (k, k), 0)


def with_glare(
    image: NDArray[np.uint8],
    fraction: float = 0.05,
    seed: int = 0,
) -> NDArray[np.uint8]:
    """Paint a near-white blob covering `fraction` of the image to simulate glare."""
    rng = np.random.default_rng(seed)
    out = image.copy()
    h, w = out.shape[:2]
    blob_area = int(h * w * fraction)
    # square blob of equivalent area, centered randomly within the frame
    side = max(1, int(blob_area**0.5))
    cx = int(rng.integers(side // 2, max(side // 2 + 1, w - side // 2)))
    cy = int(rng.integers(side // 2, max(side // 2 + 1, h - side // 2)))
    x0, y0 = max(0, cx - side // 2), max(0, cy - side // 2)
    x1, y1 = min(w, x0 + side), min(h, y0 + side)
    out[y0:y1, x0:x1] = 250  # near-white, low saturation → triggers glare detector
    return out


def canonical_clean(width: int = CARD_W, height: int = CARD_H) -> NDArray[np.uint8]:
    """A 750x1050 'canonical' (post-dewarp) card with a clean white border
    and a uniform colored center. Baseline for edge-defect tests."""
    return synth_card(40, 40, 40, 40, width=width, height=height)


def canonical_with_edge_defect(
    side: str = "top",
    length_px: int = 30,
    severity: str = "chip",
    width: int = CARD_W,
    height: int = CARD_H,
) -> NDArray[np.uint8]:
    """Canonical card with a deliberate defect on one perimeter side.

    severity:
      - "chip"      → solid dark bite in the perimeter strip
      - "whitening" → 255-white blob anomalously brighter than the
                      synthetic border (228)
      - "dirt"      → small dark spots scattered along the side
    """
    canvas = canonical_clean(width, height)
    if side == "top":
        x0 = (width - length_px) // 2
        y0 = 0
        x1, y1 = x0 + length_px, 8
    elif side == "bottom":
        x0 = (width - length_px) // 2
        y0 = height - 8
        x1, y1 = x0 + length_px, height
    elif side == "left":
        x0, y0 = 0, (height - length_px) // 2
        x1, y1 = 8, y0 + length_px
    elif side == "right":
        x0 = width - 8
        y0 = (height - length_px) // 2
        x1, y1 = width, y0 + length_px
    else:
        raise ValueError(f"unknown side: {side}")

    if severity == "chip":
        canvas[y0:y1, x0:x1] = (10, 10, 10)
    elif severity == "whitening":
        canvas[y0:y1, x0:x1] = (255, 255, 255)
    elif severity == "dirt":
        rng = np.random.default_rng(0)
        for _ in range(max(1, length_px // 5)):
            xx = int(rng.integers(x0, x1))
            yy = int(rng.integers(y0, y1))
            canvas[yy : yy + 2, xx : xx + 2] = (15, 15, 15)
    else:
        raise ValueError(f"unknown severity: {severity}")
    return canvas


def encode_jpeg(image: NDArray[np.uint8], quality: int = 92) -> bytes:
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return buf.tobytes()
