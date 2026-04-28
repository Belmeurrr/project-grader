"""Stage 2 dewarp tests.

Verifies that homography to canonical (750x1050) recovers the inner card
geometry, and that the quad-irregularity score correctly rejects bent /
off-axis quads."""

from __future__ import annotations

import numpy as np
import pytest

from pipelines.detection.dewarp import (
    CANONICAL_HEIGHT,
    CANONICAL_WIDTH,
    dewarp_to_canonical,
    quad_irregularity,
)
from tests.fixtures import card_in_scene


def _scene_with_known_quad():
    """Scene + the exact pixel coordinates of the placed card's corners."""
    img = card_in_scene(fill=0.55)
    # card_in_scene centers the card in the scene; recompute its placement to
    # get ground-truth corner coords.
    h, w = img.shape[:2]
    aspect_h_over_w = 1050 / 750
    target_w = int((0.55 * w * h / aspect_h_over_w) ** 0.5)
    target_h = int(target_w * aspect_h_over_w)
    cx = (w - target_w) // 2
    cy = (h - target_h) // 2
    quad = np.float32(
        [
            [cx, cy],
            [cx + target_w, cy],
            [cx + target_w, cy + target_h],
            [cx, cy + target_h],
        ]
    )
    return img, quad


# -----------------------------
# quad_irregularity
# -----------------------------


def test_irregularity_zero_for_perfect_rect() -> None:
    quad = np.float32([[0, 0], [100, 0], [100, 140], [0, 140]])
    assert quad_irregularity(quad) == pytest.approx(0.0, abs=1e-6)


def test_irregularity_grows_with_skew() -> None:
    flat = np.float32([[0, 0], [100, 0], [100, 140], [0, 140]])
    skewed = np.float32([[10, 0], [100, 0], [100, 140], [0, 140]])  # TL pushed right
    assert quad_irregularity(skewed) > quad_irregularity(flat)


def test_irregularity_high_for_collapsed_quad() -> None:
    nearly_collapsed = np.float32([[0, 0], [100, 0], [50, 1], [50, 1]])
    assert quad_irregularity(nearly_collapsed) >= 0.5


def test_irregularity_clipped_to_one() -> None:
    degenerate = np.float32([[0, 0], [100, 0], [0, 0], [0, 100]])
    assert 0.0 <= quad_irregularity(degenerate) <= 1.0


def test_irregularity_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match=r"\(4, 2\)"):
        quad_irregularity(np.zeros((3, 2), dtype=np.float32))


# -----------------------------
# dewarp_to_canonical
# -----------------------------


def test_dewarp_produces_canonical_dimensions() -> None:
    img, quad = _scene_with_known_quad()
    result = dewarp_to_canonical(img, quad)
    assert result.canonical.shape == (CANONICAL_HEIGHT, CANONICAL_WIDTH, 3)
    assert result.canonical.dtype == np.uint8


def test_dewarp_homography_is_3x3() -> None:
    img, quad = _scene_with_known_quad()
    result = dewarp_to_canonical(img, quad)
    assert result.homography.shape == (3, 3)


def test_dewarp_recovers_inner_image() -> None:
    """The canonical view should be dominated by the card's inner color
    (synthetic fixture has a red center on a near-white border). After
    dewarp, the center pixel must be close to the original red."""
    img, quad = _scene_with_known_quad()
    result = dewarp_to_canonical(img, quad)
    cy, cx = CANONICAL_HEIGHT // 2, CANONICAL_WIDTH // 2
    # synth_card default image_color = (200, 50, 50) in BGR
    assert result.canonical[cy, cx, 0] >= 150  # blue channel high (red in RGB)
    assert result.canonical[cy, cx, 1] < 100   # green low
    assert result.canonical[cy, cx, 2] < 100   # red channel (BGR) low


def test_dewarp_irregularity_low_for_overhead_shot() -> None:
    img, quad = _scene_with_known_quad()
    result = dewarp_to_canonical(img, quad)
    assert result.irregularity < 0.05


def test_dewarp_rejects_non_uint8() -> None:
    img, quad = _scene_with_known_quad()
    with pytest.raises(ValueError, match="uint8"):
        dewarp_to_canonical(img.astype(np.float32), quad)


def test_dewarp_rejects_grayscale() -> None:
    img, quad = _scene_with_known_quad()
    gray = img[:, :, 0]
    with pytest.raises(ValueError, match="3-channel"):
        dewarp_to_canonical(gray, quad)


def test_dewarp_custom_dimensions() -> None:
    img, quad = _scene_with_known_quad()
    result = dewarp_to_canonical(img, quad, out_w=400, out_h=560)
    assert result.canonical.shape == (560, 400, 3)
