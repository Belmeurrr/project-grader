"""Stage 2 — perspective correction.

Given the original camera image and a quad (4 ordered corner points), produce
a canonical 750x1050 orthographic view of the card via 4-point homography.

The output dimensions match the standard sport/TCG card aspect ratio
(2.5 x 3.5 inches → 5:7 → 750x1050 px at ~300 DPI). All downstream stages
(centering, corner / edge / surface models) consume this canonical image,
which means the same model weights work regardless of capture distance,
phone, or sensor resolution.

`quad_irregularity` is a quality score on the *input* quad: how
non-rectangular it is, normalized to roughly [0, 1]. Used by the grading
pipeline to reject bent / off-axis cards before dewarping makes them look
deceptively flat in the canonical view.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

CANONICAL_WIDTH = 750
CANONICAL_HEIGHT = 1050


@dataclass(frozen=True)
class DewarpResult:
    canonical: NDArray[np.uint8]   # uint8 BGR, shape (CANONICAL_HEIGHT, CANONICAL_WIDTH, 3)
    homography: NDArray[np.float64]  # 3x3
    irregularity: float            # 0 = flat overhead shot, larger = more skewed


def _validate_quad(quad: NDArray[np.floating]) -> NDArray[np.float32]:
    if quad.shape != (4, 2):
        raise ValueError(f"quad must be shape (4, 2), got {quad.shape}")
    return quad.astype(np.float32, copy=False)


def quad_irregularity(quad: NDArray[np.floating]) -> float:
    """Score how non-rectangular the input quad is.

    For a flat overhead shot of a card, opposite sides are equal in length
    and adjacent sides are perpendicular. We compute two factors:

      side_skew  — max relative difference between opposite-side lengths
      angle_skew — max deviation of any corner from a right angle, in radians

    Returns max(side_skew, angle_skew) clipped to [0, 1]. 0 means flat and
    perpendicular; 1 means severely degenerate (quad collapsed or extremely
    angled).
    """
    q = _validate_quad(quad)
    tl, tr, br, bl = q
    top = float(np.linalg.norm(tr - tl))
    bottom = float(np.linalg.norm(br - bl))
    left = float(np.linalg.norm(bl - tl))
    right = float(np.linalg.norm(br - tr))

    def _ratio_skew(a: float, b: float) -> float:
        if max(a, b) <= 0:
            return 1.0
        return abs(a - b) / max(a, b)

    side_skew = max(_ratio_skew(top, bottom), _ratio_skew(left, right))

    def _corner_angle(p_prev: NDArray[np.float32], p: NDArray[np.float32], p_next: NDArray[np.float32]) -> float:
        v1 = p_prev - p
        v2 = p_next - p
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            return float("nan")
        cosang = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        return float(np.arccos(cosang))

    angles = [
        _corner_angle(bl, tl, tr),
        _corner_angle(tl, tr, br),
        _corner_angle(tr, br, bl),
        _corner_angle(br, bl, tl),
    ]
    angle_skew = max(abs(a - np.pi / 2) for a in angles if not np.isnan(a)) / (np.pi / 2)
    return float(min(max(side_skew, angle_skew), 1.0))


def dewarp_to_canonical(
    image: NDArray[np.uint8],
    quad: NDArray[np.floating],
    out_w: int = CANONICAL_WIDTH,
    out_h: int = CANONICAL_HEIGHT,
) -> DewarpResult:
    """Apply 4-point homography to produce a canonical orthographic view.

    Args:
        image: HxWx3 uint8 BGR image.
        quad: 4x2 float corner points in the order TL, TR, BR, BL.
        out_w, out_h: canonical output dimensions (defaults match standard
            card aspect 750x1050).

    Returns:
        DewarpResult with the warped image, the 3x3 homography, and the
        input-quad irregularity score (informational, computed pre-warp).
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected 3-channel image, got shape {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"expected uint8, got {image.dtype}")

    src = _validate_quad(quad)
    dst = np.array(
        [
            [0.0, 0.0],
            [out_w - 1, 0.0],
            [out_w - 1, out_h - 1],
            [0.0, out_h - 1],
        ],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, homography, (out_w, out_h))
    return DewarpResult(
        canonical=warped,
        homography=homography.astype(np.float64),
        irregularity=quad_irregularity(quad),
    )
