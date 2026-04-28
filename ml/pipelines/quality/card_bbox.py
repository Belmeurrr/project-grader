"""Heuristic card detection — v1 placeholder for YOLOv11-seg.

Used by the capture-time quality gate to compute fill ratio and approximate
perspective angle. Real card detection / segmentation comes in pipelines/detection
once the YOLO model is trained — at that point this module is replaced as the
upstream of the rest of the pipeline. We keep it for unit tests of the quality
gating logic without requiring a trained model.

Approach: Canny edges → external contours → largest 4-vertex polygon with
plausible card aspect ratio (2.5 : 3.5 ≈ 0.714).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

CARD_ASPECT = 2.5 / 3.5  # short / long
ASPECT_TOLERANCE = 0.18  # relative
MIN_AREA_FRACTION = 0.10  # reject blobs smaller than 10% of frame


@dataclass(frozen=True)
class BBox:
    x: int
    y: int
    w: int
    h: int

    @property
    def area(self) -> int:
        return self.w * self.h

    @property
    def aspect(self) -> float:
        long = max(self.w, self.h)
        short = min(self.w, self.h)
        return short / long if long > 0 else 0.0


@dataclass(frozen=True)
class CardBBox:
    bbox: BBox
    quad: NDArray[np.int32]  # 4x2, ordered TL, TR, BR, BL
    confidence: float


def _order_quad(pts: NDArray[np.float32]) -> NDArray[np.float32]:
    """Order 4 points as TL, TR, BR, BL by sum/diff of coords."""
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).flatten()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.stack([tl, tr, br, bl]).astype(np.float32)


def detect_card_bbox(image: NDArray[np.uint8]) -> CardBBox | None:
    """Return the largest plausible card quad, or None if no card found.

    Heuristic. Replace with a YOLO-seg call once the model is trained."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    h, w = gray.shape[:2]
    frame_area = h * w

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    # Close small gaps so the card outline forms a single contour.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best: CardBBox | None = None
    best_area = 0.0
    for c in contours:
        area = float(cv2.contourArea(c))
        if area / frame_area < MIN_AREA_FRACTION:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        x, y, ww, hh = cv2.boundingRect(approx)
        bbox = BBox(int(x), int(y), int(ww), int(hh))
        if abs(bbox.aspect - CARD_ASPECT) / CARD_ASPECT > ASPECT_TOLERANCE:
            continue
        if area > best_area:
            ordered = _order_quad(approx.reshape(-1, 2).astype(np.float32))
            confidence = min(area / frame_area * 2.0, 1.0)
            best = CardBBox(
                bbox=bbox,
                quad=ordered.astype(np.int32),
                confidence=confidence,
            )
            best_area = area

    return best


def fill_ratio(card: CardBBox, image_shape: tuple[int, int]) -> float:
    """Fraction of the frame occupied by the card bbox."""
    h, w = image_shape[:2]
    return card.bbox.area / float(h * w)


def perspective_deg(card: CardBBox) -> float:
    """Estimate camera off-axis angle in degrees from the card's quadrilateral.

    For a perfect overhead shot, opposite sides of the quad are parallel and
    equal. Off-axis tilt makes opposite sides unequal in length. We use the
    ratio of side lengths to recover the angle: cos(theta) ≈ short / long.
    Returns 0 for a perfect shot, larger for steeper angles.
    """
    q = card.quad.astype(np.float32)
    # quad order: TL, TR, BR, BL
    top_len = float(np.linalg.norm(q[1] - q[0]))
    bottom_len = float(np.linalg.norm(q[2] - q[3]))
    left_len = float(np.linalg.norm(q[3] - q[0]))
    right_len = float(np.linalg.norm(q[2] - q[1]))

    horiz = min(top_len, bottom_len) / max(top_len, bottom_len) if max(top_len, bottom_len) > 0 else 1.0
    vert = min(left_len, right_len) / max(left_len, right_len) if max(left_len, right_len) > 0 else 1.0

    horiz_deg = math.degrees(math.acos(min(max(horiz, -1.0), 1.0)))
    vert_deg = math.degrees(math.acos(min(max(vert, -1.0), 1.0)))
    return max(horiz_deg, vert_deg)
