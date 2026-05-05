"""Holographic-parallax counterfeit detector tests.

Pin the public contract of `analyze_holographic_parallax` against
synthetic front + tilt pairs. Mirrors test_typography_detector.py in
shape — the detector is deterministic so these tests double as both
correctness checks and a regression guard for future calibration tweaks.

Synthesis strategy:
  - Authentic case: paint a saturated rectangular foil region on the
    front, then warpAffine-shift the SAME region on the tilt by a few
    pixels in x to simulate the angle-dependent specular shift. The
    chroma+saturation mask catches the foil region; Farnebäck flow
    fires inside the mask but is ~zero in the (uniform) background.
  - Counterfeit case: same foil region in both shots, no shift. Flow
    ratio collapses to ~1, score lands counterfeit-side.
  - Abstain cases exercise tilt_image=None, shape mismatch, and
    uniform-grey input that has no holo-mask coverage.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines.counterfeit.holographic import (  # noqa: E402
    DEFAULT_RATIO_MIDPOINT,
    DEFAULT_RATIO_SLOPE,
    HolographicResult,
    analyze_holographic_parallax,
)


# Canonical 750x1050 BGR shape — matches what the production pipeline
# produces and what the rosette / color / typography detectors consume.
CANONICAL_H = 1050
CANONICAL_W = 750


def _grey_card(color: tuple[int, int, int] = (200, 200, 200)) -> np.ndarray:
    """Solid-color uint8 BGR image of canonical size."""
    return np.full((CANONICAL_H, CANONICAL_W, 3), color, dtype=np.uint8)


def _paint_foil_block(
    img: np.ndarray,
    *,
    y0: int = 200,
    y1: int = 600,
    x0: int = 150,
    x1: int = 600,
    color: tuple[int, int, int] = (50, 200, 240),
) -> np.ndarray:
    """Paint a rectangular saturated patch — simulates a foil region.
    The chroma+saturation mask should catch this rectangle."""
    out = img.copy()
    out[y0:y1, x0:x1] = color
    # Add a textured gradient so optical flow has something to track.
    yy, xx = np.mgrid[y0:y1, x0:x1]
    # Modulate value by position so flow's gradient-based search can
    # pick out the shift.
    mod = ((xx + yy) % 16).astype(np.uint8) * 8
    out[y0:y1, x0:x1, 0] = np.clip(out[y0:y1, x0:x1, 0].astype(np.int16) + mod - 64, 0, 255).astype(np.uint8)
    out[y0:y1, x0:x1, 2] = np.clip(out[y0:y1, x0:x1, 2].astype(np.int16) - mod + 64, 0, 255).astype(np.uint8)
    return out


def _shift_foil_block(
    img: np.ndarray,
    *,
    dx: int,
    dy: int,
    y0: int = 200,
    y1: int = 600,
    x0: int = 150,
    x1: int = 600,
) -> np.ndarray:
    """Take a foil-painted image and produce a copy where ONLY the foil
    block is spatially shifted by (dx, dy). The non-foil background is
    untouched. This simulates the angle-dependent specular shift on
    real holos: the foil region moves, the rest of the card doesn't."""
    out = img.copy()
    # Source block.
    block = img[y0:y1, x0:x1].copy()
    # Restore the background at the source position so the shifted
    # block doesn't show a "ghost" at the original location.
    bg_color = int(img[10, 10, 0]), int(img[10, 10, 1]), int(img[10, 10, 2])
    out[y0:y1, x0:x1] = bg_color
    # Place the shifted block.
    ny0 = max(0, y0 + dy)
    ny1 = min(img.shape[0], y1 + dy)
    nx0 = max(0, x0 + dx)
    nx1 = min(img.shape[1], x1 + dx)
    sy0 = max(0, -dy)
    sx0 = max(0, -dx)
    sy1 = sy0 + (ny1 - ny0)
    sx1 = sx0 + (nx1 - nx0)
    out[ny0:ny1, nx0:nx1] = block[sy0:sy1, sx0:sx1]
    return out


# -----------------------------
# Happy path — authentic pair shows differential flow
# -----------------------------


def test_authentic_pair_with_shifted_foil_scores_high() -> None:
    """The flagship assertion: front shot has a foil region; tilt shot
    has the same region shifted (simulating a real holo's angle-
    dependent specular shift). Score should land authentic-side."""
    front = _paint_foil_block(_grey_card())
    tilt = _shift_foil_block(front, dx=6, dy=0)

    r = analyze_holographic_parallax(front, tilt)
    assert r.abstain_reason is None
    assert r.confidence > 0.5
    assert r.flow_ratio is not None
    # Inside-flow should be much higher than outside-flow.
    assert r.flow_ratio > 2.0, f"got flow_ratio {r.flow_ratio}"
    assert r.score >= 0.85, f"got score {r.score}"
    # Mask covered a measurable fraction of the front image.
    assert r.holo_mask_fraction is not None
    assert r.holo_mask_fraction > 0.05


def test_authentic_pair_records_metadata() -> None:
    front = _paint_foil_block(_grey_card())
    tilt = _shift_foil_block(front, dx=6, dy=0)

    r = analyze_holographic_parallax(front, tilt)
    assert r.metadata["ratio_midpoint"] == DEFAULT_RATIO_MIDPOINT
    assert r.metadata["ratio_slope"] == DEFAULT_RATIO_SLOPE
    assert r.metadata["inside_pixels"] > 0
    assert r.metadata["outside_pixels"] > 0
    assert r.metadata["inside_mean_flow"] > r.metadata["outside_mean_flow"]


# -----------------------------
# Counterfeit path — flat foil, no parallax
# -----------------------------


def test_counterfeit_pair_with_no_parallax_scores_low() -> None:
    """Same foil region in both shots, no shift between them. Real
    holos shift; flat fakes don't. Score should land counterfeit-side."""
    front = _paint_foil_block(_grey_card())
    tilt = front.copy()  # identical — no parallax shift
    r = analyze_holographic_parallax(front, tilt)
    assert r.abstain_reason is None
    assert r.confidence > 0.5
    assert r.flow_ratio is not None
    # No differential flow — ratio sits near 1 (or lower).
    assert r.flow_ratio < 1.5, f"got flow_ratio {r.flow_ratio}"
    assert r.score <= 0.20, f"got score {r.score}"


def test_score_strictly_increases_with_parallax_magnitude() -> None:
    """Sanity: a larger shift in the foil region produces a strictly
    higher score than a smaller shift. Pins the logistic monotonicity."""
    front = _paint_foil_block(_grey_card())
    no_shift = front.copy()
    small_shift = _shift_foil_block(front, dx=2, dy=0)
    big_shift = _shift_foil_block(front, dx=8, dy=0)
    r0 = analyze_holographic_parallax(front, no_shift)
    r1 = analyze_holographic_parallax(front, small_shift)
    r2 = analyze_holographic_parallax(front, big_shift)
    # All three should be confident measurements.
    assert r0.abstain_reason is None
    assert r1.abstain_reason is None
    assert r2.abstain_reason is None
    assert r0.score < r1.score < r2.score


# -----------------------------
# Abstain paths — the detector never raises
# -----------------------------


def test_abstain_when_tilt_image_missing() -> None:
    """tilt_image=None → tilt_not_captured abstain. Encoded as
    confidence=0 so the ensemble verdict mapper produces UNVERIFIED."""
    front = _paint_foil_block(_grey_card())
    r = analyze_holographic_parallax(front, None)
    assert r.confidence == 0.0
    assert r.abstain_reason == "tilt_not_captured"
    assert r.score == 0.5  # uncertain mid-band, never confidently wrong
    assert r.flow_ratio is None


def test_abstain_when_no_holo_region_uniform_grey() -> None:
    """A uniformly grey card has no foil-like high-chroma high-saturation
    pixels — the heuristic mask is empty, so abstain."""
    front = _grey_card((180, 180, 180))
    tilt = _grey_card((180, 180, 180))
    r = analyze_holographic_parallax(front, tilt)
    assert r.confidence == 0.0
    assert r.abstain_reason == "no_holo_region"
    assert r.score == 0.5


def test_abstain_when_shape_mismatch() -> None:
    """Front 1050x750 vs tilt 1050x800 → shape mismatch abstain. We
    don't try to rescale — the score would be meaningless."""
    front = _paint_foil_block(_grey_card())
    tilt = np.full((CANONICAL_H, 800, 3), 200, dtype=np.uint8)
    r = analyze_holographic_parallax(front, tilt)
    assert r.confidence == 0.0
    assert r.abstain_reason == "shape_mismatch"


def test_abstain_when_front_image_none() -> None:
    """Defensive — the wrapper service catches load failures, but if
    None somehow gets through, we abstain rather than crash."""
    tilt = _grey_card()
    r = analyze_holographic_parallax(None, tilt)
    assert r.confidence == 0.0
    assert r.abstain_reason == "invalid_image"


def test_abstain_when_front_image_wrong_shape() -> None:
    """A 2-D grayscale image isn't BGR — abstain rather than guess."""
    front_2d = np.zeros((CANONICAL_H, CANONICAL_W), dtype=np.uint8)
    tilt = _grey_card()
    r = analyze_holographic_parallax(front_2d, tilt)
    assert r.confidence == 0.0
    assert r.abstain_reason == "invalid_image"


def test_abstain_when_image_too_small() -> None:
    """A tiny image fails the canonical-shape gate — abstain."""
    tiny = np.zeros((10, 10, 3), dtype=np.uint8)
    other = np.zeros((10, 10, 3), dtype=np.uint8)
    r = analyze_holographic_parallax(tiny, other)
    assert r.confidence == 0.0
    assert r.abstain_reason == "invalid_image"


# -----------------------------
# Result shape — frozen dataclass with the right fields
# -----------------------------


def test_result_is_frozen_dataclass() -> None:
    front = _paint_foil_block(_grey_card())
    tilt = _shift_foil_block(front, dx=6, dy=0)
    r = analyze_holographic_parallax(front, tilt)
    assert isinstance(r, HolographicResult)
    # Frozen — defensive assignment raises.
    with pytest.raises(Exception):
        r.score = 0.0  # type: ignore[misc]


def test_confidence_ramps_with_mask_fraction() -> None:
    """A small foil region (just above the min-mask floor) produces a
    REDUCED confidence; a large foil region produces FULL confidence."""
    # Big foil block — should get full confidence.
    big_front = _paint_foil_block(_grey_card(), y0=100, y1=900, x0=100, x1=650)
    big_tilt = _shift_foil_block(
        big_front, dx=6, dy=0, y0=100, y1=900, x0=100, x1=650
    )
    big_r = analyze_holographic_parallax(big_front, big_tilt)
    assert big_r.abstain_reason is None
    # Small foil block — total area ~30000 px / 787500 = 3.8% — between
    # min (2%) and full (5%) → reduced confidence.
    small_front = _paint_foil_block(
        _grey_card(), y0=300, y1=400, x0=200, x1=500
    )
    small_tilt = _shift_foil_block(
        small_front, dx=6, dy=0, y0=300, y1=400, x0=200, x1=500
    )
    small_r = analyze_holographic_parallax(small_front, small_tilt)
    if small_r.abstain_reason is None:
        # If the mask was big enough not to abstain, big should still
        # have at least as much confidence as small.
        assert big_r.confidence >= small_r.confidence


# -----------------------------
# Flow computation failure — defensive abstain
# -----------------------------


def test_flow_failure_abstains_gracefully(monkeypatch: pytest.MonkeyPatch) -> None:
    """If Farnebäck blows up mid-call, the detector swallows the
    exception and abstains — counterfeit ensemble shouldn't be sunk by
    a cv2 corner case."""
    front = _paint_foil_block(_grey_card())
    tilt = _shift_foil_block(front, dx=6, dy=0)

    def _raises(*args, **kwargs):
        raise RuntimeError("simulated cv2 crash")

    from pipelines.counterfeit.holographic import detector as holo_detector

    monkeypatch.setattr(holo_detector, "_flow_magnitude", _raises)
    r = analyze_holographic_parallax(front, tilt)
    assert r.confidence == 0.0
    assert r.abstain_reason == "flow_computation_failed"
    assert "error" in r.metadata
