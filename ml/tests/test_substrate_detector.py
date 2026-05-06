"""Substrate / paper-fluorescence counterfeit detector tests.

Pin the public contract of `analyze_substrate` against synthetic image
inputs. Mirrors test_typography_detector.py / test_holographic_detector.py
— the detector is deterministic and threshold-based, so these tests
double as both correctness checks and a regression guard for future
calibration tweaks.

Synthetic strategy:
    The detector reads CIELAB b* on the canonical's outer border, so
    we build solid-color uint8 BGR images and only need to control
    the NEUTRAL-vs-blueshift difference between the front and flash
    versions of the border. We do this by constructing the flash
    image with a slight blue tint on the border (lower R/G, higher B)
    so the cv2 LAB conversion produces a measurable negative b* shift.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines.counterfeit.substrate import (  # noqa: E402
    BORDER_WIDTH_FRAC,
    DELTA_B_MIDPOINT,
    DELTA_B_SLOPE,
    MAX_CONFIDENCE,
    MIN_BORDER_PIXELS,
    SubstrateResult,
    analyze_substrate,
)
from pipelines.counterfeit.substrate import detector as substrate_detector  # noqa: E402


# Canonical 750x1050 BGR shape — matches what the production pipeline
# produces and what the rosette / color / typography detectors consume.
CANONICAL_H = 1050
CANONICAL_W = 750


def _solid_canvas(color_bgr: tuple[int, int, int]) -> np.ndarray:
    """Solid-color uint8 BGR image of canonical size."""
    return np.full((CANONICAL_H, CANONICAL_W, 3), color_bgr, dtype=np.uint8)


def _border_b_shifted_canvas(
    base_bgr: tuple[int, int, int],
    *,
    border_b_offset: int,
    border_g_offset: int = 0,
    border_r_offset: int = 0,
) -> np.ndarray:
    """Build a canvas with the body at `base_bgr` and the outer border
    tinted by adding the given per-channel offsets. Used to simulate
    a flash image whose BORDER (substrate) shifts color independently
    from the body."""
    img = _solid_canvas(base_bgr)
    # Apply the offset in the same shape as the detector's border mask:
    # outer band of width BORDER_WIDTH_FRAC, corner squares excluded.
    band = max(1, int(round(BORDER_WIDTH_FRAC * min(CANONICAL_H, CANONICAL_W))))
    corner = max(1, int(round(2 * BORDER_WIDTH_FRAC * min(CANONICAL_H, CANONICAL_W))))

    mask = np.zeros((CANONICAL_H, CANONICAL_W), dtype=bool)
    mask[:band, :] = True
    mask[CANONICAL_H - band:, :] = True
    mask[:, :band] = True
    mask[:, CANONICAL_W - band:] = True
    # Exclude corner squares (matches detector's mask)
    mask[:corner, :corner] = False
    mask[:corner, CANONICAL_W - corner:] = False
    mask[CANONICAL_H - corner:, :corner] = False
    mask[CANONICAL_H - corner:, CANONICAL_W - corner:] = False

    work = img.astype(np.int16)
    work[mask, 0] = np.clip(work[mask, 0] + border_b_offset, 0, 255)
    work[mask, 1] = np.clip(work[mask, 1] + border_g_offset, 0, 255)
    work[mask, 2] = np.clip(work[mask, 2] + border_r_offset, 0, 255)
    return work.astype(np.uint8)


# ---------------------------------------------------------------------------
# Authentic — front and flash with very similar border b*
# ---------------------------------------------------------------------------


def test_authentic_paired_flash_scores_authentic() -> None:
    """Authentic offset stock: flash image has the same near-neutral
    border b* as the no-flash version. delta_b ≈ 0 → score above the
    AUTHENTIC threshold (0.65).

    Concretely with midpoint=-3 and slope=0.4 the logistic gives
    `1/(1+exp(-1.2)) ≈ 0.769` at delta_b=0 — comfortably above the
    0.65 AUTHENTIC threshold defined in ensemble.py. The slope is
    intentionally gentle: phone-flash UV is weak, so we don't want
    to claim near-1.0 confidence on a shift-of-zero reading."""
    front = _solid_canvas((230, 230, 230))
    flash = _solid_canvas((230, 230, 230))  # identical border, no fluorescence
    r = analyze_substrate(front, flash)
    assert r.abstain_reason is None
    assert r.delta_b is not None
    assert abs(r.delta_b) < 1.0  # near-zero shift on identical inputs
    # Above the AUTHENTIC threshold (0.65) with margin.
    assert r.score >= 0.70, f"got score {r.score} for authentic-shaped pair"


# ---------------------------------------------------------------------------
# Counterfeit — flash image has bluer border (brightener fluorescence proxy)
# ---------------------------------------------------------------------------


def test_brightener_strong_blueshift_scores_counterfeit() -> None:
    """Counterfeit signature: flash image has the border substantially
    shifted toward blue (synthetic stand-in for OBA fluorescence at
    ~440nm). The CIELAB b* shifts negative; score lands counterfeit-side.
    """
    front = _solid_canvas((230, 230, 230))
    # Push B up, R/G down → CIELAB b* drops noticeably.
    flash = _border_b_shifted_canvas(
        (230, 230, 230),
        border_b_offset=+25,
        border_g_offset=-5,
        border_r_offset=-25,
    )
    r = analyze_substrate(front, flash)
    assert r.abstain_reason is None
    assert r.delta_b is not None
    assert r.delta_b <= -5.0, f"expected strong negative b* shift, got {r.delta_b}"
    # With midpoint=-3 and slope=0.4, delta_b ≈ -8 → score ~0.12, well below
    # the COUNTERFEIT threshold (0.35) at default settings.
    assert r.score <= 0.30, f"got score {r.score} for synthetic fake"


def test_strict_brightener_drives_score_near_zero() -> None:
    """Pushing the blueshift further pushes the score lower — the logistic
    is monotonic over delta_b and the synthetic case crosses well below
    the midpoint."""
    front = _solid_canvas((230, 230, 230))
    weaker = _border_b_shifted_canvas(
        (230, 230, 230),
        border_b_offset=+15,
        border_r_offset=-15,
    )
    stronger = _border_b_shifted_canvas(
        (230, 230, 230),
        border_b_offset=+40,
        border_r_offset=-40,
    )
    r_weaker = analyze_substrate(front, weaker)
    r_stronger = analyze_substrate(front, stronger)
    assert r_stronger.delta_b is not None and r_weaker.delta_b is not None
    assert r_stronger.delta_b < r_weaker.delta_b  # more blue
    assert r_stronger.score <= r_weaker.score  # logistic monotonic


# ---------------------------------------------------------------------------
# Abstain paths
# ---------------------------------------------------------------------------


def test_flash_missing_abstains() -> None:
    """No flash image → abstain UNVERIFIED with reason='flash_not_captured'."""
    front = _solid_canvas((230, 230, 230))
    r = analyze_substrate(front, None)
    assert r.confidence == 0.0
    assert r.abstain_reason == "flash_not_captured"
    assert r.delta_b is None
    assert r.border_mad is None


def test_shape_mismatch_abstains() -> None:
    """Front and flash with different shapes → abstain
    UNVERIFIED with reason='shape_mismatch'. The border mask is built
    once from the front shape; mismatched shapes can't share it."""
    front = np.full((1050, 750, 3), 230, dtype=np.uint8)
    flash = np.full((1050, 600, 3), 230, dtype=np.uint8)  # narrower
    r = analyze_substrate(front, flash)
    assert r.confidence == 0.0
    assert r.abstain_reason == "shape_mismatch"
    assert r.delta_b is None


def test_invalid_image_abstains() -> None:
    """Wrong shape (2-D), wrong channels, or too-small image → abstain
    UNVERIFIED with reason='invalid_image'."""
    front = np.zeros((1050, 750), dtype=np.uint8)  # 2-D, no channels
    flash = _solid_canvas((230, 230, 230))
    r = analyze_substrate(front, flash)
    assert r.confidence == 0.0
    assert r.abstain_reason == "invalid_image"


def test_too_small_image_abstains() -> None:
    """Tiny canonical → invalid_image at the input gate. Documented
    contract: detector never crashes, only abstains."""
    front = np.full((9, 9, 3), 230, dtype=np.uint8)
    flash = np.full((9, 9, 3), 230, dtype=np.uint8)
    r = analyze_substrate(front, flash)
    assert r.confidence == 0.0
    assert r.abstain_reason == "invalid_image"


def test_border_too_small_abstains() -> None:
    """Pass a small valid image where the border mask is below
    MIN_BORDER_PIXELS. Border on a 60x60 canvas with frac=0.04 is
    ~6px wide bands minus corner squares — well under 2000 pixels."""
    front = np.full((60, 60, 3), 230, dtype=np.uint8)
    flash = np.full((60, 60, 3), 230, dtype=np.uint8)
    r = analyze_substrate(front, flash)
    assert r.confidence == 0.0
    assert r.abstain_reason == "border_too_small"


# ---------------------------------------------------------------------------
# Confidence cap + ramp
# ---------------------------------------------------------------------------


def test_confidence_never_exceeds_max() -> None:
    """Even with a perfectly clean border (border_mad ≈ 0), confidence
    must respect the MAX_CONFIDENCE=0.7 algorithmic ceiling. Documents
    the design that phone-flash UV is too weak to ever ground-truth as
    high as a UV lamp would warrant."""
    front = _solid_canvas((230, 230, 230))
    flash = _solid_canvas((230, 230, 230))
    r = analyze_substrate(front, flash)
    assert r.abstain_reason is None
    assert r.border_mad is not None
    # Solid-color image → MAD is exactly 0 on the border.
    assert r.border_mad == 0.0
    assert r.confidence <= MAX_CONFIDENCE + 1e-9
    # Sanity: the MAX is what we landed on (capped, not interpolated below).
    assert r.confidence == MAX_CONFIDENCE


def test_confidence_ramp_high_mad_drops_confidence() -> None:
    """Noisy border (high MAD on b*) reduces confidence vs. a clean
    border. Same delta_b range, just noisier substrate → less weight
    in the ensemble. Tests the linear ramp between _CONF_MAD_FULL and
    _CONF_MAD_MIN."""
    rng = np.random.default_rng(42)

    # Build a noisy front: solid body, but border has random b*-style
    # variation. We achieve this by perturbing the BORDER pixels with
    # per-channel noise.
    base = _solid_canvas((230, 230, 230))
    h, w = base.shape[:2]
    band = max(1, int(round(BORDER_WIDTH_FRAC * min(h, w))))
    corner = max(1, int(round(2 * BORDER_WIDTH_FRAC * min(h, w))))
    mask = np.zeros((h, w), dtype=bool)
    mask[:band, :] = True
    mask[h - band:, :] = True
    mask[:, :band] = True
    mask[:, w - band:] = True
    mask[:corner, :corner] = False
    mask[:corner, w - corner:] = False
    mask[h - corner:, :corner] = False
    mask[h - corner:, w - corner:] = False

    noisy_front = base.copy()
    noise = rng.integers(-25, 26, size=(int(mask.sum()), 3), dtype=np.int16)
    noisy_front[mask] = np.clip(
        base[mask].astype(np.int16) + noise, 0, 255
    ).astype(np.uint8)
    # Use the same base for flash (no fluorescence shift) — we only
    # care about the confidence-vs-MAD relationship here.
    flash = base.copy()
    flash[mask] = np.clip(
        base[mask].astype(np.int16) + noise, 0, 255
    ).astype(np.uint8)

    clean = analyze_substrate(base, base)
    noisy = analyze_substrate(noisy_front, flash)
    assert clean.abstain_reason is None
    assert noisy.abstain_reason is None
    assert noisy.border_mad is not None and clean.border_mad is not None
    assert noisy.border_mad > clean.border_mad
    # Confidence should drop on the noisier substrate (or at least not
    # exceed the clean reading).
    assert noisy.confidence <= clean.confidence


# ---------------------------------------------------------------------------
# Result-shape / contract pinning
# ---------------------------------------------------------------------------


def test_result_is_frozen_dataclass() -> None:
    """`SubstrateResult` is a frozen dataclass — assigning fields should
    raise. Documents the immutability contract that callers can rely on
    (matches the other detector results)."""
    front = _solid_canvas((230, 230, 230))
    flash = _solid_canvas((230, 230, 230))
    r = analyze_substrate(front, flash)
    import dataclasses

    with __import__("pytest").raises(dataclasses.FrozenInstanceError):
        r.score = 0.0  # type: ignore[misc]


def test_metadata_includes_calibration_anchors() -> None:
    """Happy-path metadata records the parameters used so a later
    recalibration pass can re-derive the score from history without
    re-running the math."""
    front = _solid_canvas((230, 230, 230))
    flash = _solid_canvas((230, 230, 230))
    r = analyze_substrate(front, flash)
    assert r.abstain_reason is None
    md = r.metadata
    assert "delta_b_midpoint" in md
    assert "delta_b_slope" in md
    assert "border_width_frac" in md
    assert "image_shape" in md
    assert md["delta_b_midpoint"] == DELTA_B_MIDPOINT
    assert md["delta_b_slope"] == DELTA_B_SLOPE


def test_n_border_pixels_above_minimum_on_canonical() -> None:
    """The canonical 750x1050 canvas should produce well over
    MIN_BORDER_PIXELS in the border mask — sanity-check that we sized
    the constant so canonical images don't trip the floor."""
    front = _solid_canvas((230, 230, 230))
    flash = _solid_canvas((230, 230, 230))
    r = analyze_substrate(front, flash)
    assert r.n_border_pixels > MIN_BORDER_PIXELS
    # Concretely, with frac=0.04 on 750x1050 and corner exclusion the
    # mask is ~70k pixels. Pin a generous lower bound.
    assert r.n_border_pixels > 50_000


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


def test_module_constants_have_expected_shape() -> None:
    """The module exposes the constants the spec calls out so callers
    (recalibration tool, ensemble.py) can import them by name."""
    assert BORDER_WIDTH_FRAC == 0.04
    assert MIN_BORDER_PIXELS == 2000
    assert DELTA_B_MIDPOINT == -3.0
    assert DELTA_B_SLOPE == 0.4
    assert MAX_CONFIDENCE == 0.7


def test_logistic_anchors_match_documented_shape() -> None:
    """Sanity-check the score curve hits the documented anchors with the
    chosen midpoint/slope (-3.0, 0.4):
       delta_b ≈ 0 (authentic, no fluorescence) → score ≈ 0.77 (above 0.65)
       delta_b = -3 (midpoint) → score = 0.5
       delta_b ≈ -10 (cheap fake on photo paper) → score < 0.10
    Slope is gentle on purpose — phone-flash UV is weak, so even a strong
    synthetic delta_b shouldn't rocket all the way to ~1.0."""
    f = substrate_detector._logistic
    assert abs(f(0.0 - DELTA_B_MIDPOINT, 0.0, DELTA_B_SLOPE) - 0.77) < 0.05
    assert abs(f(DELTA_B_MIDPOINT - DELTA_B_MIDPOINT, 0.0, DELTA_B_SLOPE) - 0.5) < 1e-9
    assert f(-10.0 - DELTA_B_MIDPOINT, 0.0, DELTA_B_SLOPE) < 0.10
