"""Aggregate quality report.

Combines blur, glare, card detection, fill ratio, and perspective into a
single pass/fail decision for a captured shot. Different shot kinds have
different thresholds — corner zooms must fill less of the frame, flash shots
tolerate higher glare, etc.

The decision is intentionally conservative — the product principle is "reject
rather than guess." A failed shot tells the user exactly which checks failed
so they can retake.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from pipelines.quality.blur import blur_score
from pipelines.quality.card_bbox import detect_card_bbox, fill_ratio, perspective_deg
from pipelines.quality.glare import glare_score


@dataclass(frozen=True)
class QualityThresholds:
    """Per-shot-kind quality thresholds.

    All thresholds are tunable via the API settings — these are sensible
    defaults derived from manual inspection of mid-range phone captures.
    """

    min_blur: float = 100.0
    max_glare: float = 0.005  # 0.5% of pixels
    max_perspective_deg: float = 8.0
    min_fill_ratio: float = 0.40
    max_fill_ratio: float = 0.95
    require_card_detected: bool = True


# Shot-kind-specific overrides. Keys must match grader.db.models.ShotKind values.
THRESHOLDS_BY_SHOT: dict[str, QualityThresholds] = {
    "front_full": QualityThresholds(),
    "back_full": QualityThresholds(),
    "front_full_flash": QualityThresholds(max_glare=0.08),  # flash deliberately produces glare
    "corner_tl": QualityThresholds(
        min_fill_ratio=0.15,
        max_fill_ratio=0.60,
        require_card_detected=False,  # corner crop won't have a 4-quad
    ),
    "corner_tr": QualityThresholds(min_fill_ratio=0.15, max_fill_ratio=0.60, require_card_detected=False),
    "corner_bl": QualityThresholds(min_fill_ratio=0.15, max_fill_ratio=0.60, require_card_detected=False),
    "corner_br": QualityThresholds(min_fill_ratio=0.15, max_fill_ratio=0.60, require_card_detected=False),
    "tilt_30": QualityThresholds(max_perspective_deg=45.0, max_glare=0.05),
}


@dataclass
class QualityReport:
    passed: bool
    reasons: list[str] = field(default_factory=list)
    blur: float | None = None
    glare: float | None = None
    perspective: float | None = None
    card_fill: float | None = None
    card_detected: bool = False

    def to_dict(self) -> dict[str, float | bool | list[str] | None]:
        return {
            "passed": self.passed,
            "reasons": self.reasons,
            "blur": self.blur,
            "glare": self.glare,
            "perspective": self.perspective,
            "card_fill": self.card_fill,
            "card_detected": self.card_detected,
        }


def _thresholds_for(shot_kind: str) -> QualityThresholds:
    return THRESHOLDS_BY_SHOT.get(shot_kind, QualityThresholds())


def evaluate_shot(
    image: NDArray[np.uint8],
    shot_kind: str,
    thresholds: QualityThresholds | None = None,
) -> QualityReport:
    """Evaluate quality of a captured shot. Returns a QualityReport.

    The image is BGR uint8 — same convention as cv2.imread. Decode happens
    upstream (in the storage / service layer)."""
    if image.dtype != np.uint8:
        raise ValueError(f"expected uint8 image, got {image.dtype}")

    th = thresholds or _thresholds_for(shot_kind)
    report = QualityReport(passed=True)

    report.blur = blur_score(image)
    if report.blur < th.min_blur:
        report.passed = False
        report.reasons.append(f"too_blurry (score {report.blur:.1f} < {th.min_blur})")

    if image.ndim == 3 and image.shape[2] == 3:
        report.glare = glare_score(image)
        if report.glare > th.max_glare:
            report.passed = False
            report.reasons.append(
                f"too_much_glare ({report.glare:.3f} > {th.max_glare})"
            )

    card = detect_card_bbox(image)
    report.card_detected = card is not None

    if th.require_card_detected and card is None:
        report.passed = False
        report.reasons.append("card_not_detected")
    elif card is not None:
        report.card_fill = fill_ratio(card, image.shape[:2])
        if report.card_fill < th.min_fill_ratio:
            report.passed = False
            report.reasons.append(
                f"card_too_small (fill {report.card_fill:.2f} < {th.min_fill_ratio})"
            )
        elif report.card_fill > th.max_fill_ratio:
            report.passed = False
            report.reasons.append(
                f"card_too_close (fill {report.card_fill:.2f} > {th.max_fill_ratio})"
            )

        report.perspective = perspective_deg(card)
        if report.perspective > th.max_perspective_deg:
            report.passed = False
            report.reasons.append(
                f"too_off_axis ({report.perspective:.1f}° > {th.max_perspective_deg}°)"
            )

    return report
