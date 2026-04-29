"""Synthetic labeled corpus for the counterfeit-detection benchmark.

v1 corpus shape (50 samples — 25 authentic + 25 counterfeit):

  Authentic samples (offset-print-style, halftone-rosette patterns):
    - synth_halftone_card with cell_size in {5, 6, 7, 8} (4 variants)
    - 6 seeds per variant + 1 saturated-color flat-art (high chroma
      with no halftone — tests the color detector path) for 25 total

  Counterfeit samples (continuous-tone inkjet-style):
    - synth_continuous_tone_card (single style; 5 seeds = 5 samples)
    - synth_card with desaturated colors (gray-blues, muted greens,
      etc.) — 4 color variants × 5 seeds = 20 samples

Why these choices:
  - The two detectors target different physical signals. The corpus
    should exercise both:
      • Halftone vs. continuous-tone discriminates rosette
      • High-chroma vs. low-chroma discriminates color profile
  - "Authentic" samples include the easy case (clear halftone) and a
    hard case (no halftone but valid high chroma — should be
    SUSPICIOUS, not flagged as counterfeit).
  - "Counterfeit" samples include the easy case (low chroma + no
    halftone — both detectors flag) and harder cases where each
    detector alone might miss it.

Each sample is deterministic (seed-driven) so the benchmark is
reproducible run-to-run. New variants/seeds extend the corpus
without invalidating existing baselines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from tests.fixtures import (
    synth_card,
    synth_continuous_tone_card,
    synth_halftone_card,
)


GROUND_TRUTH_AUTHENTIC = "authentic"
GROUND_TRUTH_COUNTERFEIT = "counterfeit"


@dataclass(frozen=True)
class BenchmarkSample:
    """One labeled sample for the counterfeit-detection benchmark.

    `image` is generated lazily-per-call from a small generator spec
    rather than stored — keeps the corpus dataclass small and lets
    swap-in of real-image samples reuse the same shape (with `image`
    coming from disk instead).
    """

    sample_id: str
    ground_truth: str  # "authentic" | "counterfeit"
    variant: str
    seed: int
    image: NDArray[np.uint8]
    metadata: dict[str, Any] = field(default_factory=dict)


def _halftone_authentic_samples() -> list[BenchmarkSample]:
    """24 halftone-style authentic samples spanning cell_size + saturation.

    Real authentic cards exhibit BOTH halftone structure AND high
    chroma — they're printed at 150 LPI offset on saturated CMYK ink
    sets. The fixture defaults are desaturated (inner_color=(210,210,215))
    which would make the color detector correctly flag them as
    low-chroma; we override with saturated palettes so each authentic
    sample exercises both detector signals like a real card would.

    24 samples = 4 cell_sizes × 3 saturated palettes × 2 seeds.
    """
    saturated_palettes = [
        # (inner_color, dot_color) — BGR, both saturated
        ((230, 60, 140), (80, 20, 50)),    # magenta inner, dark plum dots
        ((60, 200, 230), (20, 80, 90)),    # cyan/teal inner, dark teal dots
        ((140, 230, 60), (50, 90, 20)),    # green inner, dark green dots
    ]
    samples: list[BenchmarkSample] = []
    for cell_size in (5, 6, 7, 8):
        for palette_idx, (inner, dot) in enumerate(saturated_palettes):
            for seed in range(2):
                variant = f"halftone-cell{cell_size}-palette{palette_idx}"
                sample_id = f"{variant}-seed{seed}"
                img = synth_halftone_card(
                    cell_size=cell_size,
                    inner_color=inner,
                    dot_color=dot,
                    seed=seed,
                )
                samples.append(
                    BenchmarkSample(
                        sample_id=sample_id,
                        ground_truth=GROUND_TRUTH_AUTHENTIC,
                        variant=variant,
                        seed=seed,
                        image=img,
                        metadata={
                            "cell_size": cell_size,
                            "inner_color_bgr": list(inner),
                            "dot_color_bgr": list(dot),
                        },
                    )
                )
    return samples


def _flat_art_authentic_samples() -> list[BenchmarkSample]:
    """1 sample: high-chroma flat art, no halftone. Tests that the
    benchmark exercises the SUSPICIOUS path — a card the rosette
    detector can't speak on but the color detector defends."""
    return [
        BenchmarkSample(
            sample_id="flat-art-saturated",
            ground_truth=GROUND_TRUTH_AUTHENTIC,
            variant="flat-art-saturated",
            seed=0,
            image=synth_card(image_color=(0, 0, 255)),
            metadata={"description": "high-chroma flat art, no halftone"},
        )
    ]


def _continuous_tone_counterfeit_samples() -> list[BenchmarkSample]:
    """5 continuous-tone counterfeit samples — vary the gradient endpoints
    rather than a seed since synth_continuous_tone_card is deterministic
    from its gradient_low / gradient_high parameters."""
    gradients = [
        ((180, 160, 140), (110, 90, 70)),    # default tan-to-brown
        ((150, 170, 200), (90, 110, 150)),   # cool blue gradient
        ((180, 180, 160), (120, 130, 110)),  # muted olive
        ((200, 170, 170), (140, 100, 100)),  # warm tan
        ((170, 170, 170), (110, 110, 110)),  # neutral gray
    ]
    samples: list[BenchmarkSample] = []
    for idx, (lo, hi) in enumerate(gradients):
        samples.append(
            BenchmarkSample(
                sample_id=f"continuous-tone-grad{idx}",
                ground_truth=GROUND_TRUTH_COUNTERFEIT,
                variant="continuous-tone",
                seed=idx,
                image=synth_continuous_tone_card(gradient_low=lo, gradient_high=hi),
                metadata={"gradient_low_bgr": list(lo), "gradient_high_bgr": list(hi)},
            )
        )
    return samples


def _desaturated_counterfeit_samples() -> list[BenchmarkSample]:
    """20 desaturated/low-chroma counterfeit samples covering 4 muted
    color palettes × 5 seeds for variety. The seed isn't used by
    synth_card (the colors are deterministic) but kept on the sample
    record for reproducibility tracing."""
    palettes = [
        ((140, 130, 120), "muted-gray-blue"),
        ((130, 140, 130), "muted-olive"),
        ((125, 125, 145), "muted-purple"),
        ((155, 145, 130), "muted-tan"),
    ]
    samples: list[BenchmarkSample] = []
    for color, label in palettes:
        for seed in range(5):
            samples.append(
                BenchmarkSample(
                    sample_id=f"desaturated-{label}-seed{seed}",
                    ground_truth=GROUND_TRUTH_COUNTERFEIT,
                    variant=f"desaturated-{label}",
                    seed=seed,
                    image=synth_card(image_color=color),
                    metadata={"image_color_bgr": list(color)},
                )
            )
    return samples


def build_default_corpus() -> Sequence[BenchmarkSample]:
    """Assemble the v1 corpus: 25 authentic + 25 counterfeit samples.

    Pure function — same inputs (none) produce the same output. New
    samples are added by extending the helper functions above; the
    benchmark's metric baselines should be re-pinned after corpus
    growth (the test asserting expected counts will fail loudly).
    """
    samples: list[BenchmarkSample] = []
    samples.extend(_halftone_authentic_samples())  # 24
    samples.extend(_flat_art_authentic_samples())  # 1 → 25 authentic
    samples.extend(_continuous_tone_counterfeit_samples())  # 5
    samples.extend(_desaturated_counterfeit_samples())  # 20 → 25 counterfeit
    return tuple(samples)


__all__ = [
    "BenchmarkSample",
    "GROUND_TRUTH_AUTHENTIC",
    "GROUND_TRUTH_COUNTERFEIT",
    "build_default_corpus",
]
