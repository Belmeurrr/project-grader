"""Perceptual image hashing for card identification.

We use two complementary 64-bit hashes:

- **pHash** — DCT-based, robust to JPEG compression and small color shifts.
  Stored on `card_variants.canonical_phash` for the pre-filter step.
- **dHash** — gradient-based, fast and robust to brightness shifts. Useful
  for cross-checking pHash near-misses.

Hamming distance between two hashes counts the differing bits. For 64-bit
hashes, distances ≤ 6 typically indicate the same card under different
capture conditions; distances ≥ 12 indicate different cards.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

HASH_BITS = 64
_PHASH_RESIZE = 32
_DHASH_HIGH = 9
_DHASH_LOW = 8


def _to_grayscale(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def compute_phash(image: NDArray[np.uint8]) -> int:
    """64-bit perceptual hash via 32x32 DCT.

    Algorithm:
      1. Resize to 32x32 grayscale.
      2. Compute 2D DCT.
      3. Take the 8x8 low-frequency subband (excluding the DC component
         when computing the median; including it for the bit comparison).
      4. Each bit = 1 if coefficient > median, else 0.
    """
    if image.dtype != np.uint8:
        raise ValueError(f"expected uint8 image, got {image.dtype}")
    gray = _to_grayscale(image)
    resized = cv2.resize(gray, (_PHASH_RESIZE, _PHASH_RESIZE), interpolation=cv2.INTER_AREA)

    dct = cv2.dct(resized.astype(np.float32))
    block = dct[:8, :8].copy()
    median = float(np.median(block.flatten()[1:]))  # exclude DC for the threshold

    bits = (block > median).astype(np.uint8).flatten()
    return _bits_to_int(bits)


def compute_dhash(image: NDArray[np.uint8]) -> int:
    """64-bit difference hash. Resize to 9x8 grayscale, then bit per
    horizontal pair: 1 if left > right.
    """
    if image.dtype != np.uint8:
        raise ValueError(f"expected uint8 image, got {image.dtype}")
    gray = _to_grayscale(image)
    resized = cv2.resize(gray, (_DHASH_HIGH, _DHASH_LOW), interpolation=cv2.INTER_AREA)

    diffs = (resized[:, :-1] > resized[:, 1:]).astype(np.uint8).flatten()
    return _bits_to_int(diffs)


def hamming_distance(a: int, b: int) -> int:
    """Number of differing bits in the lower HASH_BITS bits of two ints.

    Uses `int.bit_count()` (CPython 3.10+) — O(1) on the hardware popcnt
    where available."""
    if a < 0 or b < 0:
        raise ValueError("hashes must be non-negative")
    return ((a ^ b) & ((1 << HASH_BITS) - 1)).bit_count()


def hash_to_hex(h: int) -> str:
    return f"{h & ((1 << HASH_BITS) - 1):016x}"


def hex_to_hash(s: str) -> int:
    s = s.strip().lower()
    if s.startswith("0x"):
        s = s[2:]
    if len(s) != 16:
        raise ValueError(f"expected 16 hex chars, got {len(s)}")
    return int(s, 16)


def _bits_to_int(bits: NDArray[np.uint8]) -> int:
    if bits.size != HASH_BITS:
        raise ValueError(f"expected {HASH_BITS} bits, got {bits.size}")
    out = 0
    for bit in bits:
        out = (out << 1) | int(bit)
    return out
