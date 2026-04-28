"""Hashing tests — pHash, dHash, Hamming distance, hex round-trip."""

from __future__ import annotations

import numpy as np
import pytest

from pipelines.identification.hashing import (
    HASH_BITS,
    compute_dhash,
    compute_phash,
    hamming_distance,
    hash_to_hex,
    hex_to_hash,
)
from tests.fixtures import card_in_scene, synth_card, synth_card_with_pattern


# -----------------------------
# pHash
# -----------------------------


def test_phash_returns_unsigned_64bit_int() -> None:
    h = compute_phash(card_in_scene())
    assert isinstance(h, int)
    assert 0 <= h < (1 << HASH_BITS)


def test_phash_is_deterministic() -> None:
    img = card_in_scene()
    assert compute_phash(img) == compute_phash(img)


def test_phash_invariant_to_resize() -> None:
    """Resizing within reason should not flip more than a few bits."""
    import cv2

    img = card_in_scene()
    h1 = compute_phash(img)
    smaller = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
    h2 = compute_phash(smaller)
    assert hamming_distance(h1, h2) <= 6


def test_phash_distinguishes_different_images() -> None:
    """pHash is brightness-invariant by design, so different uniform-color
    cards may hash identically. Use cards with rich internal content."""
    a = synth_card_with_pattern(seed=1)
    b = synth_card_with_pattern(seed=2)
    assert hamming_distance(compute_phash(a), compute_phash(b)) >= 8


def test_phash_rejects_non_uint8() -> None:
    with pytest.raises(ValueError, match="uint8"):
        compute_phash(card_in_scene().astype(np.float32))


# -----------------------------
# dHash
# -----------------------------


def test_dhash_returns_unsigned_64bit_int() -> None:
    h = compute_dhash(card_in_scene())
    assert isinstance(h, int)
    assert 0 <= h < (1 << HASH_BITS)


def test_dhash_is_deterministic() -> None:
    img = card_in_scene()
    assert compute_dhash(img) == compute_dhash(img)


def test_dhash_distinguishes_horizontal_swap() -> None:
    """dHash compares horizontal neighbors. A symmetric uniform card hashes
    the same forward and reversed; use a non-symmetric content card."""
    img = synth_card_with_pattern(seed=7)
    h1 = compute_dhash(img)
    flipped = img[:, ::-1].copy()
    h2 = compute_dhash(flipped)
    assert hamming_distance(h1, h2) >= 8


# -----------------------------
# Hamming
# -----------------------------


def test_hamming_zero_for_equal() -> None:
    for v in [0, 1, 0xFFFF_FFFF_FFFF_FFFF, 0xDEAD_BEEF_FACE_CAFE]:
        assert hamming_distance(v, v) == 0


def test_hamming_distance_basic() -> None:
    assert hamming_distance(0b0000, 0b1111) == 4
    assert hamming_distance(0xFF00, 0x00FF) == 16


def test_hamming_max_for_complement() -> None:
    a = 0xAAAA_AAAA_AAAA_AAAA
    b = 0x5555_5555_5555_5555
    assert hamming_distance(a, b) == 64


def test_hamming_rejects_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        hamming_distance(-1, 0)


def test_hamming_truncates_above_64_bits() -> None:
    """Bits above bit 63 must not affect Hamming distance."""
    a = 0xFFFF_FFFF_FFFF_FFFF
    b = (0xF << 64) | a
    assert hamming_distance(a, b) == 0


# -----------------------------
# hex round-trip
# -----------------------------


def test_hash_to_hex_round_trip() -> None:
    for v in [0, 1, 0xDEAD_BEEF_FACE_CAFE, (1 << HASH_BITS) - 1]:
        assert hex_to_hash(hash_to_hex(v)) == v


def test_hex_to_hash_accepts_0x_prefix() -> None:
    assert hex_to_hash("0xdeadbeeffacecafe") == 0xDEAD_BEEF_FACE_CAFE


def test_hex_to_hash_rejects_wrong_length() -> None:
    with pytest.raises(ValueError, match="16 hex chars"):
        hex_to_hash("abcd")
