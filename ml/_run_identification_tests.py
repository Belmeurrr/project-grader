"""Inline runner for hashing + embedder + catalog + identifier tests.
Mirrors the pytest assertions; deletable once pytest is installed."""

from __future__ import annotations

import sys
import traceback

sys.path.insert(0, ".")

import numpy as np

from pipelines.identification import (
    EMBEDDING_DIM,
    HASH_BITS,
    CardCatalogEntry,
    InMemoryCatalogIndex,
    SimpleEmbedder,
    compute_dhash,
    compute_phash,
    hamming_distance,
    hash_to_hex,
    hex_to_hash,
    identify,
)
from tests.fixtures import card_in_scene, synth_card, synth_card_with_pattern

ran, passed, failed = 0, 0, 0


def case(name, fn):
    global ran, passed, failed
    ran += 1
    try:
        fn()
        print(f"PASS {name}")
        passed += 1
    except AssertionError as e:
        print(f"FAIL {name}: AssertionError: {e}")
        failed += 1
    except Exception as e:
        print(f"FAIL {name}: {type(e).__name__}: {e}")
        traceback.print_exc()
        failed += 1


# pHash
def ph1():
    h = compute_phash(card_in_scene())
    assert isinstance(h, int)
    assert 0 <= h < (1 << HASH_BITS)
case("phash_unsigned_64bit", ph1)

def ph2():
    img = card_in_scene()
    assert compute_phash(img) == compute_phash(img)
case("phash_deterministic", ph2)

def ph3():
    import cv2
    img = card_in_scene()
    h1 = compute_phash(img)
    smaller = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
    h2 = compute_phash(smaller)
    assert hamming_distance(h1, h2) <= 6, f"distance={hamming_distance(h1, h2)}"
case("phash_invariant_to_resize", ph3)

def ph4():
    a = synth_card_with_pattern(seed=1)
    b = synth_card_with_pattern(seed=2)
    d = hamming_distance(compute_phash(a), compute_phash(b))
    assert d >= 8, f"distance={d}"
case("phash_distinguishes_different", ph4)

def ph5():
    try:
        compute_phash(card_in_scene().astype(np.float32))
    except ValueError as e:
        assert "uint8" in str(e); return
    raise AssertionError("expected ValueError")
case("phash_rejects_non_uint8", ph5)

# dHash
def dh1():
    h = compute_dhash(card_in_scene())
    assert isinstance(h, int)
    assert 0 <= h < (1 << HASH_BITS)
case("dhash_unsigned_64bit", dh1)

def dh2():
    img = card_in_scene()
    assert compute_dhash(img) == compute_dhash(img)
case("dhash_deterministic", dh2)

def dh3():
    img = synth_card_with_pattern(seed=7)
    h1 = compute_dhash(img)
    flipped = img[:, ::-1].copy()
    d = hamming_distance(h1, compute_dhash(flipped))
    assert d >= 8, f"distance={d}"
case("dhash_distinguishes_horizontal_swap", dh3)

# Hamming
def ha1():
    for v in [0, 1, 0xFFFFFFFFFFFFFFFF, 0xDEADBEEFFACECAFE]:
        assert hamming_distance(v, v) == 0
case("hamming_zero_for_equal", ha1)

def ha2():
    assert hamming_distance(0, 0xF) == 4
    assert hamming_distance(0xFF00, 0x00FF) == 16
case("hamming_basic", ha2)

def ha3():
    a = 0xAAAAAAAAAAAAAAAA
    b = 0x5555555555555555
    assert hamming_distance(a, b) == 64
case("hamming_max_for_complement", ha3)

def ha4():
    try: hamming_distance(-1, 0)
    except ValueError as e:
        assert "non-negative" in str(e); return
    raise AssertionError("expected ValueError")
case("hamming_rejects_negative", ha4)

def ha5():
    a = 0xFFFFFFFFFFFFFFFF
    b = (0xF << 64) | a
    assert hamming_distance(a, b) == 0
case("hamming_truncates_above_64", ha5)

# Hex round-trip
def hx1():
    for v in [0, 1, 0xDEADBEEFFACECAFE, (1 << HASH_BITS) - 1]:
        assert hex_to_hash(hash_to_hex(v)) == v
case("hex_round_trip", hx1)

def hx2():
    assert hex_to_hash("0xdeadbeeffacecafe") == 0xDEADBEEFFACECAFE
case("hex_accepts_0x_prefix", hx2)

def hx3():
    try: hex_to_hash("abcd")
    except ValueError as e:
        assert "16 hex" in str(e); return
    raise AssertionError("expected ValueError")
case("hex_rejects_wrong_length", hx3)

# SimpleEmbedder
def se1():
    e = SimpleEmbedder()
    v = e.encode(card_in_scene())
    assert v.shape == (EMBEDDING_DIM,)
    assert v.dtype == np.float32
case("simple_embedder_shape_dtype", se1)

def se2():
    v = SimpleEmbedder().encode(card_in_scene())
    assert abs(np.linalg.norm(v) - 1.0) < 1e-5, f"norm={np.linalg.norm(v)}"
case("simple_embedder_l2_normalized", se2)

def se3():
    e = SimpleEmbedder()
    img = card_in_scene()
    assert np.allclose(e.encode(img), e.encode(img))
case("simple_embedder_deterministic", se3)

def se4():
    e = SimpleEmbedder()
    a = e.encode(synth_card(image_color=(200, 50, 50)))
    b = e.encode(synth_card(image_color=(50, 200, 50)))
    cos = float(np.dot(a, b))
    assert cos < 0.95, f"cos={cos}"
case("simple_embedder_distinguishes", se4)

def se5():
    try: SimpleEmbedder().encode(card_in_scene().astype(np.float32))
    except ValueError as e:
        assert "uint8" in str(e); return
    raise AssertionError("expected ValueError")
case("simple_embedder_rejects_non_uint8", se5)

def se6():
    try: SimpleEmbedder().encode(card_in_scene()[:, :, 0])
    except ValueError as e:
        assert "3-channel" in str(e); return
    raise AssertionError("expected ValueError")
case("simple_embedder_rejects_grayscale", se6)


# Catalog (in-memory)
def _entry(vid, image, embedder, name="Test"):
    return CardCatalogEntry(
        variant_id=vid, name=name, set_code="TST", card_number="001", game="mtg",
        canonical_phash=compute_phash(image),
        canonical_embedding=embedder.encode(image),
    )

def c1():
    e = SimpleEmbedder()
    catalog = InMemoryCatalogIndex()
    img = synth_card(image_color=(200, 50, 50))
    catalog.add(_entry("v1", img, e))
    hits = catalog.find_by_phash(compute_phash(img), max_distance=0)
    assert len(hits) == 1 and hits[0].entry.variant_id == "v1"
    assert hits[0].distance == 0
case("catalog_phash_exact_match", c1)

def c2():
    e = SimpleEmbedder()
    catalog = InMemoryCatalogIndex()
    base = synth_card(image_color=(200, 50, 50))
    catalog.add(_entry("v1", base, e))
    catalog.add(_entry("v2", synth_card(image_color=(20, 20, 240)), e))
    hits = catalog.find_by_phash(compute_phash(base), max_distance=4)
    assert hits[0].entry.variant_id == "v1"
case("catalog_phash_within_distance", c2)

def c3():
    e = SimpleEmbedder()
    catalog = InMemoryCatalogIndex()
    a = synth_card(image_color=(200, 50, 50))
    b = synth_card(image_color=(50, 200, 50))
    catalog.add(_entry("va", a, e))
    catalog.add(_entry("vb", b, e))
    nn = catalog.nearest_by_embedding(e.encode(a), top_k=2)
    assert nn[0].entry.variant_id == "va"
    assert nn[0].distance < nn[1].distance
case("catalog_embedding_nearest", c3)

def c4():
    e = SimpleEmbedder()
    catalog = InMemoryCatalogIndex()
    catalog.add(_entry("v1", synth_card(), e))
    assert catalog.get("v1") is not None
    assert catalog.get("missing") is None
case("catalog_get_by_id", c4)

def c5():
    catalog = InMemoryCatalogIndex()
    catalog.add(CardCatalogEntry(variant_id="v1", name="N", set_code="S", card_number="1", game="mtg"))
    assert catalog.find_by_phash(0, max_distance=64) == []
case("catalog_skips_null_phash", c5)

def c6():
    catalog = InMemoryCatalogIndex()
    try: catalog.find_by_phash(-1)
    except ValueError as e:
        assert "non-negative" in str(e); return
    raise AssertionError("expected ValueError")
case("catalog_phash_rejects_negative", c6)

def c7():
    try: InMemoryCatalogIndex().nearest_by_embedding(np.zeros((1, 10), dtype=np.float32))
    except ValueError as e:
        assert "1-D" in str(e); return
    raise AssertionError("expected ValueError")
case("catalog_embedding_rejects_2d", c7)


# identify orchestrator
def id1():
    e = SimpleEmbedder()
    catalog = InMemoryCatalogIndex()
    img = card_in_scene(fill=0.55)
    catalog.add(_entry("v1", img, e, name="Target"))
    catalog.add(_entry("v2", synth_card(image_color=(20, 20, 240)), e))
    result = identify(img, catalog=catalog, embedder=e)
    assert result.identified
    assert result.chosen.entry.variant_id == "v1"
    assert result.confidence >= 0.95
    assert result.chosen.method == "phash"
case("identify_phash_exact_match_high_conf", id1)

def id2():
    e = SimpleEmbedder()
    catalog = InMemoryCatalogIndex()
    target_img = synth_card(image_color=(200, 50, 50))
    catalog.add(_entry("v1", target_img, e))
    catalog.add(_entry("v2", synth_card(image_color=(50, 50, 200)), e))
    rng = np.random.default_rng(0)
    noise = rng.integers(-15, 15, size=target_img.shape, dtype=np.int16)
    noisy = np.clip(target_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    result = identify(noisy, catalog=catalog, embedder=e)
    assert result.candidates[0].entry.variant_id == "v1"
case("identify_falls_back_to_embedding", id2)

def id3():
    e = SimpleEmbedder()
    result = identify(card_in_scene(), catalog=InMemoryCatalogIndex(), embedder=e)
    assert not result.identified
    assert result.chosen is None
    assert result.confidence == 0.0
    assert result.candidates == []
case("identify_empty_catalog_returns_none", id3)

def id4():
    e = SimpleEmbedder()
    catalog = InMemoryCatalogIndex()
    catalog.add(_entry("v1", synth_card_with_pattern(seed=11), e))
    query = synth_card_with_pattern(seed=22)
    result = identify(query, catalog=catalog, embedder=e, min_accept_confidence=1.0)
    assert not result.identified
    assert result.chosen is None
    assert len(result.candidates) >= 1
    assert result.confidence < 1.0
case("identify_no_chosen_below_threshold", id4)

def id5():
    e = SimpleEmbedder()
    catalog = InMemoryCatalogIndex()
    for i in range(15):
        catalog.add(_entry(
            f"v{i}",
            synth_card(image_color=(10 * i % 255, 50, 200 - 10 * i % 255)),
            e,
        ))
    result = identify(card_in_scene(), catalog=catalog, embedder=e, top_k=5)
    assert len(result.candidates) <= 5
case("identify_top_k_bound", id5)

def id6():
    try: identify(card_in_scene().astype(np.float32), catalog=InMemoryCatalogIndex(), embedder=SimpleEmbedder())
    except ValueError as e:
        assert "uint8" in str(e); return
    raise AssertionError("expected ValueError")
case("identify_rejects_non_uint8", id6)

def id7():
    e = SimpleEmbedder()
    catalog = InMemoryCatalogIndex()
    img = card_in_scene(fill=0.55)
    catalog.add(_entry("v1", img, e))
    result = identify(img, catalog=catalog, embedder=e)
    assert result.chosen is not None
    assert result.chosen.method == "phash"
    assert "phash_distance" in result.chosen.metadata
case("identify_phash_close_match_uses_phash", id7)

def id8():
    e = SimpleEmbedder()
    catalog = InMemoryCatalogIndex()
    img = synth_card(image_color=(200, 50, 50))
    catalog.add(_entry("v1", img, e))
    result = identify(img, catalog=catalog, embedder=e)
    ids = [c.entry.variant_id for c in result.candidates]
    assert len(ids) == len(set(ids))
case("identify_dedups_across_methods", id8)


print()
print(f"{passed}/{ran} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)
