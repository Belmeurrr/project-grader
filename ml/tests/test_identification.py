"""Identification orchestrator + embedder + in-memory catalog tests."""

from __future__ import annotations

import numpy as np
import pytest

from pipelines.identification import (
    EMBEDDING_DIM,
    CardCatalogEntry,
    InMemoryCatalogIndex,
    SimpleEmbedder,
    compute_phash,
    identify,
)
from tests.fixtures import card_in_scene, synth_card, synth_card_with_pattern


def _entry(variant_id: str, image: np.ndarray, embedder: SimpleEmbedder, name: str = "Test") -> CardCatalogEntry:
    return CardCatalogEntry(
        variant_id=variant_id,
        name=name,
        set_code="TST",
        card_number="001",
        game="mtg",
        canonical_phash=compute_phash(image),
        canonical_embedding=embedder.encode(image),
    )


# -----------------------------
# SimpleEmbedder
# -----------------------------


def test_simple_embedder_dim() -> None:
    e = SimpleEmbedder()
    assert e.dim == EMBEDDING_DIM


def test_simple_embedder_returns_correct_shape_and_dtype() -> None:
    e = SimpleEmbedder()
    v = e.encode(card_in_scene())
    assert v.shape == (EMBEDDING_DIM,)
    assert v.dtype == np.float32


def test_simple_embedder_is_l2_normalized() -> None:
    e = SimpleEmbedder()
    v = e.encode(card_in_scene())
    assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-5)


def test_simple_embedder_is_deterministic() -> None:
    e = SimpleEmbedder()
    img = card_in_scene()
    assert np.allclose(e.encode(img), e.encode(img))


def test_simple_embedder_distinguishes_different_cards() -> None:
    e = SimpleEmbedder()
    a = e.encode(synth_card(image_color=(200, 50, 50)))
    b = e.encode(synth_card(image_color=(50, 200, 50)))
    cos = float(np.dot(a, b))
    assert cos < 0.95  # different cards must have meaningful angular separation


def test_simple_embedder_rejects_non_uint8() -> None:
    with pytest.raises(ValueError, match="uint8"):
        SimpleEmbedder().encode(card_in_scene().astype(np.float32))


def test_simple_embedder_rejects_grayscale() -> None:
    img = card_in_scene()[:, :, 0]
    with pytest.raises(ValueError, match="3-channel"):
        SimpleEmbedder().encode(img)


# -----------------------------
# InMemoryCatalogIndex
# -----------------------------


def test_in_memory_catalog_phash_lookup() -> None:
    e = SimpleEmbedder()
    catalog = InMemoryCatalogIndex()
    img = synth_card(image_color=(200, 50, 50))
    catalog.add(_entry("v1", img, e))
    hits = catalog.find_by_phash(compute_phash(img), max_distance=0)
    assert len(hits) == 1
    assert hits[0].entry.variant_id == "v1"
    assert hits[0].distance == 0


def test_in_memory_catalog_phash_within_distance() -> None:
    e = SimpleEmbedder()
    catalog = InMemoryCatalogIndex()
    base = synth_card(image_color=(200, 50, 50))
    catalog.add(_entry("v1", base, e))
    distractor = synth_card(image_color=(20, 20, 240))
    catalog.add(_entry("v2", distractor, e))

    hits = catalog.find_by_phash(compute_phash(base), max_distance=4)
    assert hits[0].entry.variant_id == "v1"


def test_in_memory_catalog_embedding_nearest() -> None:
    e = SimpleEmbedder()
    catalog = InMemoryCatalogIndex()
    a = synth_card(image_color=(200, 50, 50))
    b = synth_card(image_color=(50, 200, 50))
    catalog.add(_entry("v_a", a, e, name="A"))
    catalog.add(_entry("v_b", b, e, name="B"))

    nn = catalog.nearest_by_embedding(e.encode(a), top_k=2)
    assert len(nn) == 2
    assert nn[0].entry.variant_id == "v_a"
    assert nn[0].distance < nn[1].distance


def test_in_memory_catalog_get_by_variant_id() -> None:
    e = SimpleEmbedder()
    catalog = InMemoryCatalogIndex()
    catalog.add(_entry("v1", synth_card(), e))
    assert catalog.get("v1") is not None
    assert catalog.get("missing") is None


def test_in_memory_catalog_skips_entries_without_phash() -> None:
    catalog = InMemoryCatalogIndex()
    catalog.add(
        CardCatalogEntry(
            variant_id="v1", name="N", set_code="S", card_number="1", game="mtg"
        )
    )
    hits = catalog.find_by_phash(0, max_distance=64)
    assert hits == []


def test_in_memory_catalog_phash_rejects_negative() -> None:
    catalog = InMemoryCatalogIndex()
    with pytest.raises(ValueError, match="non-negative"):
        catalog.find_by_phash(-1)


def test_in_memory_catalog_embedding_rejects_2d() -> None:
    catalog = InMemoryCatalogIndex()
    with pytest.raises(ValueError, match="1-D"):
        catalog.nearest_by_embedding(np.zeros((1, 10), dtype=np.float32))


# -----------------------------
# identify orchestrator
# -----------------------------


def test_identify_returns_phash_exact_match_with_high_confidence() -> None:
    e = SimpleEmbedder()
    catalog = InMemoryCatalogIndex()
    img = card_in_scene(fill=0.55)
    target = _entry("v1", img, e, name="Target")
    distractor = _entry("v2", synth_card(image_color=(20, 20, 240)), e, name="Distractor")
    catalog.add(target)
    catalog.add(distractor)

    result = identify(img, catalog=catalog, embedder=e)
    assert result.identified
    assert result.chosen.entry.variant_id == "v1"
    assert result.confidence >= 0.95
    assert result.chosen.method == "phash"


def test_identify_falls_back_to_embedding_when_phash_misses() -> None:
    e = SimpleEmbedder()
    catalog = InMemoryCatalogIndex()
    target_img = synth_card(image_color=(200, 50, 50))
    catalog.add(_entry("v1", target_img, e, name="Target"))
    catalog.add(_entry("v2", synth_card(image_color=(50, 50, 200)), e, name="Other"))

    # Add slight noise so the pHash is no longer an exact match.
    noisy = target_img.copy()
    rng = np.random.default_rng(0)
    noise = rng.integers(-15, 15, size=noisy.shape, dtype=np.int16)
    noisy = np.clip(noisy.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    result = identify(noisy, catalog=catalog, embedder=e)
    # Whether identified or not, the target must be the top candidate.
    assert result.candidates[0].entry.variant_id == "v1"


def test_identify_returns_no_chosen_when_catalog_empty() -> None:
    e = SimpleEmbedder()
    result = identify(card_in_scene(), catalog=InMemoryCatalogIndex(), embedder=e)
    assert not result.identified
    assert result.chosen is None
    assert result.confidence == 0.0
    assert result.candidates == []


def test_identify_no_chosen_when_confidence_below_threshold() -> None:
    """The threshold gate: when min_accept_confidence is higher than any
    candidate can clear, chosen must be None even if candidates exist.
    Set threshold above what an embedding-only match can ever produce
    (cos similarity strictly < 1 for non-identical cards)."""
    e = SimpleEmbedder()
    catalog = InMemoryCatalogIndex()
    catalog.add(_entry("v1", synth_card_with_pattern(seed=11), e, name="Distractor"))

    query = synth_card_with_pattern(seed=22)
    result = identify(query, catalog=catalog, embedder=e, min_accept_confidence=1.0)
    assert not result.identified
    assert result.chosen is None
    assert len(result.candidates) >= 1
    assert result.confidence < 1.0


def test_identify_top_k_bound() -> None:
    e = SimpleEmbedder()
    catalog = InMemoryCatalogIndex()
    for i in range(15):
        catalog.add(
            _entry(
                f"v{i}",
                synth_card(image_color=(10 * i % 255, 50, 200 - 10 * i % 255)),
                e,
                name=f"Card {i}",
            )
        )
    result = identify(card_in_scene(), catalog=catalog, embedder=e, top_k=5)
    assert len(result.candidates) <= 5


def test_identify_rejects_non_uint8() -> None:
    e = SimpleEmbedder()
    catalog = InMemoryCatalogIndex()
    with pytest.raises(ValueError, match="uint8"):
        identify(card_in_scene().astype(np.float32), catalog=catalog, embedder=e)


def test_identify_phash_close_match_uses_phash_method() -> None:
    e = SimpleEmbedder()
    catalog = InMemoryCatalogIndex()
    img = card_in_scene(fill=0.55)
    catalog.add(_entry("v1", img, e))
    result = identify(img, catalog=catalog, embedder=e)
    assert result.chosen is not None
    assert result.chosen.method == "phash"
    assert "phash_distance" in result.chosen.metadata


def test_identify_dedups_same_variant_across_methods() -> None:
    """If the same variant_id appears in both pHash and embedding hits,
    candidates must contain it only once."""
    e = SimpleEmbedder()
    catalog = InMemoryCatalogIndex()
    img = synth_card(image_color=(200, 50, 50))
    catalog.add(_entry("v1", img, e))
    result = identify(img, catalog=catalog, embedder=e)
    ids = [c.entry.variant_id for c in result.candidates]
    assert len(ids) == len(set(ids))
