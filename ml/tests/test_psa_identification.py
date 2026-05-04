"""PSAIdentification dataset tests.

Stdlib-only path (manifest builder + splitter + triplet-index sampler)
is exercised without torch. The torch-dependent path
(IdentificationTripletDataset.__getitem__ + DataLoader collate) is
exercised when torch + Pillow are available; otherwise gracefully
skipped so the tests still run in stdlib-only environments.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

# Ensure ml/ root is on sys.path when running this test directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.datasets.psa_identification import (
    IdentificationSample,
    _build_key,
    _normalize_key_part,
    build_identification_samples,
    build_key_index,
    sample_triplet_indices,
    split_by_cert,
)


# Detect torch availability for the optional Dataset tests.
try:
    import torch
    from PIL import Image

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False


# --------------------------------------------------------------------------
# Manifest builder
# --------------------------------------------------------------------------


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def _record(
    cert_id: int = 1,
    grade: float = 9.0,
    image_exists: bool = True,
    tmp_path: Path | None = None,
    card_name: str = "Card",
    set_name: str = "Test Set",
    unique_card_name: bool = False,
) -> dict:
    """Build a minimal scraped.jsonl row. If `image_exists` is True an
    empty file is created at the front_image_path so the dataset's
    `Path.exists()` check passes.

    By default, `card_name` is used verbatim so two records with the
    same `card_name`/`set_name` pair share a key (and therefore form
    a valid same-key, different-cert positive partnership under the
    metric-learning sampler). Set `unique_card_name=True` to suffix
    the cert_id and force a singleton key — useful for tests that
    exercise the singleton-drop path."""
    front = (tmp_path or Path()) / f"img_{cert_id}.jpg"
    if image_exists and tmp_path is not None:
        front.write_bytes(b"\xff\xd8\xff\xd9")  # smallest possible "JPEG"
    final_card_name = f"{card_name} {cert_id}" if unique_card_name else card_name
    return {
        "cert_id": cert_id,
        "grade": grade,
        "card_name": final_card_name,
        "set_name": set_name,
        "year": 2020,
        "card_number": str(cert_id),
        "front_image_path": str(front) if image_exists else None,
        "back_image_path": None,
        "source_url": f"https://x/{cert_id}",
        "scraped_at": "2026-04-29T00:00:00Z",
    }


def test_build_identification_samples_emits_one_per_record(tmp_path: Path) -> None:
    """Two records sharing card_name+set_name (a "valid pair") survive
    the singleton-drop filter and both end up in the manifest."""
    jsonl = tmp_path / "scraped.jsonl"
    _write_jsonl(jsonl, [
        _record(cert_id=1, tmp_path=tmp_path),
        _record(cert_id=2, tmp_path=tmp_path),
    ])
    samples = build_identification_samples(jsonl)
    assert len(samples) == 2
    assert {s.cert_id for s in samples} == {1, 2}
    assert all(isinstance(s, IdentificationSample) for s in samples)


def test_build_identification_samples_skips_missing_front_image_path(tmp_path: Path) -> None:
    """Filter on missing front image happens BEFORE singleton-drop, so
    we need three records (so the two-with-image survivors form a
    valid same-key pair)."""
    jsonl = tmp_path / "scraped.jsonl"
    _write_jsonl(jsonl, [
        _record(cert_id=1, tmp_path=tmp_path),
        _record(cert_id=2, tmp_path=tmp_path),
        _record(cert_id=3, image_exists=False, tmp_path=tmp_path),
    ])
    samples = build_identification_samples(jsonl)
    assert sorted(s.cert_id for s in samples) == [1, 2]


def test_build_identification_samples_skips_records_with_missing_image_file(tmp_path: Path) -> None:
    jsonl = tmp_path / "scraped.jsonl"
    rec = _record(cert_id=1, tmp_path=tmp_path)
    Path(rec["front_image_path"]).unlink()
    _write_jsonl(jsonl, [rec])
    samples = build_identification_samples(jsonl)
    assert samples == []


def test_build_identification_samples_filters_grades_outside_range(tmp_path: Path) -> None:
    """Out-of-range grades are dropped before singleton-drop. With only
    one in-range record left, the singleton-drop then kicks in and the
    manifest is empty (correct behavior — a single sample can't form a
    same-key, different-cert positive)."""
    jsonl = tmp_path / "scraped.jsonl"
    _write_jsonl(jsonl, [
        _record(cert_id=1, grade=0.5, tmp_path=tmp_path),
        _record(cert_id=2, grade=5.0, tmp_path=tmp_path),
        _record(cert_id=3, grade=11.0, tmp_path=tmp_path),
    ])
    samples = build_identification_samples(jsonl)
    assert samples == []


def test_build_identification_samples_captures_card_and_set_name(tmp_path: Path) -> None:
    """card_name + set_name must be on the IdentificationSample so the
    metric-learning positive sampler can group same-printing records.
    Two records share the (card_name, set_name) pair so they survive
    the singleton-drop."""
    jsonl = tmp_path / "scraped.jsonl"
    _write_jsonl(jsonl, [
        _record(cert_id=1, tmp_path=tmp_path, card_name="Pikachu", set_name="Base"),
        _record(cert_id=2, tmp_path=tmp_path, card_name="Pikachu", set_name="Base"),
    ])
    samples = build_identification_samples(jsonl)
    assert all(s.card_name == "Pikachu" for s in samples)
    assert all(s.set_name == "Base" for s in samples)


def test_build_identification_samples_preserves_full_record(tmp_path: Path) -> None:
    """The whole JSONL row is kept on the IdentificationSample so a
    future positive sampler can use any field (year / variant / number)
    without re-reading the manifest."""
    jsonl = tmp_path / "scraped.jsonl"
    rec1 = _record(cert_id=1, tmp_path=tmp_path)
    rec1["custom_field"] = "future-positive-sampler-input"
    rec2 = _record(cert_id=2, tmp_path=tmp_path)
    _write_jsonl(jsonl, [rec1, rec2])
    samples = build_identification_samples(jsonl)
    rec1_sample = next(s for s in samples if s.cert_id == 1)
    assert rec1_sample.record["custom_field"] == "future-positive-sampler-input"


def test_build_identification_samples_tolerates_malformed_lines(tmp_path: Path) -> None:
    """A truncated trailing JSONL line (e.g. process killed mid-write)
    must not abort the build — the trainer should still see all the
    records that DID land cleanly. Mirrors the surface-dataset test.

    Two records to satisfy the singleton-drop; the truncated third
    line is malformed and dropped at the JSONL parser layer."""
    jsonl = tmp_path / "scraped.jsonl"
    rec1 = _record(cert_id=1, tmp_path=tmp_path)
    rec2 = _record(cert_id=2, tmp_path=tmp_path)
    jsonl.write_text(
        json.dumps(rec1) + "\n"
        + json.dumps(rec2) + "\n"
        + '{"cert_id": 99, "grade": 9.0, "front_image_path": "x',  # truncated
        encoding="utf-8",
    )
    samples = build_identification_samples(jsonl)
    assert sorted(s.cert_id for s in samples) == [1, 2]


# --------------------------------------------------------------------------
# Split helper
# --------------------------------------------------------------------------


def _samples_for_certs(
    cert_ids: list[int],
    tmp_path: Path,
    *,
    shared_key: bool = False,
) -> list[IdentificationSample]:
    """Convenience: build IdentificationSamples from a list of cert_ids
    without going through JSONL.

    By default each cert gets its own unique card_name (so each
    sample is a singleton key — the original splitter tests didn't
    care about keys). Set `shared_key=True` to give all samples the
    same `(card_name, set_name)` pair, which is what the new positive-
    sampling tests need."""
    out: list[IdentificationSample] = []
    for cid in cert_ids:
        path = tmp_path / f"img_{cid}.jpg"
        path.write_bytes(b"")
        card_name = "Shared Card" if shared_key else f"Card {cid}"
        out.append(IdentificationSample(
            cert_id=cid,
            image_path=path,
            grade=9.0,
            card_name=card_name,
            set_name="Test",
            record={"cert_id": cid, "grade": 9.0},
        ))
    return out


def test_split_by_cert_keeps_card_intact(tmp_path: Path) -> None:
    samples = _samples_for_certs(list(range(20)), tmp_path)
    train, val = split_by_cert(samples, val_fraction=0.25, seed=42)
    train_certs = {s.cert_id for s in train}
    val_certs = {s.cert_id for s in val}
    assert not (train_certs & val_certs)


def test_split_by_cert_is_deterministic_on_seed(tmp_path: Path) -> None:
    samples = _samples_for_certs(list(range(30)), tmp_path)
    a_train, a_val = split_by_cert(samples, val_fraction=0.2, seed=123)
    b_train, b_val = split_by_cert(samples, val_fraction=0.2, seed=123)
    assert [s.cert_id for s in a_train] == [s.cert_id for s in b_train]
    assert [s.cert_id for s in a_val] == [s.cert_id for s in b_val]


def test_split_by_cert_handles_empty_input() -> None:
    train, val = split_by_cert([], val_fraction=0.2)
    assert train == []
    assert val == []


# --------------------------------------------------------------------------
# Key normalization + key index
# --------------------------------------------------------------------------


def test_normalize_key_part_lowercases_and_collapses_whitespace() -> None:
    """Normalization rule: lowercase + strip + collapse internal
    whitespace runs. Two records that differ only in casing or spacing
    must collide on the same key."""
    assert _normalize_key_part("Pikachu") == "pikachu"
    assert _normalize_key_part("  Pikachu  ") == "pikachu"
    assert _normalize_key_part("Pokemon  Trainer") == "pokemon trainer"
    assert _normalize_key_part("\tPokemon\nTrainer\n") == "pokemon trainer"


def test_build_key_returns_normalized_pair() -> None:
    assert _build_key("Pikachu", "  Base Set  ") == ("pikachu", "base set")


def test_build_key_index_groups_same_key(tmp_path: Path) -> None:
    samples = _samples_for_certs([1, 2, 3], tmp_path, shared_key=True)
    idx = build_key_index(samples)
    assert len(idx) == 1
    key = next(iter(idx))
    assert sorted(idx[key]) == [0, 1, 2]


# --------------------------------------------------------------------------
# Singleton-drop in build_identification_samples
# --------------------------------------------------------------------------


def test_build_identification_samples_drops_singleton_keys(tmp_path: Path) -> None:
    """A record whose (card_name, set_name) appears only once cannot
    supply a valid same-key, different-cert positive partner — so it
    must be dropped at manifest-build time."""
    jsonl = tmp_path / "scraped.jsonl"
    _write_jsonl(jsonl, [
        _record(cert_id=1, tmp_path=tmp_path, card_name="Pikachu"),
        _record(cert_id=2, tmp_path=tmp_path, card_name="Pikachu"),
        # Singleton — no other record shares this card_name.
        _record(cert_id=3, tmp_path=tmp_path, card_name="Charizard"),
    ])
    samples = build_identification_samples(jsonl)
    assert sorted(s.cert_id for s in samples) == [1, 2]
    assert all(s.card_name == "Pikachu" for s in samples)


def test_build_identification_samples_normalizes_keys_when_grouping(tmp_path: Path) -> None:
    """Two records with cosmetic differences in card_name or set_name
    (extra whitespace, different casing) must collide on the same
    normalized key and survive the singleton-drop together."""
    jsonl = tmp_path / "scraped.jsonl"
    _write_jsonl(jsonl, [
        _record(cert_id=1, tmp_path=tmp_path, card_name="Pikachu", set_name="Base"),
        _record(cert_id=2, tmp_path=tmp_path, card_name="  PIKACHU  ", set_name="base"),
    ])
    samples = build_identification_samples(jsonl)
    assert sorted(s.cert_id for s in samples) == [1, 2]
    # Both samples have the same normalized key.
    assert len({s.key for s in samples}) == 1


def test_build_identification_samples_drops_same_cert_only_groups(tmp_path: Path) -> None:
    """If a key has multiple samples but they all share the same
    cert_id (degenerate case — shouldn't happen with one record per
    cert in the JSONL, but the filter still must enforce 'distinct
    cert' rather than just 'group size > 1')."""
    jsonl = tmp_path / "scraped.jsonl"
    rec1 = _record(cert_id=7, tmp_path=tmp_path, card_name="Pikachu")
    # Build a second JSONL row with the SAME cert_id (which the filter
    # treats as one cert under the same key).
    rec2 = _record(cert_id=7, tmp_path=tmp_path, card_name="Pikachu")
    _write_jsonl(jsonl, [rec1, rec2])
    samples = build_identification_samples(jsonl)
    assert samples == []


def test_identification_sample_auto_populates_key() -> None:
    """An IdentificationSample built without an explicit `key` must
    derive it from card_name + set_name via the normalize rule."""
    s = IdentificationSample(
        cert_id=1,
        image_path=Path("x"),
        grade=9.0,
        card_name="  PIKACHU ",
        set_name="Base",
        record={},
    )
    assert s.key == ("pikachu", "base")


# --------------------------------------------------------------------------
# Triplet-index sampler (pure stdlib)
# --------------------------------------------------------------------------


def test_sample_triplet_indices_picks_same_key_different_cert(tmp_path: Path) -> None:
    """Supervised metric learning: the positive must share the
    anchor's normalized key but have a different cert_id — that's the
    whole point of the swap from augmentation-based positives."""
    samples = _samples_for_certs([10, 20, 30], tmp_path, shared_key=True)
    rng = random.Random(0)
    for anchor_idx in range(len(samples)):
        a, p, n = sample_triplet_indices(samples, anchor_index=anchor_idx, rng=rng)
        assert a == anchor_idx
        assert samples[p].key == samples[anchor_idx].key
        assert samples[p].cert_id != samples[anchor_idx].cert_id
        assert n != anchor_idx


def test_sample_triplet_indices_negative_differs_from_anchor(tmp_path: Path) -> None:
    """Negative must NEVER equal the anchor index — otherwise the
    triplet collapses (anchor==positive==negative implies zero
    gradient with margin>0 and an all-zero embedding fixed point)."""
    samples = _samples_for_certs([1, 2, 3, 4, 5], tmp_path, shared_key=True)
    rng = random.Random(0)
    for _ in range(50):
        a, _, n = sample_triplet_indices(samples, anchor_index=2, rng=rng)
        assert a == 2
        assert n != 2


def test_sample_triplet_indices_explores_all_negatives(tmp_path: Path) -> None:
    """Over many draws, the negative sampler must hit every non-anchor
    index. This guards against an off-by-one in a future refactor that
    could otherwise silently exclude the last index."""
    samples = _samples_for_certs([1, 2, 3, 4, 5, 6], tmp_path, shared_key=True)
    rng = random.Random(0)
    anchor = 0
    seen_negatives: set[int] = set()
    for _ in range(500):
        _, _, neg = sample_triplet_indices(samples, anchor_index=anchor, rng=rng)
        seen_negatives.add(neg)
    assert seen_negatives == {1, 2, 3, 4, 5}


def test_sample_triplet_indices_explores_all_same_key_positives(tmp_path: Path) -> None:
    """Over many draws, the positive sampler must hit every same-key,
    different-cert candidate. Guards against a sampler bug that locks
    onto a single positive partner."""
    samples = _samples_for_certs([10, 11, 12, 13], tmp_path, shared_key=True)
    rng = random.Random(0)
    anchor = 0
    seen_positives: set[int] = set()
    for _ in range(500):
        _, pos, _ = sample_triplet_indices(samples, anchor_index=anchor, rng=rng)
        seen_positives.add(pos)
    assert seen_positives == {1, 2, 3}


def test_sample_triplet_indices_rejects_singleton_corpus(tmp_path: Path) -> None:
    samples = _samples_for_certs([1], tmp_path, shared_key=True)
    rng = random.Random(0)
    with pytest.raises(ValueError, match="at least 2 samples"):
        sample_triplet_indices(samples, anchor_index=0, rng=rng)


def test_sample_triplet_indices_rejects_out_of_range_anchor(tmp_path: Path) -> None:
    samples = _samples_for_certs([1, 2, 3, 4, 5], tmp_path, shared_key=True)
    rng = random.Random(0)
    with pytest.raises(ValueError, match="out of range"):
        sample_triplet_indices(samples, anchor_index=10, rng=rng)


def test_sample_triplet_indices_rejects_when_anchor_has_no_partner(tmp_path: Path) -> None:
    """If somehow a singleton-key sample slipped past the manifest
    filter, the top-level sampler should error loudly rather than
    silently fall back to anchor==positive."""
    # Construct samples directly so we can build a degenerate input
    # that bypasses build_identification_samples.
    s1 = IdentificationSample(
        cert_id=1, image_path=tmp_path / "a.jpg", grade=9.0,
        card_name="Solo", set_name="Test", record={},
    )
    s2 = IdentificationSample(
        cert_id=2, image_path=tmp_path / "b.jpg", grade=9.0,
        card_name="Other", set_name="Test", record={},
    )
    rng = random.Random(0)
    with pytest.raises(ValueError, match="no same-key, different-cert"):
        sample_triplet_indices([s1, s2], anchor_index=0, rng=rng)


# --------------------------------------------------------------------------
# Dataset (torch path — gracefully skipped when torch isn't installed)
# --------------------------------------------------------------------------


def _torch_samples_with_shared_key(
    tmp_path: Path,
    n: int,
    *,
    color_per_index: bool = True,
) -> list[IdentificationSample]:
    """Build `n` IdentificationSamples whose images live on disk and
    whose (card_name, set_name) is shared so they form a single big
    same-key group for the metric-learning positive sampler."""
    samples: list[IdentificationSample] = []
    for i in range(n):
        p = tmp_path / f"img_{i}.jpg"
        color = (i * 60 % 255, 40, 80) if color_per_index else (60, 40, 80)
        Image.new("RGB", (256, 256), color=color).save(p, "JPEG")
        samples.append(IdentificationSample(
            cert_id=i,
            image_path=p,
            grade=9.0,
            card_name="Shared Card",
            set_name="Test Set",
            record={"cert_id": i},
        ))
    return samples


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch / Pillow not installed")
def test_dataset_returns_three_tensors_of_correct_shape(tmp_path: Path) -> None:
    from training.datasets.psa_identification import IdentificationTripletDataset

    samples = _torch_samples_with_shared_key(tmp_path, n=4)
    ds = IdentificationTripletDataset(samples, image_size=64, train=True, seed=0)
    a, p_t, n = ds[0]
    assert a.shape == (3, 64, 64)
    assert p_t.shape == (3, 64, 64)
    assert n.shape == (3, 64, 64)


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch / Pillow not installed")
def test_dataset_anchor_and_positive_have_same_key_different_cert(tmp_path: Path) -> None:
    """Supervised metric-learning positive: anchor and positive must be
    SEPARATE samples with the same normalized key but different
    cert_ids. This is the core behavior swap from augmentation-only
    triplet learning."""
    from training.datasets.psa_identification import IdentificationTripletDataset

    samples = _torch_samples_with_shared_key(tmp_path, n=4)
    ds = IdentificationTripletDataset(samples, image_size=64, train=True, seed=0)
    # Probe the index sampler directly so we don't depend on tensor
    # equality (the torch transforms are stochastic anyway).
    for anchor_idx in range(len(samples)):
        pos_idx = ds._sample_positive_index(anchor_idx)
        assert pos_idx != anchor_idx
        assert samples[pos_idx].key == samples[anchor_idx].key
        assert samples[pos_idx].cert_id != samples[anchor_idx].cert_id


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch / Pillow not installed")
def test_dataset_positive_fallback_does_not_fire_on_healthy_corpus(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A corpus that's been through `build_identification_samples` (or
    is otherwise healthy) must NEVER trigger the warning-and-fall-back-
    to-anchor path. If it does, something upstream is broken — the
    fallback exists for safety but should be unreachable in practice."""
    from training.datasets.psa_identification import IdentificationTripletDataset

    samples = _torch_samples_with_shared_key(tmp_path, n=4)
    ds = IdentificationTripletDataset(samples, image_size=64, train=True, seed=0)
    with caplog.at_level("WARNING", logger="psa_identification_dataset"):
        for anchor_idx in range(len(samples)):
            for _ in range(20):
                pos_idx = ds._sample_positive_index(anchor_idx)
                assert pos_idx != anchor_idx  # cannot be the anchor
    fallback_warnings = [
        r for r in caplog.records
        if "no same-key, different-cert positive partner" in r.getMessage()
    ]
    assert fallback_warnings == []


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch / Pillow not installed")
def test_dataset_positive_fallback_fires_when_corpus_unfiltered(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """If a singleton sample slips into the Dataset (e.g. someone
    bypassed `build_identification_samples`), the positive sampler
    must fall back to the anchor index AND log a warning so the
    operator can investigate."""
    from training.datasets.psa_identification import IdentificationTripletDataset

    # Two samples with DIFFERENT keys — both are singletons and
    # neither has a valid positive partner.
    samples: list[IdentificationSample] = []
    for i in range(2):
        p = tmp_path / f"img_{i}.jpg"
        Image.new("RGB", (200, 200), color=(i * 100, 0, 0)).save(p, "JPEG")
        samples.append(IdentificationSample(
            cert_id=i,
            image_path=p,
            grade=9.0,
            card_name=f"Unique Card {i}",
            set_name="Test",
            record={"cert_id": i},
        ))

    ds = IdentificationTripletDataset(samples, image_size=64, train=True, seed=0)
    with caplog.at_level("WARNING", logger="psa_identification_dataset"):
        pos_idx = ds._sample_positive_index(0)
    assert pos_idx == 0  # fallback returned the anchor
    fallback_warnings = [
        r for r in caplog.records
        if "no same-key, different-cert positive partner" in r.getMessage()
    ]
    assert len(fallback_warnings) == 1


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch / Pillow not installed")
def test_dataset_negative_image_differs_from_anchor(tmp_path: Path) -> None:
    """Negative sample comes from a DIFFERENT record than the anchor —
    which means a different on-disk image with a different mean color.
    Sanity-check: the negative tensor should not be identical to the
    anchor tensor in eval mode."""
    from training.datasets.psa_identification import IdentificationTripletDataset

    samples = _torch_samples_with_shared_key(tmp_path, n=3)
    ds = IdentificationTripletDataset(samples, image_size=64, train=False, seed=0)
    a, _, n = ds[0]
    assert not torch.equal(a, n)


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch / Pillow not installed")
def test_dataset_collates_into_batch(tmp_path: Path) -> None:
    """A DataLoader's default collate must work on triplets — i.e. all
    three tensors per item must have stack-compatible shapes. This
    catches accidental shape drift between samples (e.g. if augmentation
    introduced randomness in output size)."""
    from torch.utils.data import DataLoader

    from training.datasets.psa_identification import IdentificationTripletDataset

    samples = _torch_samples_with_shared_key(tmp_path, n=4)
    ds = IdentificationTripletDataset(samples, image_size=64, train=False, seed=0)
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
    a, p_t, n = next(iter(loader))
    assert a.shape == (4, 3, 64, 64)
    assert p_t.shape == (4, 3, 64, 64)
    assert n.shape == (4, 3, 64, 64)


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch / Pillow not installed")
def test_dataset_rejects_singleton_corpus(tmp_path: Path) -> None:
    """A 1-sample corpus has no possible negative — the Dataset
    constructor must reject it loudly rather than infinite-loop in the
    negative sampler."""
    from training.datasets.psa_identification import IdentificationTripletDataset

    p = tmp_path / "img.jpg"
    Image.new("RGB", (200, 200), color=(0, 0, 0)).save(p, "JPEG")
    sample = IdentificationSample(
        cert_id=1, image_path=p, grade=9.0,
        card_name="x", set_name="y", record={},
    )
    with pytest.raises(ValueError, match="at least 2 samples"):
        IdentificationTripletDataset([sample], image_size=64, train=True)
