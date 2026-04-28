"""Sanity-check the detection training config without invoking ultralytics.

The full trainer requires GPU + a labeled dataset, so we don't run training
in tests. What we *can* check: the config file parses, all required fields
are present, augmentation values that would break card grading (vertical or
horizontal flip) are zero, and image_size is sane."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

CONFIG_PATH = (
    Path(__file__).parent.parent / "training" / "configs" / "detection.yaml"
)


@pytest.fixture(scope="module")
def cfg() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def test_config_parses(cfg: dict) -> None:
    assert isinstance(cfg, dict)


def test_required_top_level_keys_present(cfg: dict) -> None:
    for key in ("dataset", "model", "train", "augment", "mlflow", "output"):
        assert key in cfg, f"missing top-level key: {key}"


def test_single_class_for_v1(cfg: dict) -> None:
    """v1 detection is binary (card vs no-card); face is a Stage 4 concern."""
    assert cfg["dataset"]["num_classes"] == 1
    assert cfg["dataset"]["class_names"] == ["card"]


def test_no_flips_in_augmentation(cfg: dict) -> None:
    """A horizontally flipped Charizard card is not a Charizard card. Flips
    must be off for any card-recognition-adjacent task."""
    assert cfg["augment"]["flipud"] == 0.0
    assert cfg["augment"]["fliplr"] == 0.0


def test_image_size_is_reasonable(cfg: dict) -> None:
    sz = cfg["model"]["image_size"]
    assert isinstance(sz, int)
    assert 512 <= sz <= 2048


def test_base_model_is_yolov11_seg(cfg: dict) -> None:
    assert cfg["model"]["base"].endswith("-seg.pt")
    assert "yolo11" in cfg["model"]["base"]


def test_training_hyperparameters_in_sane_ranges(cfg: dict) -> None:
    train = cfg["train"]
    assert 1 <= train["epochs"] <= 1000
    assert 1 <= train["batch_size"] <= 256
    assert 1e-6 <= train["lr0"] <= 1e-1
    assert 0.0 <= train["lrf"] <= 1.0
    assert 0.0 <= train["weight_decay"] <= 1e-2
