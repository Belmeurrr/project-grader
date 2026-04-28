import pytest

from pipelines.grading.centering import (
    CenteringRatios,
    measure_centering,
    psa_subgrade_from_ratios,
)
from tests.fixtures import synth_card


def test_perfectly_centered_card_yields_50_50() -> None:
    img = synth_card(40, 40, 40, 40)
    result = measure_centering(img)
    assert result.ratios.left == pytest.approx(50.0, abs=1.0)
    assert result.ratios.right == pytest.approx(50.0, abs=1.0)
    assert result.ratios.top == pytest.approx(50.0, abs=1.0)
    assert result.ratios.bottom == pytest.approx(50.0, abs=1.0)


def test_55_45_left_right_offset_detected() -> None:
    # 55/45 horizontally: left border = 55% of total, right = 45%
    total_lr = 80
    left = int(total_lr * 0.55)
    right = total_lr - left
    img = synth_card(left, right, 40, 40)
    result = measure_centering(img)
    assert result.ratios.left == pytest.approx(55.0, abs=2.0)
    assert result.ratios.right == pytest.approx(45.0, abs=2.0)


def test_severe_off_center_70_30_detected() -> None:
    img = synth_card(56, 24, 40, 40)  # 70/30 horizontally
    result = measure_centering(img)
    assert result.ratios.left == pytest.approx(70.0, abs=2.0)
    assert result.ratios.right == pytest.approx(30.0, abs=2.0)


def test_psa_subgrade_perfect_is_10() -> None:
    front = CenteringRatios(50.0, 50.0, 50.0, 50.0)
    assert psa_subgrade_from_ratios(front=front) == 10.0


def test_psa_subgrade_55_45_is_10() -> None:
    front = CenteringRatios(55.0, 45.0, 50.0, 50.0)
    assert psa_subgrade_from_ratios(front=front) == 10.0


def test_psa_subgrade_60_40_is_9() -> None:
    front = CenteringRatios(60.0, 40.0, 50.0, 50.0)
    assert psa_subgrade_from_ratios(front=front) == 9.0


def test_psa_subgrade_70_30_is_7() -> None:
    front = CenteringRatios(70.0, 30.0, 50.0, 50.0)
    assert psa_subgrade_from_ratios(front=front) == 7.0


def test_psa_subgrade_held_back_by_back_centering() -> None:
    front = CenteringRatios(52.0, 48.0, 50.0, 50.0)  # would be 10 alone
    back = CenteringRatios(92.0, 8.0, 50.0, 50.0)    # back tolerances looser; 92/8 still drops it
    subgrade = psa_subgrade_from_ratios(front=front, back=back)
    assert subgrade < 10.0
    assert subgrade <= 6.0


def test_worst_axis_chooses_more_off_center() -> None:
    ratios = CenteringRatios(55.0, 45.0, 70.0, 30.0)
    assert ratios.worst_axis == "tb"


def test_invalid_image_dtype_rejected() -> None:
    img = synth_card().astype("float32")
    with pytest.raises(ValueError, match="uint8"):
        measure_centering(img)  # type: ignore[arg-type]
