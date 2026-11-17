import numpy as np

from iwr.model import IWRModel


def _apply_mask(blue_water_mm, phenology_state):
    return IWRModel._apply_active_crop_mask_to_iwr(None, blue_water_mm, phenology_state)


def test_inactive_crop_reports_zero_iwr_but_keeps_et_actual() -> None:
    et_potential = np.array([[5.0]], dtype=np.float32)
    et_actual = np.array([[2.0]], dtype=np.float32)
    blue_water = np.maximum(0.0, et_potential - et_actual)
    phenology_state = np.array([[0]], dtype=np.uint8)

    result = _apply_mask(blue_water, phenology_state)

    np.testing.assert_allclose(blue_water, np.array([[3.0]], dtype=np.float32), rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(result, np.array([[0.0]], dtype=np.float32), rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(et_actual, np.array([[2.0]], dtype=np.float32), rtol=0.0, atol=1e-6)


def test_active_crop_reports_blue_water_requirement() -> None:
    et_potential = np.array([[5.0]], dtype=np.float32)
    et_actual = np.array([[2.0]], dtype=np.float32)
    blue_water = np.maximum(0.0, et_potential - et_actual)
    phenology_state = np.array([[1]], dtype=np.uint8)

    result = _apply_mask(blue_water, phenology_state)

    np.testing.assert_allclose(result, np.array([[3.0]], dtype=np.float32), rtol=0.0, atol=1e-6)


def test_active_crop_without_water_stress_reports_zero_iwr() -> None:
    et_potential = np.array([[5.0]], dtype=np.float32)
    et_actual = np.array([[5.0]], dtype=np.float32)
    blue_water = np.maximum(0.0, et_potential - et_actual)
    phenology_state = np.array([[1]], dtype=np.uint8)

    result = _apply_mask(blue_water, phenology_state)

    np.testing.assert_allclose(result, np.array([[0.0]], dtype=np.float32), rtol=0.0, atol=1e-6)


def test_multiple_growing_seasons_only_report_iwr_during_active_periods() -> None:
    daily_state = np.zeros((366, 1, 1), dtype=np.uint8)
    daily_state[30:60, 0, 0] = 1
    daily_state[210:240, 0, 0] = 1

    blue_water = np.full((366, 1, 1), 5.0, dtype=np.float32)
    masked = IWRModel._apply_active_crop_mask_to_iwr(None, blue_water, daily_state)

    assert float(masked[0, 0, 0]) == 0.0
    assert float(masked[30, 0, 0]) == 5.0
    assert float(masked[59, 0, 0]) == 5.0
    assert float(masked[60, 0, 0]) == 0.0
    assert float(masked[210, 0, 0]) == 5.0
    assert float(masked[239, 0, 0]) == 5.0
    assert float(masked[240, 0, 0]) == 0.0
    assert float(masked.sum()) == 300.0


def test_invalid_pixels_remain_nan() -> None:
    blue_water = np.array([[np.nan, 3.0], [3.0, np.nan]], dtype=np.float32)
    phenology_state = np.array([[0, 1], [1, 0]], dtype=np.uint8)

    result = _apply_mask(blue_water, phenology_state)

    np.testing.assert_equal(np.isnan(result[0, 0]), True)
    np.testing.assert_equal(result[0, 1], 3.0)
    np.testing.assert_equal(result[1, 0], 3.0)
    np.testing.assert_equal(np.isnan(result[1, 1]), True)
