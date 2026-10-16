import numpy as np

from iwr.model import IWRModel


def _build_model() -> IWRModel:
    model = IWRModel.__new__(IWRModel)
    model._last_balance_correction_stats = None
    return model


def test_close_negative_water_balance_no_correction_required() -> None:
    model = _build_model()
    available_water = np.array([[110.0]], dtype=np.float32)
    et_actual = np.array([[5.0]], dtype=np.float32)
    deep_percolation = np.array([[20.0]], dtype=np.float32)

    corrected_et, corrected_deep_percolation = model._close_negative_water_balance(
        available_water_mm=available_water,
        et_actual_mm=et_actual,
        deep_percolation_mm=deep_percolation,
    )
    soil_storage_new = available_water - corrected_et - corrected_deep_percolation

    np.testing.assert_allclose(corrected_et, [[5.0]])
    np.testing.assert_allclose(corrected_deep_percolation, [[20.0]])
    np.testing.assert_allclose(soil_storage_new, [[85.0]])
    assert model._last_balance_correction_stats == {
        "pixels_requiring_correction": 0,
        "max_original_deficit_mm": 0.0,
        "min_scale_factor": 1.0,
    }


def test_close_negative_water_balance_scales_proportionally() -> None:
    model = _build_model()
    available_water = np.array([[10.0]], dtype=np.float32)
    et_actual = np.array([[4.0]], dtype=np.float32)
    deep_percolation = np.array([[16.0]], dtype=np.float32)

    corrected_et, corrected_deep_percolation = model._close_negative_water_balance(
        available_water_mm=available_water,
        et_actual_mm=et_actual,
        deep_percolation_mm=deep_percolation,
    )
    soil_storage_new = available_water - corrected_et - corrected_deep_percolation

    np.testing.assert_allclose(corrected_et, [[2.0]])
    np.testing.assert_allclose(corrected_deep_percolation, [[8.0]])
    np.testing.assert_allclose(soil_storage_new, [[0.0]])
    assert model._last_balance_correction_stats is not None
    assert model._last_balance_correction_stats["pixels_requiring_correction"] == 1
    np.testing.assert_allclose(model._last_balance_correction_stats["max_original_deficit_mm"], 10.0)
    np.testing.assert_allclose(model._last_balance_correction_stats["min_scale_factor"], 0.5)


def test_close_negative_water_balance_preserves_outflow_proportionality() -> None:
    model = _build_model()
    available_water = np.array([[10.0]], dtype=np.float32)
    et_actual = np.array([[4.0]], dtype=np.float32)
    deep_percolation = np.array([[16.0]], dtype=np.float32)

    corrected_et, corrected_deep_percolation = model._close_negative_water_balance(
        available_water_mm=available_water,
        et_actual_mm=et_actual,
        deep_percolation_mm=deep_percolation,
    )

    np.testing.assert_allclose(et_actual / deep_percolation, corrected_et / corrected_deep_percolation)


def test_close_negative_water_balance_handles_zero_outflow() -> None:
    model = _build_model()
    available_water = np.array([[10.0]], dtype=np.float32)
    et_actual = np.array([[0.0]], dtype=np.float32)
    deep_percolation = np.array([[0.0]], dtype=np.float32)

    corrected_et, corrected_deep_percolation = model._close_negative_water_balance(
        available_water_mm=available_water,
        et_actual_mm=et_actual,
        deep_percolation_mm=deep_percolation,
    )

    np.testing.assert_allclose(corrected_et, [[0.0]])
    np.testing.assert_allclose(corrected_deep_percolation, [[0.0]])


def test_close_negative_water_balance_preserves_nan_pixels() -> None:
    model = _build_model()
    available_water = np.array([[np.nan, 10.0]], dtype=np.float32)
    et_actual = np.array([[np.nan, 4.0]], dtype=np.float32)
    deep_percolation = np.array([[np.nan, 16.0]], dtype=np.float32)

    corrected_et, corrected_deep_percolation = model._close_negative_water_balance(
        available_water_mm=available_water,
        et_actual_mm=et_actual,
        deep_percolation_mm=deep_percolation,
    )

    assert np.isnan(corrected_et[0, 0])
    assert np.isnan(corrected_deep_percolation[0, 0])
    np.testing.assert_allclose(corrected_et[0, 1], 2.0)
    np.testing.assert_allclose(corrected_deep_percolation[0, 1], 8.0)


def test_close_negative_water_balance_vectorized_pixels_are_independent() -> None:
    model = _build_model()
    available_water = np.array(
        [
            [110.0, 10.0],
            [8.0, np.nan],
        ],
        dtype=np.float32,
    )
    et_actual = np.array(
        [
            [5.0, 4.0],
            [0.0, np.nan],
        ],
        dtype=np.float32,
    )
    deep_percolation = np.array(
        [
            [20.0, 16.0],
            [0.0, np.nan],
        ],
        dtype=np.float32,
    )

    corrected_et, corrected_deep_percolation = model._close_negative_water_balance(
        available_water_mm=available_water,
        et_actual_mm=et_actual,
        deep_percolation_mm=deep_percolation,
    )
    soil_storage_new = available_water - corrected_et - corrected_deep_percolation

    expected_et = np.array(
        [
            [5.0, 2.0],
            [0.0, np.nan],
        ],
        dtype=np.float32,
    )
    expected_deep_percolation = np.array(
        [
            [20.0, 8.0],
            [0.0, np.nan],
        ],
        dtype=np.float32,
    )
    expected_storage = np.array(
        [
            [85.0, 0.0],
            [8.0, np.nan],
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(corrected_et, expected_et, equal_nan=True)
    np.testing.assert_allclose(corrected_deep_percolation, expected_deep_percolation, equal_nan=True)
    np.testing.assert_allclose(soil_storage_new, expected_storage, equal_nan=True)
    assert model._last_balance_correction_stats is not None
    assert model._last_balance_correction_stats["pixels_requiring_correction"] == 1