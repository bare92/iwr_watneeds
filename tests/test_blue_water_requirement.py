import numpy as np

from iwr.model import IWRModel


def _build_model() -> IWRModel:
    model = IWRModel.__new__(IWRModel)
    model._last_balance_correction_stats = None
    return model


def test_blue_water_equals_gap_between_potential_and_actual_et() -> None:
    model = _build_model()
    et_potential = np.array([[6.0]], dtype=np.float32)
    et_actual = np.array([[4.0]], dtype=np.float32)

    blue_water = model._compute_blue_water_requirement(et_potential, et_actual, apply_irrigation=True)

    np.testing.assert_allclose(blue_water, [[2.0]])
    np.testing.assert_allclose(et_actual + blue_water, et_potential)


def test_blue_water_is_zero_when_actual_equals_potential() -> None:
    model = _build_model()
    et_potential = np.array([[6.0]], dtype=np.float32)
    et_actual = np.array([[6.0]], dtype=np.float32)

    blue_water = model._compute_blue_water_requirement(et_potential, et_actual, apply_irrigation=True)

    np.testing.assert_allclose(blue_water, [[0.0]])


def test_blue_water_uses_corrected_actual_et() -> None:
    model = _build_model()
    et_potential = np.array([[6.0]], dtype=np.float32)
    corrected_et_actual = np.array([[3.0]], dtype=np.float32)

    blue_water = model._compute_blue_water_requirement(et_potential, corrected_et_actual, apply_irrigation=True)

    np.testing.assert_allclose(blue_water, [[3.0]])


def test_blue_water_does_not_refill_soil_storage_to_raw() -> None:
    model = _build_model()
    s_previous = np.array([[40.0]], dtype=np.float32)
    peff = np.array([[0.0]], dtype=np.float32)
    et_potential = np.array([[6.0]], dtype=np.float32)
    et_actual = np.array([[4.0]], dtype=np.float32)
    deep_percolation = np.array([[0.0]], dtype=np.float32)

    blue_water = model._compute_blue_water_requirement(et_potential, et_actual, apply_irrigation=True)
    soil_storage_new = s_previous + peff - et_actual - deep_percolation

    np.testing.assert_allclose(blue_water, [[2.0]])
    np.testing.assert_allclose(soil_storage_new, [[36.0]])
    assert float(soil_storage_new[0, 0]) != 60.0


def test_blue_water_is_zero_when_irrigation_disabled() -> None:
    model = _build_model()
    et_potential = np.array([[6.0]], dtype=np.float32)
    et_actual = np.array([[4.0]], dtype=np.float32)
    s_previous = np.array([[40.0]], dtype=np.float32)
    peff = np.array([[0.0]], dtype=np.float32)
    deep_percolation = np.array([[0.0]], dtype=np.float32)

    blue_water = model._compute_blue_water_requirement(et_potential, et_actual, apply_irrigation=False)
    soil_storage_new = s_previous + peff - et_actual - deep_percolation

    np.testing.assert_allclose(blue_water, [[0.0]])
    np.testing.assert_allclose(soil_storage_new, [[36.0]])


def test_blue_water_preserves_nan_pixels_independently() -> None:
    model = _build_model()
    et_potential = np.array([[np.nan, 6.0]], dtype=np.float32)
    et_actual = np.array([[np.nan, 4.0]], dtype=np.float32)

    blue_water = model._compute_blue_water_requirement(et_potential, et_actual, apply_irrigation=True)

    assert np.isnan(blue_water[0, 0])
    np.testing.assert_allclose(blue_water[0, 1], 2.0)


def test_blue_water_vectorized_identity_holds_for_finite_irrigated_pixels() -> None:
    model = _build_model()
    et_potential = np.array(
        [
            [6.0, 6.0],
            [6.0, np.nan],
        ],
        dtype=np.float32,
    )
    et_actual = np.array(
        [
            [4.0, 6.0],
            [3.0, np.nan],
        ],
        dtype=np.float32,
    )

    blue_water = model._compute_blue_water_requirement(et_potential, et_actual, apply_irrigation=True)

    finite_pixels = np.isfinite(et_potential) & np.isfinite(et_actual)
    assert np.all(blue_water[finite_pixels] >= 0.0)
    np.testing.assert_allclose(et_actual[finite_pixels] + blue_water[finite_pixels], et_potential[finite_pixels])