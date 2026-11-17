import numpy as np
import pytest

from iwr.model import IWRInputs, IWRModel
from iwr.crops import Crop
from iwr.soil import Soil


def _build_model() -> IWRModel:
    model = IWRModel.__new__(IWRModel)
    model._last_balance_correction_stats = None
    return model


def test_blue_water_equals_gap_between_potential_and_actual_et() -> None:
    model = _build_model()
    et_potential = np.array([[6.0]], dtype=np.float32)
    et_actual = np.array([[4.0]], dtype=np.float32)

    blue_water = model._compute_blue_water_requirement(et_potential, et_actual)

    np.testing.assert_allclose(blue_water, [[2.0]])
    np.testing.assert_allclose(et_actual + blue_water, et_potential)


def test_blue_water_is_zero_when_actual_equals_potential() -> None:
    model = _build_model()
    et_potential = np.array([[6.0]], dtype=np.float32)
    et_actual = np.array([[6.0]], dtype=np.float32)

    blue_water = model._compute_blue_water_requirement(et_potential, et_actual)

    np.testing.assert_allclose(blue_water, [[0.0]])


def test_blue_water_uses_corrected_actual_et() -> None:
    model = _build_model()
    et_potential = np.array([[6.0]], dtype=np.float32)
    corrected_et_actual = np.array([[3.0]], dtype=np.float32)

    blue_water = model._compute_blue_water_requirement(et_potential, corrected_et_actual)

    np.testing.assert_allclose(blue_water, [[3.0]])


def test_blue_water_does_not_refill_soil_storage_to_raw() -> None:
    model = _build_model()
    s_previous = np.array([[40.0]], dtype=np.float32)
    peff = np.array([[0.0]], dtype=np.float32)
    et_potential = np.array([[6.0]], dtype=np.float32)
    et_actual = np.array([[4.0]], dtype=np.float32)
    deep_percolation = np.array([[0.0]], dtype=np.float32)

    blue_water = model._compute_blue_water_requirement(et_potential, et_actual)
    soil_storage_new = s_previous + peff - et_actual - deep_percolation

    np.testing.assert_allclose(blue_water, [[2.0]])
    np.testing.assert_allclose(soil_storage_new, [[36.0]])
    assert float(soil_storage_new[0, 0]) != 60.0


def test_report_blue_water_requirement_can_be_disabled_without_changing_balance() -> None:
    crop = Crop(
        name="test_crop",
        root_max_depth=1.0,
        kc_initial=0.3,
        kc_max=1.0,
        kc_sen=0.4,
        p=0.5,
        crop_raster=np.ones((1, 1), dtype=np.float32),
    )
    soil = Soil(
        field_capacity=np.array([[0.5]], dtype=np.float32),
        wilting_point=np.array([[0.4]], dtype=np.float32),
        max_infiltration_rate=np.array([[10.0]], dtype=np.float32),
        texture_map=np.array([[1]], dtype=np.uint8),
        soil_moisture=np.array([[0.5]], dtype=np.float32),
    )
    inputs = IWRInputs(
        scheduler=None,
        output_dir=".",
        crop=crop,
        soil=soil,
        phenology=None,
        working_grid=np.ones((1, 1), dtype=np.uint8),
        report_blue_water_requirement=False,
        simulate_irrigation=False,
    )
    model = IWRModel(inputs)
    et_potential = np.array([[5.0]], dtype=np.float32)
    et_actual = np.array([[2.0]], dtype=np.float32)
    blue_water = model._compute_blue_water_requirement(et_potential, et_actual)
    reported_iwr = model._apply_active_crop_mask_to_iwr(blue_water, np.array([[1]], dtype=np.uint8))
    reported_iwr_zero = np.zeros_like(blue_water, dtype=np.float32)
    reported_iwr_zero[~np.isfinite(blue_water)] = np.nan

    np.testing.assert_allclose(blue_water, np.array([[3.0]], dtype=np.float32))
    np.testing.assert_allclose(reported_iwr, np.array([[3.0]], dtype=np.float32))
    np.testing.assert_allclose(reported_iwr_zero, np.array([[0.0]], dtype=np.float32))


def test_irrigation_area_mask_restricts_reported_iwr() -> None:
    crop = Crop(
        name="test_crop",
        root_max_depth=1.0,
        kc_initial=0.3,
        kc_max=1.0,
        kc_sen=0.4,
        p=0.5,
        crop_raster=np.ones((1, 1), dtype=np.float32),
    )
    soil = Soil(
        field_capacity=np.array([[0.5]], dtype=np.float32),
        wilting_point=np.array([[0.4]], dtype=np.float32),
        max_infiltration_rate=np.array([[10.0]], dtype=np.float32),
        texture_map=np.array([[1]], dtype=np.uint8),
        soil_moisture=np.array([[0.5]], dtype=np.float32),
    )
    inputs = IWRInputs(
        scheduler=None,
        output_dir=".",
        crop=crop,
        soil=soil,
        phenology=None,
        working_grid=np.ones((1, 1), dtype=np.uint8),
        report_blue_water_requirement=True,
        simulate_irrigation=False,
        irrigation_area_mask=np.array([[1]], dtype=np.uint8),
    )
    model = IWRModel(inputs)
    blue_water = np.array([[3.0]], dtype=np.float32)
    active = np.array([[1]], dtype=np.uint8)
    iwr = model._apply_active_crop_mask_to_iwr(blue_water, active)
    iwr = model._apply_irrigation_area_mask_to_iwr(iwr, model.irrigation_area_mask)

    np.testing.assert_allclose(iwr, np.array([[3.0]], dtype=np.float32))


def test_blue_water_preserves_nan_pixels_independently() -> None:
    model = _build_model()
    et_potential = np.array([[np.nan, 6.0]], dtype=np.float32)
    et_actual = np.array([[np.nan, 4.0]], dtype=np.float32)

    blue_water = model._compute_blue_water_requirement(et_potential, et_actual)

    assert np.isnan(blue_water[0, 0])
    np.testing.assert_allclose(blue_water[0, 1], 2.0)


def test_blue_water_vectorized_identity_holds_for_finite_pixels() -> None:
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

    blue_water = model._compute_blue_water_requirement(et_potential, et_actual)

    finite_pixels = np.isfinite(et_potential) & np.isfinite(et_actual)
    assert np.all(blue_water[finite_pixels] >= 0.0)
    np.testing.assert_allclose(et_actual[finite_pixels] + blue_water[finite_pixels], et_potential[finite_pixels])


def test_physical_irrigation_mode_raises_not_implemented() -> None:
    crop = Crop(
        name="test_crop",
        root_max_depth=1.0,
        kc_initial=0.3,
        kc_max=1.0,
        kc_sen=0.4,
        p=0.5,
        crop_raster=np.ones((1, 1), dtype=np.float32),
    )
    soil = Soil(
        field_capacity=np.array([[0.5]], dtype=np.float32),
        wilting_point=np.array([[0.4]], dtype=np.float32),
        max_infiltration_rate=np.array([[10.0]], dtype=np.float32),
        texture_map=np.array([[1]], dtype=np.uint8),
        soil_moisture=np.array([[0.5]], dtype=np.float32),
    )
    inputs = IWRInputs(
        scheduler=None,
        output_dir=".",
        crop=crop,
        soil=soil,
        phenology=None,
        working_grid=np.ones((1, 1), dtype=np.uint8),
        report_blue_water_requirement=True,
        simulate_irrigation=True,
    )

    with pytest.raises(NotImplementedError, match="Physical irrigation simulation is not implemented"):
        IWRModel(inputs)
