import numpy as np

from iwr.crops import Crop
from iwr.model import IWRInputs, IWRModel
from iwr.soil import Soil


def _make_model(soil_moisture_mm: float, p: float) -> IWRModel:
    crop = Crop(
        name="test_crop",
        root_max_depth=1.0,
        kc_initial=0.3,
        kc_max=1.0,
        kc_sen=0.4,
        p=p,
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
        effective_rainfall=1.0,
    )
    model = IWRModel(inputs)
    model.TAW = np.array([[100.0]], dtype=np.float32)
    model.RAW = np.array([[23.0]], dtype=np.float32)
    model.critical_storage_mm = np.array([[77.0]], dtype=np.float32)
    model.soil_moisture_mm = np.array([[soil_moisture_mm]], dtype=np.float32)
    return model


def test_soil_water_stress_threshold_uses_remaining_storage_not_depletion() -> None:
    model = _make_model(soil_moisture_mm=40.0, p=0.23)

    np.testing.assert_allclose(model.RAW, np.array([[23.0]], dtype=np.float32), rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(model.critical_storage_mm, np.array([[77.0]], dtype=np.float32), rtol=0.0, atol=1e-6)

    ks = model._compute_ks()
    et_potential_mm = np.array([[5.0]], dtype=np.float32)
    et_actual_mm = et_potential_mm * ks
    deep_percolation_mm = model._compute_deep_percolation()
    blue_water_requirement_mm = model._compute_blue_water_requirement(
        et_potential_mm=et_potential_mm,
        et_actual_mm=et_actual_mm,
        apply_irrigation=True,
    )

    np.testing.assert_allclose(ks, np.array([[0.5194805]], dtype=np.float32), rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(et_actual_mm, np.array([[2.5974026]], dtype=np.float32), rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(deep_percolation_mm, np.array([[0.0]], dtype=np.float32), rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(blue_water_requirement_mm, np.array([[2.4025974]], dtype=np.float32), rtol=0.0, atol=1e-6)


def test_deep_percolation_uses_remaining_storage_threshold_when_storage_exceeds_critical_storage() -> None:
    model = _make_model(soil_moisture_mm=80.0, p=0.23)

    ks = model._compute_ks()
    et_potential_mm = np.array([[5.0]], dtype=np.float32)
    et_actual_mm = et_potential_mm * ks
    deep_percolation_mm = model._compute_deep_percolation()

    np.testing.assert_allclose(ks, np.array([[1.0]], dtype=np.float32), rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(et_actual_mm, np.array([[5.0]], dtype=np.float32), rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(
        deep_percolation_mm,
        np.array([[1.3043478]], dtype=np.float32),
        rtol=0.0,
        atol=1e-6,
    )


def test_taw_ignores_unclassified_soil_pixels() -> None:
    crop = Crop(
        name="test_crop",
        root_max_depth=1.0,
        kc_initial=0.3,
        kc_max=1.0,
        kc_sen=0.4,
        p=0.23,
        crop_raster=np.ones((2, 2), dtype=np.float32),
    )
    soil = Soil(
        field_capacity=np.array([[0.5, np.nan], [0.0, 0.6]], dtype=np.float32),
        wilting_point=np.array([[0.4, np.nan], [0.0, 0.3]], dtype=np.float32),
        max_infiltration_rate=np.array([[10.0, 10.0], [10.0, 10.0]], dtype=np.float32),
        texture_map=np.array([[1, 0], [0, 1]], dtype=np.uint8),
        soil_moisture=np.array([[0.5, np.nan], [np.nan, 0.45]], dtype=np.float32),
    )
    inputs = IWRInputs(
        scheduler=None,
        output_dir=".",
        crop=crop,
        soil=soil,
        phenology=None,
        working_grid=np.ones((2, 2), dtype=np.uint8),
        effective_rainfall=1.0,
    )
    model = IWRModel(inputs)

    taw = model._compute_TAW()
    assert np.isfinite(taw[0, 0])
    assert np.isnan(taw[0, 1])
    assert np.isnan(taw[1, 0])
    assert np.isfinite(taw[1, 1])
    np.testing.assert_allclose(taw[0, 0], 100.0, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(taw[1, 1], 300.0, rtol=1e-6, atol=1e-6)


def test_model_valid_mask_excludes_invalid_soil_pixels() -> None:
    crop = Crop(
        name="test_crop",
        root_max_depth=1.0,
        kc_initial=0.3,
        kc_max=1.0,
        kc_sen=0.4,
        p=0.23,
        crop_raster=np.array([[1.0, 1.0]], dtype=np.float32),
    )
    soil = Soil(
        field_capacity=np.array([[0.5, np.nan]], dtype=np.float32),
        wilting_point=np.array([[0.4, np.nan]], dtype=np.float32),
        max_infiltration_rate=np.array([[10.0, 10.0]], dtype=np.float32),
        texture_map=np.array([[1, 0]], dtype=np.uint8),
        soil_moisture=np.array([[0.5, np.nan]], dtype=np.float32),
    )
    inputs = IWRInputs(
        scheduler=None,
        output_dir=".",
        crop=crop,
        soil=soil,
        phenology=None,
        working_grid=np.ones((1, 2), dtype=np.uint8),
        effective_rainfall=1.0,
    )

    model = IWRModel(inputs)

    assert model.valid_mask.shape == (1, 2)
    assert model.valid_mask[0, 0]
    assert not model.valid_mask[0, 1]
