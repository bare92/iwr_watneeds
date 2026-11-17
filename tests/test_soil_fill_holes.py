import numpy as np
import pytest

from iwr.soil import Soil
from iwr.soil import _fill_internal_nodata_holes


def test_fill_internal_nodata_holes_fills_nans_inside_valid_mask() -> None:
    raster = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, np.nan, 6.0],
            [7.0, 8.0, 9.0],
        ],
        dtype=np.float32,
    )
    valid_mask = np.ones(raster.shape, dtype=bool)

    filled = _fill_internal_nodata_holes(raster, valid_mask=valid_mask)

    assert np.isfinite(filled[2, 2])
    assert filled[1, 1] in {2.0, 4.0, 6.0, 8.0}


def test_fill_internal_nodata_holes_leaves_nans_outside_valid_mask() -> None:
    raster = np.array(
        [
            [1.0, np.nan, 3.0],
            [4.0, np.nan, 6.0],
            [7.0, 8.0, 9.0],
        ],
        dtype=np.float32,
    )
    valid_mask = np.array(
        [
            [True, False, True],
            [True, True, True],
            [True, True, True],
        ],
        dtype=bool,
    )

    filled = _fill_internal_nodata_holes(raster, valid_mask=valid_mask)

    assert np.isnan(filled[0, 1])
    assert np.isfinite(filled[1, 1])


def test_fill_internal_nodata_holes_raises_when_valid_mask_has_no_finite_sources() -> None:
    raster = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, np.nan, 6.0],
            [7.0, 8.0, 9.0],
        ],
        dtype=np.float32,
    )
    valid_mask = np.zeros(raster.shape, dtype=bool)
    valid_mask[1, 1] = True

    with pytest.raises(ValueError, match="no finite source pixels exist within valid_mask"):
        _fill_internal_nodata_holes(raster, valid_mask=valid_mask)


def test_derive_usda_texture_populates_soil_properties() -> None:
    soil = Soil(
        texture_map=np.zeros((2, 2), dtype=np.uint8),
        field_capacity=np.full((2, 2), np.nan, dtype=np.float32),
        wilting_point=np.full((2, 2), np.nan, dtype=np.float32),
        max_infiltration_rate=np.full((2, 2), np.nan, dtype=np.float32),
        clay_raster=np.array([[10.0, 10.0], [10.0, 10.0]], dtype=np.float32),
        sand_raster=np.array([[50.0, 50.0], [50.0, 50.0]], dtype=np.float32),
        silt_raster=np.array([[40.0, 40.0], [40.0, 40.0]], dtype=np.float32),
    )

    soil.derive_usda_texture()

    assert np.all(np.isfinite(soil.field_capacity))
    assert np.all(np.isfinite(soil.wilting_point))
    assert np.all(np.isfinite(soil.max_infiltration_rate))
    assert np.all(soil.field_capacity == 0.31)