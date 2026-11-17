import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

from iwr.tools import aggregate_fractional


def test_aggregate_fractional_builds_multiband_reference_fraction_grid(tmp_path):
    land_cover = np.array(
        [[1, 1],
         [2, 2]],
        dtype=np.uint8,
    )

    land_cover_path = tmp_path / "land_cover.tif"
    with rasterio.open(
        land_cover_path,
        "w",
        driver="GTiff",
        height=land_cover.shape[0],
        width=land_cover.shape[1],
        count=1,
        dtype=rasterio.uint8,
        crs="EPSG:4326",
        transform=from_origin(0, 2, 1, 1),
    ) as dst:
        dst.write(land_cover, 1)

    reference = np.ones((1, 1), dtype=np.float32)
    reference_path = tmp_path / "reference.tif"
    with rasterio.open(
        reference_path,
        "w",
        driver="GTiff",
        height=reference.shape[0],
        width=reference.shape[1],
        count=1,
        dtype=rasterio.float32,
        crs="EPSG:4326",
        transform=from_origin(0, 2, 2, 2),
    ) as dst:
        dst.write(reference, 1)

    mapping = pd.DataFrame(
        {
            "value": [1, 2],
            "land_cover_name": ["forest", "water"],
        }
    )

    output_path = tmp_path / "fractional_land_cover.tif"
    data, profile = aggregate_fractional(
        land_cover_path,
        mapping,
        reference_path,
        output_path=output_path,
    )

    assert data.shape == (2, 1, 1)
    np.testing.assert_allclose(data[0, 0, 0], 0.5)
    np.testing.assert_allclose(data[1, 0, 0], 0.5)
    assert profile["count"] == 2

    with rasterio.open(output_path) as src:
        output = src.read()

    np.testing.assert_allclose(output[0, 0, 0], 0.5)
    np.testing.assert_allclose(output[1, 0, 0], 0.5)


def test_aggregate_fractional_keeps_zero_as_valid_background_value(tmp_path):
    land_cover = np.array(
        [[0, 1],
         [1, 1]],
        dtype=np.uint8,
    )

    land_cover_path = tmp_path / "land_cover_zero_is_valid.tif"
    with rasterio.open(
        land_cover_path,
        "w",
        driver="GTiff",
        height=land_cover.shape[0],
        width=land_cover.shape[1],
        count=1,
        dtype=rasterio.uint8,
        crs="EPSG:4326",
        transform=from_origin(0, 2, 1, 1),
        nodata=0,
    ) as dst:
        dst.write(land_cover, 1)

    reference = np.ones((1, 1), dtype=np.float32)
    reference_path = tmp_path / "reference.tif"
    with rasterio.open(
        reference_path,
        "w",
        driver="GTiff",
        height=reference.shape[0],
        width=reference.shape[1],
        count=1,
        dtype=rasterio.float32,
        crs="EPSG:4326",
        transform=from_origin(0, 2, 2, 2),
    ) as dst:
        dst.write(reference, 1)

    mapping = pd.DataFrame(
        {
            "value": [0, 1],
            "land_cover_name": ["background", "crop"],
        }
    )

    data = aggregate_fractional(
        land_cover_path,
        mapping,
        reference_path,
    )

    np.testing.assert_allclose(data[0, 0, 0], 0.25)
    np.testing.assert_allclose(data[1, 0, 0], 0.75)
