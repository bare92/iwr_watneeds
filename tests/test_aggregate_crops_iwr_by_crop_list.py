from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from iwr.tools import aggregate_crops_iwr


def test_aggregate_crops_iwr_reads_crop_folders_and_sums_iwr(tmp_path):
    output_dir = tmp_path / "outputs"

    maize_dir = output_dir / "maize" / "iwr"
    wheat_dir = output_dir / "wheat" / "iwr"
    maize_dir.mkdir(parents=True)
    wheat_dir.mkdir(parents=True)

    transform = from_origin(0, 2, 1, 1)
    array_maize = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    array_wheat = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)

    maize_path = maize_dir / "iwr_2024_01_01_maize.tif"
    wheat_path = wheat_dir / "iwr_2024_01_01_wheat.tif"

    for path, data in [(maize_path, array_maize), (wheat_path, array_wheat)]:
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype=rasterio.float32,
            crs="EPSG:4326",
            transform=transform,
            nodata=np.nan,
        ) as dst:
            dst.write(data, 1)

    aggregate_crops_iwr(output_dir=output_dir, crop_names=["maize", "wheat"])

    aggregated_path = output_dir / "aggregated_iwr" / "iwr_2024_01_01.tif"
    assert aggregated_path.exists()

    with rasterio.open(aggregated_path) as src:
        aggregated = src.read(1)

    expected = array_maize + array_wheat
    np.testing.assert_allclose(aggregated, expected)


def test_aggregate_crops_iwr_uses_weighted_average_when_fractions_are_given(tmp_path):
    output_dir = tmp_path / "outputs"

    maize_dir = output_dir / "maize" / "iwr"
    wheat_dir = output_dir / "wheat" / "iwr"
    maize_dir.mkdir(parents=True)
    wheat_dir.mkdir(parents=True)

    transform = from_origin(0, 2, 1, 1)
    maize_iwr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    wheat_iwr = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    maize_frac = np.array([[0.75, 0.25], [0.60, 0.40]], dtype=np.float32)
    wheat_frac = np.array([[0.25, 0.75], [0.40, 0.60]], dtype=np.float32)

    for path, data in [(maize_dir / "iwr_2024_01_01_maize.tif", maize_iwr), (wheat_dir / "iwr_2024_01_01_wheat.tif", wheat_iwr)]:
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype=rasterio.float32,
            crs="EPSG:4326",
            transform=transform,
            nodata=np.nan,
        ) as dst:
            dst.write(data, 1)

    aggregate_crops_iwr(
        output_dir=output_dir,
        model_output_dir=output_dir,
        crop_names=["maize", "wheat"],
        crop_fractions={"maize": maize_frac, "wheat": wheat_frac},
    )

    with rasterio.open(output_dir / "aggregated_iwr" / "iwr_2024_01_01.tif") as src:
        aggregated = src.read(1)

    expected = np.array([[2.0, 5.0], [4.0, 6.0]], dtype=np.float32)
    np.testing.assert_allclose(aggregated, expected)
