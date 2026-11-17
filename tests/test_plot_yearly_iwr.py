from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

import plot


def _write_raster(path: Path, values):
    profile = {
        "driver": "GTiff",
        "height": values.shape[0],
        "width": values.shape[1],
        "count": 1,
        "dtype": "float32",
        "transform": from_origin(0, 10000, 5000, 5000),
        "compress": "lzw",
        "nodata": None,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(values.astype("float32"), 1)


def test_plot_yearly_iwr_creates_barplot_and_reference_line(tmp_path):
    iwr_dir = tmp_path / "iwr_data"
    iwr_dir.mkdir()

    year_2022 = np.array([[100.0, 100.0], [100.0, 100.0]], dtype="float32")
    year_2023 = np.array([[50.0, 50.0], [50.0, 50.0]], dtype="float32")

    _write_raster(iwr_dir / "iwr_2022_01_01.tif", year_2022)
    _write_raster(iwr_dir / "iwr_2022_12_31.tif", year_2022)
    _write_raster(iwr_dir / "iwr_2023_01_01.tif", year_2023)

    out_dir = tmp_path / "plots"
    result = plot.plot_yearly_iwr(
        iwr_folder=iwr_dir,
        plot_out_folder=out_dir,
        reference_value=2.0,
    )

    assert result.exists()
    assert result.parent == out_dir
    assert result.name == "yearly_iwr.png"

    summary = plot.summarize_yearly_iwr(iwr_dir)
    assert list(summary["year"]) == [2022, 2023]
    assert summary.loc[summary["year"] == 2022, "million_m3"].iloc[0] > 0


def test_plot_yearly_iwr_uses_crop_fraction_when_summing_volume(tmp_path):
    iwr_dir = tmp_path / "iwr_data"
    iwr_dir.mkdir()

    year_2022 = np.array([[100.0, 100.0], [100.0, 100.0]], dtype="float32")
    crop_fraction = np.array([[0.5, 0.5], [0.5, 0.5]], dtype="float32")

    _write_raster(iwr_dir / "iwr_2022_01_01.tif", year_2022)

    summary = plot.summarize_yearly_iwr(iwr_dir, crop_raster=crop_fraction)

    expected_m3 = float(np.sum(year_2022 * crop_fraction) * 1e-3 * 5000 * 5000)
    np.testing.assert_allclose(summary.loc[summary["year"] == 2022, "m3"].iloc[0], expected_m3)
    np.testing.assert_allclose(summary.loc[summary["year"] == 2022, "million_m3"].iloc[0], expected_m3 / 1_000_000.0)


def test_plot_yearly_iwr_includes_crop_name_in_filename(tmp_path):
    iwr_dir = tmp_path / "iwr_data"
    iwr_dir.mkdir()

    year_2022 = np.array([[100.0, 100.0], [100.0, 100.0]], dtype="float32")
    _write_raster(iwr_dir / "iwr_2022_01_01.tif", year_2022)

    out_dir = tmp_path / "plots"
    result = plot.plot_yearly_iwr(
        iwr_folder=iwr_dir,
        plot_out_folder=out_dir,
        crop_name="grassland",
    )

    assert result.exists()
    assert result.name == "yearly_iwr_grassland.png"
