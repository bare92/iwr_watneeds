import numpy as np
import rasterio
from rasterio.transform import from_origin

from iwr.phenology import Phenology


def _write_raster(path, array, *, transform):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=array.dtype,
        transform=transform,
    ) as dst:
        dst.write(array, 1)


def test_create_spam_dataset_downloads_and_aligns_layers(tmp_path, monkeypatch):
    reference = np.ones((4, 5), dtype=np.uint8)
    reference_path = tmp_path / "reference.tif"
    ref_transform = from_origin(0, 10, 2, 2)
    _write_raster(reference_path, reference, transform=ref_transform)

    base_url = "https://example.invalid/phenology/"
    source_dir = tmp_path / "source"
    source_dir.mkdir()

    # Create source rasters that differ in size and origin from the reference grid.
    for name in [
        "phenos1_v05.tif",
        "phenos2_v05.tif",
        "phenom1_v05.tif",
        "phenom2_v05.tif",
        "phenoe1_v05.tif",
        "phenoe2_v05.tif",
        "phenosen1_v05.tif",
        "phenosen2_v05.tif",
        "phenonseasons_v05.tif",
    ]:
        array = np.full((2, 3), 1, dtype=np.uint8)
        source_path = source_dir / name
        _write_raster(
            source_path,
            array,
            transform=from_origin(1, 9, 3, 3),
        )

    def fake_download(url, target_path):
        target = str(target_path)
        filename = url.rsplit("/", 1)[-1]
        source_path = source_dir / filename
        if not source_path.exists():
            raise FileNotFoundError(filename)
        with open(source_path, "rb") as src, open(target, "wb") as dst:
            dst.write(src.read())
        return target, None

    monkeypatch.setattr("iwr.phenology.urllib.request.urlretrieve", fake_download)

    output_dir = tmp_path / "output"
    paths = Phenology.create_spam_dataset(str(reference_path), str(output_dir), base_url=base_url)

    assert set(paths) == {
        "phenoe1",
        "phenoe2",
        "phenom1",
        "phenom2",
        "phenonseasons",
        "phenos1",
        "phenos2",
        "phenosen1",
        "phenosen2",
    }

    for layer_name in paths:
        assert paths[layer_name].endswith("_aligned.tif")
        with rasterio.open(paths[layer_name]) as src:
            assert src.shape == reference.shape
            assert src.transform == ref_transform

    phenology = Phenology.from_spam(
        valid_mask=reference,
        phenoe1=rasterio.open(paths["phenoe1"]).read(1),
        phenoe2=rasterio.open(paths["phenoe2"]).read(1),
        phenom1=rasterio.open(paths["phenom1"]).read(1),
        phenom2=rasterio.open(paths["phenom2"]).read(1),
        phenonseasons=rasterio.open(paths["phenonseasons"]).read(1),
        phenos1=rasterio.open(paths["phenos1"]).read(1),
        phenos2=rasterio.open(paths["phenos2"]).read(1),
        phenosen1=rasterio.open(paths["phenosen1"]).read(1),
        phenosen2=rasterio.open(paths["phenosen2"]).read(1),
    )

    assert phenology.state.shape == (36, 4, 5)
