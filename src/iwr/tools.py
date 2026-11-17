from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds, reproject


def _clear_raster_env() -> None:
    """Drop stale PROJ/GDAL env vars that can poison Rasterio CRS discovery."""
    for key in ("PROJ_LIB", "PROJ_DATA", "GDAL_DATA"):
        os.environ.pop(key, None)


def save_debug_raster_png(
    raster: np.ndarray,
    name: str,
    folder: str | Path | None = None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
) -> Path:
    """Save a 2D raster array to a PNG file inside the project tmp_img folder.

    Parameters
    ----------
    raster : np.ndarray
        2D NumPy array to visualize.
    name : str
        Base file name (without extension).
    folder : str | Path | None
        Output directory. Defaults to the workspace tmp_img folder.
    cmap : str
        Matplotlib colormap to use.
    vmin, vmax : float | None
        Optional color scaling bounds.

    Returns
    -------
    pathlib.Path
        Full path to the saved PNG.
    """
    if raster is None:
        raise ValueError("raster cannot be None")

    arr = np.asarray(raster)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D raster array, got shape {arr.shape}")

    if folder is None:
        workspace_root = Path(__file__).resolve().parents[2]
        folder = workspace_root / "tmp_img"
    else:
        folder = Path(folder)

    folder.mkdir(parents=True, exist_ok=True)

    safe_name = name.strip().replace(" ", "_")
    output_path = folder / f"{safe_name}.png"

    masked = np.ma.masked_invalid(arr)
    fig, ax = plt.subplots(figsize=(8, 8))
    image = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(safe_name)
    fig.colorbar(image, ax=ax, shrink=0.8)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path


def get_raster_alignment_reference(raster_path: str | Path) -> dict[str, object]:
    raster_path = Path(raster_path)

    if not raster_path.is_file():
        raise FileNotFoundError(raster_path)

    _clear_raster_env()

    with rasterio.open(raster_path) as src:
        if src.crs is None:
            raise ValueError(f"{raster_path} does not have a CRS")

        return {
            "path": raster_path,
            "crs": src.crs,
            "transform": src.transform,
            "resolution": src.res,
            "bounds": src.bounds,
            "width": src.width,
            "height": src.height,
            "nodata": src.nodata,
        }

def get_wgs84_extent_with_buffer(
    reference_georef: dict[str, object],
    buffer_degrees: float = 0.5,
) -> dict[str, object]:
    """
    Convert a raster reference extent to WGS84 and apply a geographic buffer.

    Parameters
    ----------
    reference_georef : dict[str, object]
        Metadata dict returned by get_raster_alignment_reference(...).
        Must contain at least "crs" and "bounds".
    buffer_degrees : float
        Buffer added on all sides in degrees after reprojection to WGS84.

    Returns
    -------
    dict[str, object]
        Dictionary with:
        - xmin, ymin, xmax, ymax in EPSG:4326
        - width, height as raster pixel counts from the reference grid
        - resx, resy as the output pixel size in degrees
        - crs set to EPSG:4326
        - bbox as a tuple (xmin, ymin, xmax, ymax)
    """
    source_crs = reference_georef.get("crs")
    bounds = reference_georef.get("bounds")
    width = reference_georef.get("width")
    height = reference_georef.get("height")

    if source_crs is None:
        raise ValueError("reference_georef['crs'] is required")
    if bounds is None:
        raise ValueError("reference_georef['bounds'] is required")
    if width is None or height is None:
        raise ValueError("reference_georef['width'] and ['height'] are required")

    xmin, ymin, xmax, ymax = transform_bounds(
        source_crs,
        "EPSG:4326",
        bounds.left,
        bounds.bottom,
        bounds.right,
        bounds.top,
        densify_pts=21,
    )

    xmin -= buffer_degrees
    ymin -= buffer_degrees
    xmax += buffer_degrees
    ymax += buffer_degrees

    width = int(width)
    height = int(height)

    resx = (xmax - xmin) / width
    resy = (ymax - ymin) / height

    return {
        "crs": CRS.from_epsg(4326),
        "xmin": xmin,
        "ymin": ymin,
        "xmax": xmax,
        "ymax": ymax,
        "width": width,
        "height": height,
        "resx": resx,
        "resy": resy,
        "bbox": (xmin, ymin, xmax, ymax),
    }

def aggregate_fractional(
    land_cover_path: str | Path,
    class_mapping: pd.DataFrame,
    reference_grid: str | Path,
    output_path: str | Path | None = None,
    value_col: str = "value",
    class_name_col: str = "land_cover_name",
    return_percentage: bool = False,
) -> tuple[np.ndarray, dict] | np.ndarray:
    """Aggregate a higher-resolution land-cover raster to a lower-resolution reference grid.

    The raster is classified as either:

    - categorical/class-presence data (binary masks or integer land-cover codes), or
    - existing fractional data already stored in [0, 1].

    The function retains zero-valued background pixels as valid data and masks only
    true NoData values. The result is always returned on the reference grid and the
    reference-domain mask is applied before writing the output.
    """
    required_columns = {value_col, class_name_col}
    if not isinstance(class_mapping, pd.DataFrame):
        raise TypeError("class_mapping must be a pandas.DataFrame")
    if not required_columns.issubset(class_mapping.columns):
        missing = sorted(required_columns - set(class_mapping.columns))
        raise ValueError(
            f"class_mapping is missing required columns: {missing}"
        )

    land_cover_path = Path(land_cover_path)
    reference_grid = Path(reference_grid)

    if not land_cover_path.is_file():
        raise FileNotFoundError(land_cover_path)
    if not reference_grid.is_file():
        raise FileNotFoundError(reference_grid)

    _clear_raster_env()

    class_df = class_mapping[[value_col, class_name_col]].drop_duplicates().copy()
    class_df[value_col] = pd.to_numeric(class_df[value_col], errors="raise")

    with rasterio.open(land_cover_path) as src:
        source = src.read(1).astype(np.float32)
        source_profile = src.profile.copy()
        source_transform = src.transform
        source_crs = src.crs
        source_nodata = src.nodata

        if source_nodata is not None and not np.isclose(float(source_nodata), 0.0):
            source[source == source_nodata] = np.nan

    with rasterio.open(reference_grid) as ref:
        ref_profile = ref.profile.copy()
        ref_shape = (ref.height, ref.width)
        ref_transform = ref.transform
        ref_crs = ref.crs
        ref_valid = np.isfinite(ref.read(1).astype(np.float32))

    if source_crs != ref_crs:
        raise ValueError(
            f"CRS mismatch between land cover ({source_crs}) and reference grid ({ref_crs})."
        )

    class_values = class_df[value_col].astype(np.float32).tolist()
    class_names = class_df[class_name_col].astype(str).tolist()

    finite_values = np.asarray(source[np.isfinite(source)], dtype=np.float32)
    if finite_values.size == 0:
        raise ValueError("No finite values were found in the land-cover raster.")

    integer_like = np.all(np.isclose(finite_values, np.round(finite_values)))
    use_categorical = integer_like or np.any(
        np.isin(np.unique(finite_values), np.asarray(class_values, dtype=np.float32))
    )

    fractional_stack = np.full(
        (len(class_values), ref_shape[0], ref_shape[1]),
        np.nan,
        dtype=np.float32,
    )

    for band_index, land_cover_value in enumerate(class_values):
        if use_categorical:
            indicator = np.full(source.shape, 0.0, dtype=np.float32)
            valid_source = np.isfinite(source)
            indicator[valid_source & (source == land_cover_value)] = 1.0
            source_for_reproject = indicator
            source_nodata = np.nan
        else:
            source_for_reproject = source.copy()
            source_nodata = np.nan

        destination = np.full(ref_shape, np.nan, dtype=np.float32)
        reproject(
            source=source_for_reproject,
            destination=destination,
            src_transform=source_transform,
            src_crs=source_crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            src_nodata=source_nodata,
            dst_nodata=np.nan,
            resampling=Resampling.average,
        )

        destination[~ref_valid] = np.nan
        if return_percentage:
            destination *= 100.0
        fractional_stack[band_index] = destination

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_profile = ref_profile.copy()
        output_profile.update(
            driver="GTiff",
            count=len(class_values),
            dtype=rasterio.float32,
            nodata=np.nan,
            compress="lzw",
        )

        with rasterio.open(output_path, "w", **output_profile) as dst:
            dst.write(fractional_stack)
            for band_number, description in enumerate(class_names, start=1):
                dst.set_band_description(band_number, str(description))

        return fractional_stack, output_profile

    return fractional_stack


def build_forcing_dataframe(
    precipitation_folder: str,
    pet_folder: str,
    start_date: str,
    end_date: str,
    precipitation_prefix: str = "P_P_sim_",
    pet_prefix: str = "PET_",
) -> pd.DataFrame:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    if end < start:
        raise ValueError("end_date must be on or after start_date")

    dates = pd.date_range(start=start, end=end, freq="D")

    records = []

    for date in dates:
        timestamp = date.strftime("%Y-%m-%d_00_00_00")

        p_path = Path(precipitation_folder) / f"{precipitation_prefix}{timestamp}.tif"
        pet_path = Path(pet_folder) / f"{pet_prefix}{timestamp}.tif"

        if not p_path.exists():
            raise FileNotFoundError(f"Missing precipitation file: {p_path}")

        if not pet_path.exists():
            raise FileNotFoundError(f"Missing PET file: {pet_path}")



        records.append(
            {
                "time": date,
                "p_path": str(p_path),
                "pet_path": str(pet_path),
            }
        )

    return pd.DataFrame.from_records(records)


from pathlib import Path

import numpy as np
import re

import pandas as pd
import rasterio


def _normalize_crop_fractions(fractions: np.ndarray) -> np.ndarray:
    fractions = np.asarray(fractions, dtype=np.float32)
    finite_fraction = np.isfinite(fractions)

    if np.any(fractions[finite_fraction] < 0.0):
        raise ValueError(
            "crop_fractions contains values < 0."
        )

    if np.any(fractions[finite_fraction] > 1.0 + 1e-6):
        if np.all(fractions[finite_fraction] <= 100.0 + 1e-6):
            return fractions / 100.0
        raise ValueError(
            "crop_fractions contains values > 1. "
            "Expected fractions in the range [0, 1]."
        )

    return fractions


from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


def _date_from_iwr_raster_name(path: str | Path) -> str | None:
    """Return YYYY_MM_DD from filenames such as iwr_2024_01_31.tif or iwr_2024_01_31_maize.tif."""
    stem = Path(path).stem
    match = re.search(r"iwr_(\d{4})_(\d{2})_(\d{2})(?:_.*)?$", stem)
    if match is None:
        return None
    return f"{match.group(1)}_{match.group(2)}_{match.group(3)}"


def aggregate_crops_iwr(
    output_dir: str | Path | None = None,
    model_output_dir: str | Path | dict | None = None,
    crop_names: list[str] | None = None,
    crop_fractions: dict[str, str | Path | np.ndarray] | str | Path | None = None,
) -> Path:
    """Aggregate crop-specific IWR rasters to daily whole-domain values.

    Parameters
    ----------
    output_dir : str | Path | None
        Directory where the aggregated rasters should be written.
    model_output_dir : str | Path | dict | None
        Either the root folder containing crop subfolders or a dictionary containing
        the keys ``model_output_dir`` and ``fractional_land_cover``/``crop_fractions``.
    crop_names : list[str] | None
        Crop names to aggregate. If omitted, they are inferred from the fraction dict.
    crop_fractions : dict[str, ...] | str | Path | None
        Fraction raster(s) for each crop. When provided, the aggregation is performed
        as a weighted average using the crop fractions. When omitted, a pure sum is
        returned for compatibility with the existing pipeline.

    Returns
    -------
    Path
        Directory containing the output rasters.
    """
    if isinstance(model_output_dir, dict):
        config = model_output_dir
        if output_dir is None:
            output_dir = config.get("output_dir")
        model_output_dir = config.get("model_output_dir") or config.get("model_output_folder")
        if crop_names is None:
            crop_names = config.get("crop_names")
        if crop_fractions is None:
            crop_fractions = config.get("fractional_land_cover") or config.get("crop_fractions")

    if output_dir is None:
        raise ValueError("output_dir must be provided.")
    if model_output_dir is None:
        raise ValueError("model_output_dir must be provided.")

    output_dir = Path(output_dir)
    model_output_dir = Path(model_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if crop_names is None:
        if crop_fractions is None:
            raise ValueError("crop_names or crop_fractions must be provided.")
        if isinstance(crop_fractions, dict):
            crop_names = list(crop_fractions.keys())
        elif isinstance(crop_fractions, (str, Path)):
            raise ValueError("When crop_fractions is a single raster, crop_names must be provided.")
        else:
            raise TypeError("crop_fractions must be a dict, raster path, or array dictionary.")

    crop_names = [str(name) for name in crop_names]

    if crop_fractions is None:
        crop_fraction_map = {crop_name: None for crop_name in crop_names}
    elif isinstance(crop_fractions, dict):
        crop_fraction_map = {}
        for crop_name, fraction in crop_fractions.items():
            fraction_path_or_array = fraction
            if isinstance(fraction_path_or_array, (str, Path)):
                with rasterio.open(fraction_path_or_array) as src:
                    fraction_array = src.read(1).astype(np.float32)
            elif isinstance(fraction_path_or_array, np.ndarray):
                fraction_array = np.asarray(fraction_path_or_array, dtype=np.float32)
            else:
                raise TypeError(f"Unsupported fraction type for crop {crop_name!r}: {type(fraction_path_or_array)!r}")
            crop_fraction_map[str(crop_name)] = fraction_array
    elif isinstance(crop_fractions, (str, Path)):
        with rasterio.open(crop_fractions) as src:
            if src.count < len(crop_names):
                raise ValueError(
                    "crop_fractions raster has fewer bands than crop_names. "
                    "Expected one band per crop in the same order as crop_names."
                )
        with rasterio.open(crop_fractions) as src:
            band_arrays = [src.read(band_index + 1).astype(np.float32) for band_index in range(src.count)]
        crop_fraction_map = {str(crop_names[index]): band_arrays[index] for index in range(len(crop_names))}
    else:
        raise TypeError("crop_fractions must be a dict, raster path, or None.")

    daily_files: dict[str, dict[str, Path]] = {}
    for crop_name in crop_names:
        crop_iwr_dir = model_output_dir / crop_name / "iwr"
        if not crop_iwr_dir.exists():
            continue
        for file_path in sorted(crop_iwr_dir.glob("iwr_*.tif")):
            date_key = _date_from_iwr_raster_name(file_path)
            if date_key is None:
                continue
            daily_files.setdefault(date_key, {})[crop_name] = file_path

    if not daily_files:
        raise FileNotFoundError(f"No IWR rasters were found under {model_output_dir}.")

    out_dir = output_dir / "aggregated_iwr_crops"
    out_dir.mkdir(parents=True, exist_ok=True)

    for date_key, crop_paths in sorted(daily_files.items()):
        reference_path = next(iter(crop_paths.values()))
        with rasterio.open(reference_path) as src:
            ref_profile = src.profile.copy()
            reference_data = src.read(1).astype(np.float32)

        weighted_sum = np.zeros_like(reference_data, dtype=np.float64)
        weight_total = np.zeros_like(reference_data, dtype=np.float64)
        used_any = False

        for crop_name, crop_path in crop_paths.items():
            with rasterio.open(crop_path) as src:
                crop_data = src.read(1).astype(np.float32)

            if crop_name in crop_fraction_map and crop_fraction_map[crop_name] is not None:
                fraction = np.asarray(crop_fraction_map[crop_name], dtype=np.float32)
                if fraction.shape != crop_data.shape:
                    raise ValueError(
                        f"Fraction raster for crop {crop_name!r} has shape {fraction.shape}, "
                        f"but IWR raster has shape {crop_data.shape}."
                    )
                fraction = np.nan_to_num(fraction, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                fraction = np.ones_like(crop_data, dtype=np.float32)

            if np.any(np.isfinite(crop_data)):
                used_any = True

            crop_weighted = np.nan_to_num(crop_data, nan=0.0, posinf=0.0, neginf=0.0) * fraction
            weighted_sum += crop_weighted
            weight_total += np.nan_to_num(fraction, nan=0.0, posinf=0.0, neginf=0.0)

        if not used_any:
            aggregated = np.full_like(reference_data, np.nan, dtype=np.float32)
        else:
            aggregated = np.divide(
                weighted_sum,
                weight_total,
                out=np.full_like(weighted_sum, np.nan, dtype=np.float64),
                where=weight_total > 0,
            )
            aggregated = np.asarray(aggregated, dtype=np.float32)

        output_path = out_dir / f"iwr_{date_key}.tif"
        profile = ref_profile.copy()
        profile.update(driver="GTiff", count=1, dtype=rasterio.float32, nodata=np.nan, compress="lzw")
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(aggregated, 1)

    return out_dir

from pathlib import Path
import numpy as np
import rasterio


def aggregate_time_iwr(iwr_folder, aggregation_window="dekad"):
    """Aggregate daily IWR rasters over dekads or months.

    Expected input filenames:
        iwr_YYYY_MM_DD.tif

    Aggregation windows:
        dekad 1: days 1-10
        dekad 2: days 11-20
        dekad 3: days 21-end of month

    Output files are named using the last available date of the
    aggregation window, e.g.:

        iwr_2024_01_10.tif
        iwr_2024_01_20.tif
        iwr_2024_01_31.tif

    For monthly aggregation:
        iwr_2024_01_31.tif
    """

    iwr_folder = Path(iwr_folder)
    output_folder = Path(f"{iwr_folder}_{aggregation_window}")
    output_folder.mkdir(parents=True, exist_ok=True)

    if aggregation_window not in ["dekad", "month"]:
        raise ValueError(
            f"Unsupported aggregation window: {aggregation_window}"
        )

    iwr_file_list = sorted(iwr_folder.glob("iwr_*.tif"))

    if not iwr_file_list:
        raise ValueError(f"No IWR files found in {iwr_folder}")

    groups = {}

    for iwr_file in iwr_file_list:

        parts = iwr_file.stem.split("_")

        if len(parts) < 4:
            continue

        try:
            year = int(parts[1])
            month = int(parts[2])
            day = int(parts[3])
        except ValueError:
            continue

        if aggregation_window == "dekad":

            if day <= 10:
                dekad = 1
            elif day <= 20:
                dekad = 2
            else:
                dekad = 3

            key = (year, month, dekad)

        else:
            key = (year, month)

        groups.setdefault(key, []).append(iwr_file)

    # ---------------------------------------------------------
    # Aggregate each time window
    # ---------------------------------------------------------

    for key, files in groups.items():

        aggregated_iwr = None

        # Last available day in this aggregation window
        last_day = max(
            int(path.stem.split("_")[3])
            for path in files
        )

        for file in files:

            with rasterio.open(file) as src:

                data = src.read(1).astype(np.float32)

                if aggregated_iwr is None:
                    aggregated_iwr = np.zeros_like(
                        data,
                        dtype=np.float32,
                    )

                aggregated_iwr += np.nan_to_num(
                    data,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )

        if aggregated_iwr is None:
            continue

        year, month = key[:2]

        output_path = (
            output_folder
            / f"iwr_{year}_{month:02d}_{last_day:02d}.tif"
        )

        with rasterio.open(files[0]) as src:
            profile = src.profile.copy()

        profile.update(
            driver="GTiff",
            dtype=rasterio.float32,
            count=1,
            compress="lzw",
            nodata=np.nan,
        )

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(
                aggregated_iwr.astype(np.float32),
                1,
            )

    return output_folder

def weighted_average_rasters_by_depth(
        raster_paths,
        depth_list_cm,
        nodata_to_nan=True,
    ):
        """
        Load rasters and compute a depth-weighted average.

        Parameters
        ----------
        raster_paths : list[str | Path]
            Paths to rasters ordered the same way as depth_list_cm.
        depth_list_cm : list[float]
            Weights for each raster, typically layer thicknesses in cm.
            Example: [5, 10, 15, 30, 40, 100]
        nodata_to_nan : bool
            If True, convert nodata pixels to NaN before averaging.

        Returns
        -------
        averaged : np.ndarray
            Weighted average raster.
        profile : dict
            Raster profile copied from the first raster.
        """
        if len(raster_paths) != len(depth_list_cm):
            raise ValueError("raster_paths and depth_list_cm must have the same length")

        if len(raster_paths) == 0:
            raise ValueError("raster_paths cannot be empty")

        raster_paths = [Path(path) for path in raster_paths]
        weights = np.asarray(depth_list_cm, dtype=np.float32)

        weighted_sum = None
        weight_sum = None
        profile = None
        ref_shape = None
        ref_transform = None
        ref_crs = None

        for path, weight in zip(raster_paths, weights):
            with rasterio.open(path) as src:
                data = src.read(1).astype(np.float32)

                if profile is None:
                    profile = src.profile.copy()
                    ref_shape = src.shape
                    ref_transform = src.transform
                    ref_crs = src.crs
                    weighted_sum = np.zeros_like(data, dtype=np.float32)
                    weight_sum = np.zeros_like(data, dtype=np.float32)
                else:
                    if src.shape != ref_shape:
                        raise ValueError(f"Shape mismatch for {path}: {src.shape} != {ref_shape}")
                    if src.transform != ref_transform:
                        raise ValueError(f"Transform mismatch for {path}")
                    if src.crs != ref_crs:
                        raise ValueError(f"CRS mismatch for {path}")

                if nodata_to_nan and src.nodata is not None:
                    data[data == src.nodata] = np.nan

                valid = np.isfinite(data)
                weighted_sum[valid] += data[valid] * weight
                weight_sum[valid] += weight

        averaged = np.full_like(weighted_sum, np.nan, dtype=np.float32)
        valid_pixels = weight_sum > 0
        averaged[valid_pixels] = weighted_sum[valid_pixels] / weight_sum[valid_pixels]

        return averaged, profile

