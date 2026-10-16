from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from rasterio.crs import CRS
from rasterio.warp import transform_bounds

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

    for key in ("PROJ_LIB", "PROJ_DATA", "GDAL_DATA"):
        os.environ.pop(key, None)

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


def aggregate_crops_iwr(
    output_dir: str | Path,
    crop_fractions_path: str | Path,
    scheduler: pd.DataFrame,
    crop_band_map: dict[int, str],
    start_date: str | None = None,
    end_date: str | None = None,
    output_folder_name: str = "aggregated_iwr",
) -> None:
    """
    Aggregate crop-specific IWR rasters as a crop-fraction-weighted average.

    For each pixel and timestep:

        IWR_aggregated =
            sum(fraction_c * IWR_c)
            -----------------------

    The output therefore represents the mean IWR [mm] over the
    cropped/irrigated fraction of the pixel, NOT the equivalent depth
    over the entire pixel.

    Pixels with total crop fraction == 0 are written as NaN.

    Parameters
    ----------
    output_dir : str or Path
        Main output directory containing crop-specific folders.

        Expected structure:

            output_dir/
                maize/
                    iwr/
                        iwr_YYYY_MM_DD_maize.tif
                wheat/
                    iwr/
                        iwr_YYYY_MM_DD_wheat.tif
                ...

    crop_fractions_path : str or Path
        Multiband raster containing crop fractions in the range [0, 1].

    scheduler : pandas.DataFrame
        Scheduler containing a "time" column.

    crop_band_map : dict[int, str]
        Mapping between 1-based raster band number and crop name.

        Example:

            {
                1: "barley",
                5: "maize",
                20: "vineyard",
            }

    start_date, end_date : str, optional
        Optional date limits.

    output_folder_name : str
        Name of output folder.
    """

    output_dir = Path(output_dir)
    crop_fractions_path = Path(crop_fractions_path)

    # --------------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------------

    current_scheduler = scheduler.copy()

    current_scheduler["time"] = pd.to_datetime(
        current_scheduler["time"]
    )

    if start_date is not None:
        current_scheduler = current_scheduler[
            current_scheduler["time"] >= pd.Timestamp(start_date)
        ]

    if end_date is not None:
        current_scheduler = current_scheduler[
            current_scheduler["time"] <= pd.Timestamp(end_date)
        ]

    if current_scheduler.empty:
        raise ValueError(
            "No timesteps remain after applying the requested date range."
        )

    # --------------------------------------------------------------
    # Read crop fractions
    # --------------------------------------------------------------

    with rasterio.open(crop_fractions_path) as src:

        fractions = src.read().astype(np.float32)

        profile = src.profile.copy()

        fraction_shape = (src.height, src.width)
        fraction_transform = src.transform
        fraction_crs = src.crs

        nodata = src.nodata

    if nodata is not None:
        fractions[fractions == nodata] = np.nan

    fractions = _normalize_crop_fractions(fractions)

    # --------------------------------------------------------------
    # Validate fractions
    # --------------------------------------------------------------

    finite = np.isfinite(fractions)

    if np.any(fractions[finite] < 0.0):
        raise ValueError(
            "crop fraction raster contains values < 0."
        )

    if np.any(fractions[finite] > 1.0 + 1e-6):
        raise ValueError(
            "crop fraction raster contains values > 1. "
            "Expected fractions in [0, 1]."
        )

    if max(crop_band_map) > fractions.shape[0]:
        raise ValueError(
            "crop_band_map refers to a band that does not exist "
            "in crop_fractions_path."
        )

    # --------------------------------------------------------------
    # Output directory
    # --------------------------------------------------------------

    aggregated_folder = (
        output_dir / output_folder_name
    )

    aggregated_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    profile.update(
        driver="GTiff",
        dtype=rasterio.float32,
        count=1,
        compress="lzw",
        nodata=np.nan,
    )

    # --------------------------------------------------------------
    # Loop over timesteps
    # --------------------------------------------------------------

    for _, row in current_scheduler.iterrows():

        timestamp = pd.Timestamp(row["time"])

        formatted_time = timestamp.strftime(
            "%Y_%m_%d"
        )

        # Numerator:
        #
        #     sum(fraction * IWR)
        weighted_iwr_sum = np.zeros(
            fraction_shape,
            dtype=np.float32,
        )

        # Denominator:
        #
        #     sum(fraction)
        fraction_sum = np.zeros(
            fraction_shape,
            dtype=np.float32,
        )

        crops_used = 0

        # ----------------------------------------------------------
        # Loop over crops
        # ----------------------------------------------------------

        for band_number, crop_name in crop_band_map.items():

            # rasterio bands are 1-based, numpy arrays are 0-based
            band_index = band_number - 1

            crop_fraction = fractions[band_index]

            iwr_path = (
                output_dir
                / crop_name
                / "iwr"
                / f"iwr_{formatted_time}_{crop_name}.tif"
            )

            if not iwr_path.exists():
                print(
                    f"WARNING: missing IWR raster: {iwr_path}"
                )
                continue

            with rasterio.open(iwr_path) as src:

                # --------------------------------------------------
                # Grid consistency
                # --------------------------------------------------

                if src.shape != fraction_shape:
                    raise ValueError(
                        f"Shape mismatch for {iwr_path}: "
                        f"{src.shape} != {fraction_shape}"
                    )

                if src.transform != fraction_transform:
                    raise ValueError(
                        f"Transform mismatch for {iwr_path}"
                    )

                if src.crs != fraction_crs:
                    raise ValueError(
                        f"CRS mismatch for {iwr_path}"
                    )

                iwr_data = src.read(1).astype(
                    np.float32
                )

                iwr_nodata = src.nodata

            if iwr_nodata is not None:
                iwr_data[
                    iwr_data == iwr_nodata
                ] = np.nan

            # ------------------------------------------------------
            # Valid contribution
            # ------------------------------------------------------

            valid = (
                np.isfinite(crop_fraction)
                & np.isfinite(iwr_data)
                & (crop_fraction > 0.0)
            )

            # Numerator
            weighted_iwr_sum[valid] += (
                crop_fraction[valid]
                * iwr_data[valid]
            )

            # Denominator
            fraction_sum[valid] += (
                crop_fraction[valid]
            )

            crops_used += 1

        # ----------------------------------------------------------
        # Weighted average
        # ----------------------------------------------------------

        aggregated_iwr = np.full(
            fraction_shape,
            np.nan,
            dtype=np.float32,
        )

        valid_pixels = fraction_sum > 0.0

        aggregated_iwr[valid_pixels] = (
            weighted_iwr_sum[valid_pixels]
            / fraction_sum[valid_pixels]
        )

        # ----------------------------------------------------------
        # Save
        # ----------------------------------------------------------

        output_path = (
            aggregated_folder
            / f"iwr_{formatted_time}.tif"
        )

        with rasterio.open(
            output_path,
            "w",
            **profile,
        ) as dst:

            dst.write(
                aggregated_iwr,
                1,
            )

        print(
            f"Saved weighted-average IWR: "
            f"{output_path} "
            f"({crops_used} crops)"
        )

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