from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio


def _extract_iwr_datetime(path: str | Path):
    """Return a pandas timestamp from filenames like iwr_2024_01_31.tif."""
    stem = Path(path).stem
    parts = stem.split("_")
    if len(parts) < 4:
        return None
    try:
        year, month, day = parts[1], parts[2], parts[3]
        return pd.Timestamp(f"{year}-{month}-{day}")
    except ValueError:
        return None


def _period_key(date: pd.Timestamp, aggregation_window: str):
    if aggregation_window == "month":
        return (date.year, date.month)
    if aggregation_window == "dekad":
        return (date.year, date.month, (date.day - 1) // 10 + 1)
    raise ValueError("aggregation_window must be 'month' or 'dekad'")


def _period_label(key, aggregation_window: str):
    if aggregation_window == "month":
        return f"{key[1]:02d}"
    if aggregation_window == "dekad":
        _, month, dekad = key
        return f"{month:02d}_dekad{dekad}"
    raise ValueError("aggregation_window must be 'month' or 'dekad'")


def _extract_year_from_name(filename: str) -> int | None:
    """Extract a 4-digit year from a raster filename."""
    match = re.search(r"(?<!\d)(\d{4})(?!\d)", filename)
    if match is None:
        return None
    return int(match.group(1))


def _pixel_area_m2(transform: rasterio.Affine) -> float:
    """Return the pixel area in m^2 for a georeferenced raster."""
    if transform is None:
        return 1.0
    x_res = abs(transform.a)
    y_res = abs(transform.e)
    if x_res == 0.0 or y_res == 0.0:
        return 1.0
    return float(x_res * y_res)


def summarize_yearly_iwr(
    iwr_folder: str | Path,
    crop_raster: np.ndarray | str | Path | None = None,
) -> pd.DataFrame:
    """Summarize annual IWR totals for all valid pixels in a folder of rasters.

    Each valid raster value is interpreted as mm of irrigation water requirement.
    When a crop-fraction raster is provided, each pixel is weighted by its crop
    fraction before conversion to m3 and then to million m3 for reporting.
    """
    folder = Path(iwr_folder)
    if not folder.exists():
        raise FileNotFoundError(f"IWR folder does not exist: {folder}")

    if crop_raster is not None:
        if isinstance(crop_raster, (str, Path)):
            with rasterio.open(crop_raster) as src:
                crop_raster = src.read(1).astype(np.float64)
        else:
            crop_raster = np.asarray(crop_raster, dtype=np.float64)

        if crop_raster.ndim != 2:
            raise ValueError("crop_raster must be a 2D array or path to a raster.")

    year_totals: dict[int, float] = {}

    for raster_path in sorted(folder.rglob("*.tif")):
        name = raster_path.name.lower()
        if "iwr" not in name:
            continue

        year = _extract_year_from_name(raster_path.name)
        if year is None:
            continue

        with rasterio.open(raster_path) as src:
            data = src.read(1).astype(np.float64)
            valid = np.isfinite(data)
            if not np.any(valid):
                continue

            if crop_raster is not None:
                if crop_raster.shape != data.shape:
                    raise ValueError(
                        "crop_raster shape does not match IWR raster shape: "
                        f"{crop_raster.shape} != {data.shape}"
                    )
                valid = valid & np.isfinite(crop_raster) & (crop_raster > 0.0)
                weighted_data = data[valid] * crop_raster[valid]
            else:
                weighted_data = data[valid]

            pixel_area = _pixel_area_m2(src.transform)
            total_m3 = float(np.sum(weighted_data) * 1e-3 * pixel_area)

        year_totals[year] = year_totals.get(year, 0.0) + total_m3

    if not year_totals:
        raise ValueError(
            f"No IWR rasters with year information were found under: {folder}"
        )

    df = pd.DataFrame(
        {
            "year": sorted(year_totals),
            "m3": [year_totals[year] for year in sorted(year_totals)],
        }
    )
    df["million_m3"] = df["m3"] / 1_000_000.0
    return df


def plot_yearly_iwr(
    iwr_folder: str | Path,
    plot_out_folder: str | Path,
    reference_value: float | None = None,
    filename: str = "yearly_iwr.png",
    title: str = "Yearly IWR",
    show: bool = False,
    dpi: int = 200,
    crop_name: str | None = None,
    crop_raster: np.ndarray | str | Path | None = None,
) -> Path:
    """Create a yearly IWR bar plot with an optional red reference line.

    Parameters
    ----------
    iwr_folder : str or Path
        Folder containing IWR GeoTIFF rasters.
    plot_out_folder : str or Path
        Folder where the output plot will be saved.
    reference_value : float, optional
        Constant reference value in million m3. If provided it is shown as a
        dashed red horizontal line.
    filename : str, optional
        Name of the PNG file to save.
    title : str, optional
        Plot title.
    show : bool, optional
        Whether to show the plot interactively.
    dpi : int, optional
        Output resolution in dots per inch.
    crop_name : str, optional
        Crop name to include in the default filename and title.

    Returns
    -------
    pathlib.Path
        Path to the saved PNG figure.
    """
    summary = summarize_yearly_iwr(iwr_folder, crop_raster=crop_raster)
    output_dir = Path(plot_out_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    if crop_name:
        crop_slug = crop_name.strip().lower().replace(" ", "_")
        if filename == "yearly_iwr.png":
            filename = f"yearly_iwr_{crop_slug}.png"
        if title == "Yearly IWR":
            title = f"Yearly IWR - {crop_name}"

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(summary["year"].astype(str), summary["million_m3"], color="steelblue")

    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel("IWR (million m3)")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    if reference_value is not None:
        ax.axhline(
            reference_value,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Reference: {reference_value:.2f} million m3",
        )
        ax.legend()

    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + max(summary["million_m3"]) * 0.02,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.tight_layout()
    plot_path = output_dir / filename
    fig.savefig(plot_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    if show:
        plt.show()

    return plot_path


def plot_iwr_vs_climatology(
    mean_folder: str | Path,
    q1_folder: str | Path,
    q3_folder: str | Path,
    iwr_path: str | Path,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    aggregation_window: str = "dekad",
    output_folder: str | Path | None = None,
    filename: str = "iwr_vs_climatology.png",
    title: str | None = None,
    show: bool = False,
    dpi: int = 200,
) -> Path:
    """Plot a selected year against the climatology mean and IQR bands.

    Parameters
    ----------
    mean_folder, q1_folder, q3_folder : str or Path
        Folders containing the climatology rasters named like
        iwr_mean_01_dekad1.tif, iwr_q1_01_dekad1.tif, and iwr_q3_01_dekad1.tif.
    iwr_path : str or Path
        Folder or raster file containing the actual IWR rasters to evaluate.
    start_date, end_date : str or pandas.Timestamp
        Date range of the target year or segment to compare.
    aggregation_window : {'month', 'dekad'}
        Temporal step used to align the comparison with the climatology files.
    output_folder : str or Path, optional
        Destination folder for the PNG. Defaults to the current directory.
    filename : str
        PNG filename to save.
    title : str, optional
        Optional title for the plot.
    show : bool
        Whether to show the figure interactively.
    dpi : int
        PNG resolution.

    Returns
    -------
    pathlib.Path
        Path to the saved plot image.
    """
    if aggregation_window not in {"month", "dekad"}:
        raise ValueError("aggregation_window must be 'month' or 'dekad'")

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    if end_ts < start_ts:
        raise ValueError("end_date must be on or after start_date")

    mean_folder = Path(mean_folder)
    q1_folder = Path(q1_folder)
    q3_folder = Path(q3_folder)

    iwr_path = Path(iwr_path)
    if iwr_path.is_dir():
        paths = []
        for candidate in sorted(iwr_path.rglob("iwr_*.tif")):
            dt = _extract_iwr_datetime(candidate)
            if dt is not None and start_ts <= dt <= end_ts:
                paths.append(candidate)
    elif iwr_path.is_file():
        paths = [iwr_path]
    else:
        raise FileNotFoundError(f"IWR path does not exist: {iwr_path}")

    if not paths:
        raise FileNotFoundError(
            f"No IWR rasters found between {start_ts.date()} and {end_ts.date()} in {iwr_path}."
        )

    actual_periods = {}
    for path in paths:
        dt = _extract_iwr_datetime(path)
        if dt is None:
            continue
        key = (dt.year, dt.month, (dt.day - 1) // 10 + 1) if aggregation_window == "dekad" else (dt.year, dt.month)
        with rasterio.open(path) as src:
            array = src.read(1).astype(np.float64)
            valid = np.isfinite(array)
            if np.any(valid):
                actual_periods.setdefault(key, []).append(float(np.nanmean(array[valid])))

    if not actual_periods:
        raise ValueError("No valid IWR values were found for the selected period.")

    climatology_series = {"mean": [], "q1": [], "q3": [], "labels": []}
    for key in sorted(actual_periods.keys()):
        label = _period_label(key, aggregation_window)
        climatology_series["labels"].append(label)

        mean_path = next(iter(sorted((mean_folder).glob(f"iwr_mean_{label}.tif"))), None)
        q1_path = next(iter(sorted((q1_folder).glob(f"iwr_q1_{label}.tif"))), None)
        q3_path = next(iter(sorted((q3_folder).glob(f"iwr_q3_{label}.tif"))), None)

        if mean_path is not None:
            with rasterio.open(mean_path) as src:
                arr = src.read(1).astype(np.float64)
                valid = np.isfinite(arr)
                climatology_series["mean"].append(float(np.nanmean(arr[valid])) if np.any(valid) else np.nan)
        else:
            climatology_series["mean"].append(np.nan)

        if q1_path is not None:
            with rasterio.open(q1_path) as src:
                arr = src.read(1).astype(np.float64)
                valid = np.isfinite(arr)
                climatology_series["q1"].append(float(np.nanmean(arr[valid])) if np.any(valid) else np.nan)
        else:
            climatology_series["q1"].append(np.nan)

        if q3_path is not None:
            with rasterio.open(q3_path) as src:
                arr = src.read(1).astype(np.float64)
                valid = np.isfinite(arr)
                climatology_series["q3"].append(float(np.nanmean(arr[valid])) if np.any(valid) else np.nan)
        else:
            climatology_series["q3"].append(np.nan)

    actual_values = []
    for key in sorted(actual_periods.keys()):
        values = actual_periods[key]
        actual_values.append(float(np.nanmean(values)))

    x = np.arange(len(climatology_series["labels"]))
    mean_vals = np.asarray(climatology_series["mean"], dtype=np.float64)
    q1_vals = np.asarray(climatology_series["q1"], dtype=np.float64)
    q3_vals = np.asarray(climatology_series["q3"], dtype=np.float64)
    actual_vals = np.asarray(actual_values, dtype=np.float64)

    output_dir = Path(output_folder) if output_folder is not None else Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / filename

    if title is None:
        title = f"IWR vs climatology ({start_ts.strftime('%Y-%m-%d')} to {end_ts.strftime('%Y-%m-%d')})"

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x, actual_vals, color="tab:blue", marker="o", linewidth=2, label="Actual IWR")
    ax.plot(x, mean_vals, color="tab:orange", linewidth=2, label="Climatology mean")
    ax.fill_between(x, q1_vals, q3_vals, color="tab:orange", alpha=0.25, label="IQR")
    ax.set_xticks(x)
    ax.set_xticklabels(climatology_series["labels"], rotation=45)
    ax.set_title(title)
    ax.set_xlabel(aggregation_window.capitalize())
    ax.set_ylabel("IWR")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    if show:
        plt.show()

    return plot_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot yearly irrigation water requirement (IWR) from GeoTIFF rasters.",
    )
    parser.add_argument(
        "iwr_folder",
        type=str,
        help="Folder containing IWR GeoTIFF rasters.",
    )
    parser.add_argument(
        "plot_out_folder",
        type=str,
        help="Folder where the plot PNG will be saved.",
    )
    parser.add_argument(
        "--reference-value",
        type=float,
        default=None,
        help="Optional constant reference value in million m3, shown as a red dashed line.",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="yearly_iwr.png",
        help="Name of the output PNG file.",
    )
    parser.add_argument(
        "--crop-name",
        type=str,
        default=None,
        help="Optional crop name used in the default output filename and title.",
    )
    args = parser.parse_args()

    plot_yearly_iwr(
        iwr_folder=args.iwr_folder,
        plot_out_folder=args.plot_out_folder,
        reference_value=args.reference_value,
        filename=args.filename,
        crop_name=args.crop_name,
    )


if __name__ == "__main__":
    main()
