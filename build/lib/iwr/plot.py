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


def summarize_yearly_iwr(iwr_folder: str | Path) -> pd.DataFrame:
    """Summarize annual IWR totals for all valid pixels in a folder of rasters.

    Each valid raster value is interpreted as mm of irrigation water requirement.
    The total volume is calculated as:

        sum(valid_pixels) * pixel_area_m2 * 1e-3  # m3

    and returned as million m3 for plotting and reporting.
    """
    folder = Path(iwr_folder)
    if not folder.exists():
        raise FileNotFoundError(f"IWR folder does not exist: {folder}")

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

            pixel_area = _pixel_area_m2(src.transform)
            total_m3 = float(np.sum(data[valid]) * 1e-3 * pixel_area)

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

    Returns
    -------
    pathlib.Path
        Path to the saved PNG figure.
    """
    summary = summarize_yearly_iwr(iwr_folder)
    output_dir = Path(plot_out_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

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
    args = parser.parse_args()

    plot_yearly_iwr(
        iwr_folder=args.iwr_folder,
        plot_out_folder=args.plot_out_folder,
        reference_value=args.reference_value,
        filename=args.filename,
    )


if __name__ == "__main__":
    main()
