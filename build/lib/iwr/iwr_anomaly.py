from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy import stats


def _extract_iwr_datetime(path: str | Path):
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
        return (date.month,)
    if aggregation_window == "dekad":
        return (date.month, (date.day - 1) // 10 + 1)
    raise ValueError("aggregation_window must be 'month' or 'dekad'")


def _write_raster_like(arr: np.ndarray, template_path: str | Path, output_path: str | Path):
    with rasterio.open(template_path) as src:
        profile = src.profile.copy()

    profile.update(
        driver="GTiff",
        dtype=rasterio.float32,
        count=1,
        compress="lzw",
        nodata=np.nan,
    )

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(arr.astype(np.float32), 1)


def compute_mean_and_interquartile_range(
    data_folder,
    output_folder,
    start_date,
    end_date,
    aggregation_window="dekad",
):
    """Compute climatological mean and q1/q3 for each month or dekad.

    Parameters
    ----------
    data_folder : str or Path
        Folder containing daily IWR rasters named like iwr_YYYY_MM_DD.tif.
    output_folder : str or Path
        Root folder where the statistics are written.
    start_date, end_date : str
        Reference period bounds.
    aggregation_window : {'month', 'dekad'}
        Temporal aggregation used to define the climatological groups.

    Returns
    -------
    dict
        Mapping of statistic names to the folders where rasters were written.
    """
    data_folder = Path(data_folder)
    output_root = Path(output_folder)

    if aggregation_window not in {"month", "dekad"}:
        raise ValueError("aggregation_window must be 'month' or 'dekad'")

    stats_dirs = {
        "mean": output_root / "mean",
        "q1": output_root / "q1",
        "q3": output_root / "q3",
    }
    for stat_dir in stats_dirs.values():
        stat_dir.mkdir(parents=True, exist_ok=True)

    iwr_files = []
    for file_path in sorted(data_folder.rglob("iwr_*.tif")):
        dt = _extract_iwr_datetime(file_path)
        if dt is None:
            continue
        if pd.Timestamp(start_date) <= dt <= pd.Timestamp(end_date):
            iwr_files.append(file_path)

    if not iwr_files:
        raise FileNotFoundError(
            f"No IWR rasters found between {start_date} and {end_date} in {data_folder}."
        )

    grouped_files = {}
    for file_path in iwr_files:
        dt = _extract_iwr_datetime(file_path)
        key = _period_key(dt, aggregation_window)
        grouped_files.setdefault(key, []).append(file_path)

    written = {"mean": [], "q1": [], "q3": []}

    for key, files_in_group in sorted(grouped_files.items()):
        stack = []
        for file_path in files_in_group:
            with rasterio.open(file_path) as src:
                stack.append(src.read(1).astype(np.float32))

        stack = np.stack(stack, axis=0)

        mean_raster = np.nanmean(stack, axis=0)

        q1_raster = np.full(mean_raster.shape, np.nan, dtype=np.float32)
        q3_raster = np.full(mean_raster.shape, np.nan, dtype=np.float32)

        for idx in np.ndindex(mean_raster.shape):
            values = stack[:, idx[0], idx[1]]
            finite_values = values[np.isfinite(values)]
            if finite_values.size == 0:
                continue
            q1_raster[idx] = np.quantile(finite_values, 0.25)
            q3_raster[idx] = np.quantile(finite_values, 0.75)

        if aggregation_window == "month":
            month = key[0]
            label = f"{month:02d}"
            mean_name = f"iwr_mean_{label}.tif"
            q1_name = f"iwr_q1_{label}.tif"
            q3_name = f"iwr_q3_{label}.tif"
        else:
            month, dekad = key
            label = f"{month:02d}_dekad{dekad}"
            mean_name = f"iwr_mean_{label}.tif"
            q1_name = f"iwr_q1_{label}.tif"
            q3_name = f"iwr_q3_{label}.tif"

        template_path = files_in_group[0]
        mean_output = stats_dirs["mean"] / mean_name
        q1_output = stats_dirs["q1"] / q1_name
        q3_output = stats_dirs["q3"] / q3_name

        _write_raster_like(mean_raster, template_path, mean_output)
        _write_raster_like(q1_raster, template_path, q1_output)
        _write_raster_like(q3_raster, template_path, q3_output)

        written["mean"].append(mean_output)
        written["q1"].append(q1_output)
        written["q3"].append(q3_output)

    return written


def _period_label(key, aggregation_window: str):
    if aggregation_window == "month":
        return f"{key[0]:02d}"
    if aggregation_window == "dekad":
        month, dekad = key
        return f"{month:02d}_dekad{dekad}"
    raise ValueError("aggregation_window must be 'month' or 'dekad'")


def _build_reference_groups(data_folder, start_date, end_date, aggregation_window):
    data_folder = Path(data_folder)
    files = []
    for file_path in sorted(data_folder.rglob("iwr_*.tif")):
        dt = _extract_iwr_datetime(file_path)
        if dt is None:
            continue
        if pd.Timestamp(start_date) <= dt <= pd.Timestamp(end_date):
            files.append(file_path)

    if not files:
        raise FileNotFoundError(
            f"No IWR rasters found between {start_date} and {end_date} in {data_folder}."
        )

    grouped = {}
    for file_path in files:
        dt = _extract_iwr_datetime(file_path)
        key = _period_key(dt, aggregation_window)
        grouped.setdefault(key, []).append(file_path)
    return grouped


class IWRAnomaly:
    """Object storing climatology statistics and exposing anomaly calculations."""

    def __init__(
        self,
        data_folder,
        output_folder,
        start_date,
        end_date,
        aggregation_window="dekad",
    ):
        self.data_folder = Path(data_folder)
        self.output_folder = Path(output_folder)
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.aggregation_window = aggregation_window
        self.climatology = {
            "mean": {},
            "q1": {},
            "q3": {},
            "gamma_shape": {},
            "gamma_scale": {},
            "gamma_mean": {},
        }

    def compute_climatology(self):
        """Compute mean/q1/q3 and gamma-fitted climatology for each period."""
        if self.aggregation_window not in {"month", "dekad"}:
            raise ValueError("aggregation_window must be 'month' or 'dekad'")

        stat_dirs = {
            "mean": self.output_folder / "mean",
            "q1": self.output_folder / "q1",
            "q3": self.output_folder / "q3",
            "gamma_shape": self.output_folder / "gamma_shape",
            "gamma_scale": self.output_folder / "gamma_scale",
            "gamma_mean": self.output_folder / "gamma_mean",
        }
        for directory in stat_dirs.values():
            directory.mkdir(parents=True, exist_ok=True)

        summary_files = compute_mean_and_interquartile_range(
            self.data_folder,
            self.output_folder,
            self.start_date,
            self.end_date,
            self.aggregation_window,
        )

        for key_name, raster_paths in summary_files.items():
            for raster_path in raster_paths:
                label = raster_path.stem.replace(f"iwr_{key_name}_", "")
                self.climatology[key_name][label] = raster_path

        grouped_files = _build_reference_groups(
            self.data_folder,
            self.start_date,
            self.end_date,
            self.aggregation_window,
        )

        for key, files_in_group in sorted(grouped_files.items()):
            stack = []
            for file_path in files_in_group:
                with rasterio.open(file_path) as src:
                    stack.append(src.read(1).astype(np.float32))
            stack = np.stack(stack, axis=0)

            gamma_shape_raster = np.full(stack.shape[1:], np.nan, dtype=np.float32)
            gamma_scale_raster = np.full(stack.shape[1:], np.nan, dtype=np.float32)
            gamma_mean_raster = np.full(stack.shape[1:], np.nan, dtype=np.float32)

            for idx in np.ndindex(stack.shape[1:]):
                values = stack[:, idx[0], idx[1]]
                finite_values = values[np.isfinite(values)]
                if finite_values.size < 2 or np.any(finite_values <= 0):
                    continue
                try:
                    shape, loc, scale = stats.gamma.fit(finite_values, floc=0)
                except Exception:
                    continue
                gamma_shape_raster[idx] = float(shape)
                gamma_scale_raster[idx] = float(scale)
                gamma_mean_raster[idx] = float(shape * scale)

            label = _period_label(key, self.aggregation_window)
            shape_path = stat_dirs["gamma_shape"] / f"iwr_gamma_shape_{label}.tif"
            scale_path = stat_dirs["gamma_scale"] / f"iwr_gamma_scale_{label}.tif"
            mean_path = stat_dirs["gamma_mean"] / f"iwr_gamma_mean_{label}.tif"

            template_path = files_in_group[0]
            _write_raster_like(gamma_shape_raster, template_path, shape_path)
            _write_raster_like(gamma_scale_raster, template_path, scale_path)
            _write_raster_like(gamma_mean_raster, template_path, mean_path)

            self.climatology["gamma_shape"][label] = shape_path
            self.climatology["gamma_scale"][label] = scale_path
            self.climatology["gamma_mean"][label] = mean_path

        return self.climatology

    def compute_anomaly(self, observation_file, output_file=None):
        """Compute current period anomaly against the fitted climatology baseline."""
        if not self.climatology.get("gamma_mean"):
            raise ValueError("The climatology has not been computed yet. Call compute_climatology() first.")

        observation_dt = _extract_iwr_datetime(observation_file)
        if observation_dt is None:
            raise ValueError(f"Could not parse a date from {observation_file}.")

        label = _period_label(_period_key(observation_dt, self.aggregation_window), self.aggregation_window)
        gamma_mean_path = self.climatology["gamma_mean"].get(label)
        if gamma_mean_path is None:
            raise KeyError(
                f"No gamma climatology exists for {label} in {self.output_folder}. "
                "Call compute_climatology() first."
            )

        with rasterio.open(observation_file) as src:
            obs = src.read(1).astype(np.float32)
        with rasterio.open(gamma_mean_path) as src:
            clim = src.read(1).astype(np.float32)

        anomaly = obs - clim
        if output_file is not None:
            _write_raster_like(anomaly, observation_file, output_file)
        return anomaly


def compute_gamma_climatology(
    data_folder,
    output_folder,
    start_date,
    end_date,
    aggregation_window="dekad",
):
    """Create and compute a gamma-fitted climatology object."""
    anomaly = IWRAnomaly(
        data_folder=data_folder,
        output_folder=output_folder,
        start_date=start_date,
        end_date=end_date,
        aggregation_window=aggregation_window,
    )
    anomaly.compute_climatology()
    return anomaly


Anomaly = IWRAnomaly

  