from dataclasses import dataclass
import time
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import rasterio

from iwr.crops import Crop
from iwr.soil import Soil
from iwr.phenology import Phenology
from iwr.tools import save_debug_raster_png


@dataclass
class IWRInputs:
    scheduler: pd.DataFrame
    output_dir: str
    crop: Crop
    soil: Soil
    phenology: Phenology
    working_grid: NDArray[np.uint8]
    effective_rainfall: float = 0.95
    debug_mode: bool = False
    apply_irrigation: bool = True
    irrigation_mask: NDArray[np.uint8] | None = None  # Optional mask for irrigation application


class IWRModel:
    def __init__(self, inputs: IWRInputs):
        self.inputs = inputs
        self._validate_inputs()

        self.soil_moisture_mm = None
        self.TAW = None
        self.RAW = None
        self._last_balance_correction_stats = None
        if inputs.irrigation_mask is not None:
            self.valid_mask = np.logical_and(self.inputs.working_grid, self.inputs.crop.crop_raster > 0)
            self.valid_mask = np.logical_and(self.valid_mask, self.inputs.irrigation_mask > 0)
        else:
            self.valid_mask = np.logical_and(self.inputs.working_grid, self.inputs.crop.crop_raster > 0)

    def _validate_inputs(self) -> None:
        if not 0.0 <= self.inputs.effective_rainfall <= 1.0:
            raise ValueError("effective_rainfall must be between 0 and 1.")

        if self.inputs.crop.crop_raster.ndim < 2:
            raise ValueError("crop_raster must be at least a 2-dimensional array.")

        if self.inputs.working_grid is not None:
            if self.inputs.working_grid.shape != self.inputs.crop.crop_raster.shape[-2:]:
                raise ValueError("working_grid and crop_raster spatial dimensions do not match.")

        if self.inputs.crop.root_max_depth <= 0:
            raise ValueError("crop.root_max_depth must be > 0.")

        if not 0.0 <= self.inputs.crop.p <= 1.0:
            raise ValueError("crop.p must be between 0 and 1.")

        field_capacity = np.asarray(self.inputs.soil.field_capacity, dtype=np.float32)
        wilting_point = np.asarray(self.inputs.soil.wilting_point, dtype=np.float32)
        max_infiltration = np.asarray(self.inputs.soil.max_infiltration_rate, dtype=np.float32)

        if np.any((field_capacity < 0.0) | (field_capacity > 1.0)):
            raise ValueError("soil field_capacity must be a volumetric fraction in the range [0, 1].")

        if np.any((wilting_point < 0.0) | (wilting_point > field_capacity)):
            raise ValueError("soil wilting_point must satisfy 0 <= wilting_point < field_capacity <= 1.")

        if np.any(max_infiltration < 0.0):
            raise ValueError("soil.max_infiltration_rate must be >= 0.")

        # The internal soil-water storage is water above wilting point, in mm.
        if self.inputs.soil.soil_moisture is None:
            self.inputs.soil.soil_moisture = self.inputs.soil.field_capacity

    def run(self):
        run_start = time.perf_counter()
        iwr_path, et_actual_path, soil_moisture_path = self.create_output_directory()
        print(f"RUN {self.inputs.crop.name} from {self.inputs.scheduler['time'].min()} to {self.inputs.scheduler['time'].max()}")

        # TAW and RAW are static for the current crop + soil combination, in mm.
        self.TAW = self._compute_TAW()
        self.RAW = self._compute_RAW()

        self.soil_moisture_mm = self._volumetric_to_storage(self.inputs.soil.soil_moisture)
        self.soil_moisture_mm = np.clip(self.soil_moisture_mm, 0.0, self.TAW)

        for index, row in self.inputs.scheduler.iterrows():
            # Read the current-day forcing rasters using the row values directly.
            total_precipitation_mm = self._read_raster(row["p_path"])
            et0_mm = self._read_raster(row["pet_path"])

            # Effective precipitation is in mm/day.
            effective_precipitation_mm = total_precipitation_mm * self.inputs.effective_rainfall

            # Convert the current volumetric soil moisture to the internal storage state, in mm.
            self.soil_moisture_mm = self._volumetric_to_storage(self.inputs.soil.soil_moisture)
            self.soil_moisture_mm = np.clip(self.soil_moisture_mm, 0.0, self.TAW)

            # Kc is tied to the actual timestamp, not the DataFrame index.
            kc_map = self._get_kc_for_time(row["time"])
            phenology_state = self._get_phenology_state_for_time(row["time"])
            ks = self._compute_ks()

            et_potential_mm = et0_mm * kc_map
            # ETa in mm/day = ET0 * Kc * Ks.
            et_actual_mm = et_potential_mm * ks

            deep_percolation_mm = self._compute_deep_percolation()
            available_water_mm = self.soil_moisture_mm + effective_precipitation_mm
            et_actual_mm, deep_percolation_mm = self._close_negative_water_balance(
                available_water_mm=available_water_mm,
                et_actual_mm=et_actual_mm,
                deep_percolation_mm=deep_percolation_mm,
            )

            soil_storage_candidate = available_water_mm - et_actual_mm - deep_percolation_mm
            tolerance = 1e-6
            finite_soil_storage = np.isfinite(soil_storage_candidate)
            materially_negative = finite_soil_storage & (soil_storage_candidate < -tolerance)
            if np.any(materially_negative):
                min_balance = float(np.nanmin(soil_storage_candidate[materially_negative]))
                raise RuntimeError(
                    "Water balance closure failed after ETa/deep-percolation correction; "
                    f"minimum residual soil storage: {min_balance:.6f} mm"
                )

            small_negative = finite_soil_storage & (soil_storage_candidate < 0.0)
            if np.any(small_negative):
                soil_storage_candidate = soil_storage_candidate.copy()
                soil_storage_candidate[small_negative] = 0.0

            runoff_mm, soil_storage_new = self._compute_runoff_from_excess_storage(
                soil_storage_candidate=soil_storage_candidate,
                taw_mm=self.TAW,
            )

            finite_balance = (
                np.isfinite(soil_storage_candidate)
                & np.isfinite(soil_storage_new)
                & np.isfinite(runoff_mm)
            )
            if np.any(finite_balance):
                if np.any(soil_storage_new[finite_balance] > self.TAW[finite_balance] + tolerance):
                    max_storage = float(np.max(soil_storage_new[finite_balance] - self.TAW[finite_balance]))
                    raise RuntimeError(
                        "Runoff closure failed to enforce TAW; "
                        f"maximum storage exceedance: {max_storage:.6f} mm"
                    )
                if np.any(runoff_mm[finite_balance] < -tolerance):
                    min_runoff = float(np.min(runoff_mm[finite_balance]))
                    raise RuntimeError(
                        "Runoff closure produced negative runoff; "
                        f"minimum runoff: {min_runoff:.6f} mm"
                    )
                runoff_balance_error = np.abs(soil_storage_candidate[finite_balance] - (soil_storage_new[finite_balance] + runoff_mm[finite_balance]))
                if np.any(runoff_balance_error > tolerance):
                    max_balance_error = float(np.max(runoff_balance_error))
                    raise RuntimeError(
                        "Runoff closure failed mass conservation; "
                        f"maximum residual: {max_balance_error:.6f} mm"
                    )

            # AIDA/WATNEEDS consumptive blue-water requirement:
            # additional water required to raise stressed ET to potential ET.
            # This is not a soil-storage refill-to-RAW irrigation depth.
            blue_water_mm = self._compute_blue_water_requirement(
                et_potential_mm=et_potential_mm,
                et_actual_mm=et_actual_mm,
                apply_irrigation=self.inputs.apply_irrigation,
            )
            iwr = blue_water_mm

            if self.inputs.debug_mode and self._last_balance_correction_stats is not None:
                stats = self._last_balance_correction_stats
                print(
                    "Water-balance closure stats "
                    f"for {pd.Timestamp(row['time']).strftime('%Y-%m-%d')}: "
                    f"pixels={stats['pixels_requiring_correction']}, "
                    f"max_original_deficit_mm={stats['max_original_deficit_mm']:.6f}, "
                    f"min_scale_factor={stats['min_scale_factor']:.6f}"
                )
                finite_runoff = np.isfinite(runoff_mm)
                runoff_pixels = int(np.count_nonzero(finite_runoff & (runoff_mm > 0.0)))
                max_runoff = float(np.max(runoff_mm[finite_runoff])) if np.any(finite_runoff) else 0.0
                print(
                    "Runoff stats "
                    f"for {pd.Timestamp(row['time']).strftime('%Y-%m-%d')}: "
                    f"pixels={runoff_pixels}, "
                    f"max_runoff_mm={max_runoff:.6f}"
                )
            self.soil_moisture_mm = soil_storage_new

            # Convert back to volumetric soil moisture for the Soil object.
            theta_new = self._storage_to_volumetric(self.soil_moisture_mm)
            theta_new = np.clip(theta_new, self.inputs.soil.wilting_point, self.inputs.soil.field_capacity)
            self.inputs.soil.soil_moisture = theta_new

            # Raster outputs are written to disk as before.
            formatted_time = pd.Timestamp(row["time"]).strftime("%Y_%m_%d")
            iwr_output_path = iwr_path / f"iwr_{formatted_time}_{self.inputs.crop.name}.tif"
            et_actual_output_path = et_actual_path / f"et_actual_{formatted_time}.tif"
            soil_moisture_output_path = soil_moisture_path / f"soil_moisture_{formatted_time}.tif"
            self._write_raster(iwr_output_path, iwr, row["p_path"])
            self._write_raster(et_actual_output_path, et_actual_mm, row["pet_path"])
            self._write_raster(soil_moisture_output_path, theta_new, row["p_path"])

            if self.inputs.debug_mode:
                self._save_debug_day_plot(
                    timestamp=row["time"],
                    precipitation_mm=effective_precipitation_mm,
                    kc_map=kc_map,
                    phenology_state=phenology_state,
                    et0_mm=et0_mm,
                    et_actual_mm=et_actual_mm,
                    deep_percolation_mm=deep_percolation_mm,
                    soil_moisture_mm=self.soil_moisture_mm,
                    iwr=iwr,
                )

        total_elapsed = time.perf_counter() - run_start
        print(f"Total run time: {total_elapsed:.3f} seconds")

    def _read_raster(self, raster_path: str) -> NDArray[np.float32]:
        with rasterio.open(raster_path) as src:
            data = src.read(1).astype(np.float32)
            nodata = src.nodata

        # Nodata values are invalid and must be converted to NaN regardless of other masks.
        if nodata is not None:
            data[data == nodata] = np.nan

        if self.valid_mask is not None:
            data[self.valid_mask == 0] = np.nan

        return data

    def _write_raster(self, raster_path: str, data: NDArray[np.float32], reference_raster_path: str):
        with rasterio.open(reference_raster_path) as src:
            profile = src.profile.copy()

        profile["driver"] = "GTiff"
        profile.update(
            dtype=rasterio.float32,
            count=1,
            compress="lzw",
            nodata=np.nan,
        )

        with rasterio.open(raster_path, "w", **profile) as dst:
            dst.write(data.astype(np.float32), 1)

    def _volumetric_to_storage(self, theta: NDArray[np.float32] | float) -> NDArray[np.float32]:
        # theta is volumetric moisture [m3/m3]; root_max_depth is in m.
        # S is available water above wilting point, in mm.
        theta = np.asarray(theta, dtype=np.float32)
        root_depth_m = np.asarray(self.inputs.crop.root_max_depth, dtype=np.float32)
        theta_wp = np.asarray(self.inputs.soil.wilting_point, dtype=np.float32)
        return (theta - theta_wp) * root_depth_m * 1000.0

    def _storage_to_volumetric(self, storage_mm: NDArray[np.float32] | float) -> NDArray[np.float32]:
        # S is in mm; convert back to volumetric soil moisture [m3/m3].
        storage_mm = np.asarray(storage_mm, dtype=np.float32)
        root_depth_m = np.asarray(self.inputs.crop.root_max_depth, dtype=np.float32)
        theta_wp = np.asarray(self.inputs.soil.wilting_point, dtype=np.float32)
        return theta_wp + storage_mm / (root_depth_m * 1000.0)

    def _compute_RAW(self) -> NDArray[np.float32]:
        # RAW = p * TAW, both in mm.
        if self.TAW is None:
            self.TAW = self._compute_TAW()
        return self.inputs.crop.p * self.TAW

    def _compute_TAW(self) -> NDArray[np.float32]:
        # TAW = (FC - WP) * root_depth * 1000, in mm.
        return (self.inputs.soil.field_capacity - self.inputs.soil.wilting_point) * self.inputs.crop.root_max_depth * 1000.0

    def _compute_ks(self) -> NDArray[np.float32]:
        # Ks = S / RAW if S < RAW, else Ks = 1; must remain between 0 and 1.
        if self.soil_moisture_mm is None:
            if self.inputs.soil.soil_moisture is None:
                raise ValueError("soil_moisture must be initialized before computing Ks.")
            self.soil_moisture_mm = self._volumetric_to_storage(self.inputs.soil.soil_moisture)
            self.soil_moisture_mm = np.clip(self.soil_moisture_mm, 0.0, self.TAW)

        soil_storage = np.asarray(self.soil_moisture_mm, dtype=np.float32)
        raw = np.asarray(self.RAW, dtype=np.float32)
        ks = np.ones_like(soil_storage, dtype=np.float32)
        valid = np.isfinite(soil_storage) & np.isfinite(raw) & (raw > 0.0)
        invalid_mask = valid & (soil_storage < raw)
        ks[invalid_mask] = soil_storage[invalid_mask] / raw[invalid_mask]
        ks = np.clip(ks, 0.0, 1.0)
        ks[~np.isfinite(soil_storage)] = np.nan
        return ks

    def _compute_deep_percolation(self) -> NDArray[np.float32]:
        # D = Fmax * (S - RAW)/(TAW - RAW) for RAW <= S <= TAW; D = 0 otherwise.
        if self.soil_moisture_mm is None:
            if self.inputs.soil.soil_moisture is None:
                raise ValueError("soil_moisture must be initialized before computing deep percolation.")
            self.soil_moisture_mm = self._volumetric_to_storage(self.inputs.soil.soil_moisture)
            self.soil_moisture_mm = np.clip(self.soil_moisture_mm, 0.0, self.TAW)

        soil_storage = np.asarray(self.soil_moisture_mm, dtype=np.float32)
        raw = np.asarray(self.RAW, dtype=np.float32)
        taw = np.asarray(self.TAW, dtype=np.float32)
        deep_percolation = np.zeros_like(soil_storage, dtype=np.float32)
        denominator = taw - raw
        valid = np.isfinite(soil_storage) & np.isfinite(denominator)
        valid &= denominator != 0.0
        valid &= soil_storage >= raw
        valid &= soil_storage <= taw

        relative_storage = np.zeros_like(soil_storage, dtype=np.float32)
        relative_storage[valid] = ((soil_storage[valid] - raw[valid]) / denominator[valid])
        relative_storage = np.clip(relative_storage, 0.0, 1.0)
        deep_percolation = self.inputs.soil.max_infiltration_rate * relative_storage
        deep_percolation[~np.isfinite(soil_storage)] = np.nan
        return deep_percolation

    def _close_negative_water_balance(
        self,
        available_water_mm: NDArray[np.float32],
        et_actual_mm: NDArray[np.float32],
        deep_percolation_mm: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        available_water_mm = np.asarray(available_water_mm, dtype=np.float64)
        et_actual_mm = np.asarray(et_actual_mm, dtype=np.float64)
        deep_percolation_mm = np.asarray(deep_percolation_mm, dtype=np.float64)

        available_water_safe = available_water_mm.copy()
        finite_available = np.isfinite(available_water_safe)
        available_water_safe[finite_available] = np.maximum(available_water_safe[finite_available], 0.0)

        total_outflow = et_actual_mm + deep_percolation_mm
        finite_pixels = (
            np.isfinite(available_water_safe)
            & np.isfinite(et_actual_mm)
            & np.isfinite(deep_percolation_mm)
            & np.isfinite(total_outflow)
        )
        oversubscribed = finite_pixels & (total_outflow > available_water_safe) & (total_outflow > 0.0)

        scale_factor = np.ones_like(total_outflow, dtype=np.float64)
        scale_factor[oversubscribed] = available_water_safe[oversubscribed] / total_outflow[oversubscribed]
        scale_factor = np.clip(scale_factor, 0.0, 1.0)

        corrected_et_actual_mm = et_actual_mm * scale_factor
        corrected_deep_percolation_mm = deep_percolation_mm * scale_factor

        corrected_total_outflow = corrected_et_actual_mm + corrected_deep_percolation_mm
        needs_renormalization = oversubscribed & (corrected_total_outflow > available_water_safe)
        if np.any(needs_renormalization):
            renormalization_factor = np.ones_like(total_outflow, dtype=np.float64)
            renormalization_factor[needs_renormalization] = (
                available_water_safe[needs_renormalization]
                / corrected_total_outflow[needs_renormalization]
            )
            corrected_et_actual_mm[needs_renormalization] *= renormalization_factor[needs_renormalization]
            corrected_deep_percolation_mm[needs_renormalization] *= renormalization_factor[needs_renormalization]

        corrected_et_actual_mm[~np.isfinite(et_actual_mm)] = np.nan
        corrected_deep_percolation_mm[~np.isfinite(deep_percolation_mm)] = np.nan

        if np.any(oversubscribed):
            deficits = total_outflow[oversubscribed] - available_water_safe[oversubscribed]
            max_original_deficit_mm = float(np.max(deficits))
            min_scale_factor = float(np.min(scale_factor[oversubscribed]))
        else:
            max_original_deficit_mm = 0.0
            min_scale_factor = 1.0

        self._last_balance_correction_stats = {
            "pixels_requiring_correction": int(np.count_nonzero(oversubscribed)),
            "max_original_deficit_mm": max_original_deficit_mm,
            "min_scale_factor": min_scale_factor,
        }

        return corrected_et_actual_mm, corrected_deep_percolation_mm

    def _compute_blue_water_requirement(
        self,
        et_potential_mm: NDArray[np.float32],
        et_actual_mm: NDArray[np.float32],
        apply_irrigation: bool,
    ) -> NDArray[np.float32]:
        et_potential_mm = np.asarray(et_potential_mm, dtype=np.float32)
        et_actual_mm = np.asarray(et_actual_mm, dtype=np.float32)

        blue_water_mm = np.full_like(et_potential_mm, np.nan, dtype=np.float32)
        finite_pixels = np.isfinite(et_potential_mm) & np.isfinite(et_actual_mm)
        if apply_irrigation:
            blue_water_mm[finite_pixels] = np.maximum(0.0, et_potential_mm[finite_pixels] - et_actual_mm[finite_pixels])
        else:
            blue_water_mm[finite_pixels] = 0.0

        return blue_water_mm

    def _compute_runoff_from_excess_storage(
        self,
        soil_storage_candidate: NDArray[np.float32],
        taw_mm: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        soil_storage_candidate = np.asarray(soil_storage_candidate, dtype=np.float64)
        taw_mm = np.asarray(taw_mm, dtype=np.float64)

        runoff_mm = np.full_like(soil_storage_candidate, np.nan, dtype=np.float64)
        soil_storage_new = np.full_like(soil_storage_candidate, np.nan, dtype=np.float64)
        finite_pixels = np.isfinite(soil_storage_candidate) & np.isfinite(taw_mm)

        runoff_mm[finite_pixels] = np.maximum(0.0, soil_storage_candidate[finite_pixels] - taw_mm[finite_pixels])
        soil_storage_new[finite_pixels] = soil_storage_candidate[finite_pixels] - runoff_mm[finite_pixels]

        return runoff_mm, soil_storage_new

    def _get_kc_for_time(self, time_value) -> NDArray[np.float32]:
        day_of_year = int(pd.Timestamp(time_value).dayofyear)
        max_day = self.inputs.phenology.Kc_matrix.values.shape[0] - 1
        day_index = min(max(day_of_year - 1, 0), max_day)
        return self.inputs.phenology.Kc_matrix.values[day_index].astype(np.float32)

    def _get_phenology_state_for_time(self, time_value) -> NDArray[np.uint8]:
        day_of_year = int(pd.Timestamp(time_value).dayofyear)
        daily_state = self.inputs.phenology._state_to_daily()
        max_day = daily_state.shape[0] - 1
        day_index = min(max(day_of_year - 1, 0), max_day)
        return daily_state[day_index].astype(np.uint8)

    def _save_debug_day_plot(
        self,
        timestamp,
        precipitation_mm: NDArray[np.float32],
        kc_map: NDArray[np.float32],
        phenology_state: NDArray[np.uint8],
        et0_mm: NDArray[np.float32],
        et_actual_mm: NDArray[np.float32],
        deep_percolation_mm: NDArray[np.float32],
        soil_moisture_mm: NDArray[np.float32],
        iwr: NDArray[np.float32],
    ) -> None:
        debug_dir = Path(self.inputs.output_dir) / self.inputs.crop.name /  "debug_png"
        debug_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 4, figsize=(24, 10))
        axes = axes.ravel()

        plots = [
            ("precipitation (mm)", precipitation_mm, "Blues"),
            ("kc (-)", kc_map, "viridis"),
            ("phenology_state", phenology_state, "phenology"),
            ("et0 (mm)", et0_mm, "YlOrRd"),
            ("actual_et (mm)", et_actual_mm, "OrRd"),
            ("deep_percolation (mm)", deep_percolation_mm, "PuBu"),
            ("soil_moisture_end (mm)", soil_moisture_mm, "YlGn"),
            ("blue water / IWR (mm)", iwr, "RdPu"),
        ]

        for ax, (title, data, cmap) in zip(axes, plots):
            if cmap == "phenology":
                array = np.asarray(data, dtype=np.float32)
                masked = np.ma.masked_invalid(array)
                phenology_cmap = ListedColormap(["#f3f4f6", "#65a30d", "#f59e0b", "#b91c1c"])
                phenology_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], phenology_cmap.N)
                image = ax.imshow(masked, cmap=phenology_cmap, norm=phenology_norm)
                colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, ticks=[0, 1, 2, 3])
                colorbar.ax.set_yticklabels(["inactive", "growing", "maximum", "senescence"])
            else:
                array = np.asarray(data, dtype=np.float32)
                masked = np.ma.masked_invalid(array)
                image = ax.imshow(masked, cmap=cmap)
                fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(title)
            ax.set_axis_off()

        day_label = pd.Timestamp(timestamp).strftime("%Y_%m_%d")
        fig.suptitle(f"{self.inputs.crop.name} - {day_label}", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(debug_dir / f"{self.inputs.crop.name}_{day_label}_debug.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def create_output_directory(self) -> tuple[Path, Path, Path]:
        output_path = Path(self.inputs.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        curr_crop_path = output_path / self.inputs.crop.name
        curr_crop_path.mkdir(parents=True, exist_ok=True)

        iwr_path = curr_crop_path / "iwr"
        iwr_path.mkdir(parents=True, exist_ok=True)

        et_actual_path = curr_crop_path / "et_actual"
        et_actual_path.mkdir(parents=True, exist_ok=True)

        soil_moisture_path = curr_crop_path / "soil_moisture"
        soil_moisture_path.mkdir(parents=True, exist_ok=True)

        return iwr_path, et_actual_path, soil_moisture_path

