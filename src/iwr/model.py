from dataclasses import dataclass
import time
from pathlib import Path
import warnings

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
    debug_plot_stride: int = 1
    debug_plot_max_pixels: int = 250_000
    report_blue_water_requirement: bool = True
    simulate_irrigation: bool = False
    irrigation_area_mask: NDArray[np.uint8] | None = None

    # Deprecated compatibility aliases.
    apply_irrigation: bool | None = None
    irrigation_mask: NDArray[np.uint8] | None = None

    def __post_init__(self):
        if self.apply_irrigation is not None:
            warnings.warn(
                "'apply_irrigation' is deprecated; use 'report_blue_water_requirement' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if self.report_blue_water_requirement != self.apply_irrigation and self.report_blue_water_requirement is not True:
                raise ValueError(
                    "Contradictory configuration: apply_irrigation and report_blue_water_requirement specify different values."
                )
            self.report_blue_water_requirement = self.apply_irrigation

        if self.irrigation_mask is not None:
            warnings.warn(
                "'irrigation_mask' is deprecated; use 'irrigation_area_mask' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if self.irrigation_area_mask is not None:
                legacy_mask = np.asarray(self.irrigation_mask, dtype=np.uint8)
                new_mask = np.asarray(self.irrigation_area_mask, dtype=np.uint8)
                if not np.array_equal(legacy_mask, new_mask):
                    raise ValueError(
                        "Contradictory configuration: irrigation_mask and irrigation_area_mask specify different values."
                    )
            self.irrigation_area_mask = self.irrigation_mask


class IWRModel:
    def __init__(self, inputs: IWRInputs):
        self.inputs = inputs
        if self.inputs.simulate_irrigation:
            raise NotImplementedError(
                "Physical irrigation simulation is not implemented yet. "
                "simulate_irrigation=True requires a separate irrigation scheduling strategy "
                "for timing, application depth, efficiency, and water-balance coupling."
            )
        self._validate_inputs()

        self.soil_moisture_mm = None
        self.TAW = None
        self.RAW = None
        self.critical_storage_mm = None
        self._last_balance_correction_stats = None
        working_grid = np.asarray(self.inputs.working_grid, dtype=np.float32)
        crop_fraction = np.asarray(self.inputs.crop.crop_raster, dtype=np.float32)
        domain_mask = np.isfinite(working_grid)
        self.crop_domain_mask = domain_mask & (crop_fraction > 0.0)
        if not np.any(self.crop_domain_mask):
            raise ValueError(
                "Empty working-domain intersection with positive crop fractions. "
                "Check the reference raster validity mask and crop fraction raster."
            )
        if self.inputs.debug_plot_stride < 1:
            raise ValueError("debug_plot_stride must be >= 1.")
        if self.inputs.debug_plot_max_pixels < 1:
            raise ValueError("debug_plot_max_pixels must be >= 1.")

        self.valid_mask = self.crop_domain_mask
        self.irrigation_area_mask = self.inputs.irrigation_area_mask
        if self.irrigation_area_mask is not None:
            irrigation_area = np.asarray(self.irrigation_area_mask, dtype=np.uint8)
            if irrigation_area.shape != self.crop_domain_mask.shape:
                raise ValueError("irrigation_area_mask must have the same spatial dimensions as the crop domain.")
            self.irrigation_area_mask = irrigation_area > 0

    def _validate_inputs(self) -> None:
        if not 0.0 <= self.inputs.effective_rainfall <= 1.0:
            raise ValueError("effective_rainfall must be between 0 and 1.")

        if self.inputs.crop.crop_raster.ndim < 2:
            raise ValueError("crop_raster must be at least a 2-dimensional array.")

        if self.inputs.working_grid is not None:
            if self.inputs.working_grid.shape != self.inputs.crop.crop_raster.shape[-2:]:
                raise ValueError("working_grid and crop_raster spatial dimensions do not match.")

        if self.inputs.irrigation_area_mask is not None:
            irrigation_area = np.asarray(self.inputs.irrigation_area_mask, dtype=np.uint8)
            if irrigation_area.shape != self.inputs.working_grid.shape:
                raise ValueError("irrigation_area_mask must use the same spatial grid as the working area.")

        if not np.isfinite(self.inputs.crop.root_max_depth) or self.inputs.crop.root_max_depth <= 0:
            raise ValueError("crop.root_max_depth must be a finite value > 0.")

        if not 0.0 <= self.inputs.crop.p <= 1.0:
            raise ValueError("crop.p must be between 0 and 1.")

        field_capacity = np.asarray(self.inputs.soil.field_capacity, dtype=np.float32)
        wilting_point = np.asarray(self.inputs.soil.wilting_point, dtype=np.float32)
        max_infiltration = np.asarray(self.inputs.soil.max_infiltration_rate, dtype=np.float32)

        valid_soil = np.isfinite(field_capacity) & np.isfinite(wilting_point)
        if not np.any(valid_soil):
            raise ValueError("soil field_capacity and wilting_point contain no valid finite values.")

        if not np.all(np.isfinite(max_infiltration[valid_soil])):
            raise ValueError("soil.max_infiltration_rate contains NaN or infinite values on valid soil pixels.")

        if np.any((field_capacity[valid_soil] < 0.0) | (field_capacity[valid_soil] > 1.0)):
            raise ValueError("soil field_capacity must be a volumetric fraction in the range [0, 1].")

        if np.any((wilting_point[valid_soil] < 0.0) | (wilting_point[valid_soil] > field_capacity[valid_soil])):
            raise ValueError("soil wilting_point must satisfy 0 <= wilting_point < field_capacity <= 1.")

        if np.any(max_infiltration[valid_soil] < 0.0):
            raise ValueError("soil.max_infiltration_rate must be >= 0.")

        # The internal soil-water storage is water above wilting point, in mm.
        if self.inputs.soil.soil_moisture is None:
            self.inputs.soil.soil_moisture = self.inputs.soil.field_capacity

    def run(self):
        run_start = time.perf_counter()
        iwr_path, et_potential_path, et_actual_path, soil_moisture_path = self.create_output_directory()
        print(f"RUN {self.inputs.crop.name} from {self.inputs.scheduler['time'].min()} to {self.inputs.scheduler['time'].max()}")

        # TAW, RAW and critical_storage_mm are static for the current crop + soil combination, in mm.
        self.TAW = self._compute_TAW()
        self.RAW = self._compute_RAW()
        self.critical_storage_mm = self._compute_critical_storage()

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

            # WATNEEDS consumptive blue-water requirement:
            # additional water required to raise stressed ET to potential ET.
            # This is a theoretical demand metric and does not physically irrigate the soil.
            blue_water_mm = self._compute_blue_water_requirement(
                et_potential_mm=et_potential_mm,
                et_actual_mm=et_actual_mm,
            )
            iwr = blue_water_mm.copy()
            if self.inputs.report_blue_water_requirement:
                iwr = self._apply_active_crop_mask_to_iwr(
                    blue_water_mm=iwr,
                    phenology_state=phenology_state,
                )
                if self.inputs.irrigation_area_mask is not None:
                    iwr = self._apply_irrigation_area_mask_to_iwr(
                        iwr=iwr,
                        irrigation_area_mask=self.irrigation_area_mask,
                    )
            else:
                iwr = np.zeros_like(blue_water_mm, dtype=np.float32)
                iwr[~np.isfinite(blue_water_mm)] = np.nan

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
            et_potential_output_path = et_potential_path / f"et_potential_{formatted_time}.tif"
            et_actual_output_path = et_actual_path / f"et_actual_{formatted_time}.tif"
            soil_moisture_output_path = soil_moisture_path / f"soil_moisture_{formatted_time}.tif"
            self._write_raster(iwr_output_path, iwr, row["p_path"])
            self._write_raster(et_potential_output_path, et_potential_mm, row["pet_path"])
            self._write_raster(et_actual_output_path, et_actual_mm, row["pet_path"])
            self._write_raster(soil_moisture_output_path, theta_new, row["p_path"])

            if self.inputs.debug_mode and index % self.inputs.debug_plot_stride == 0:
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
        # RAW = p * TAW, both in mm. This is the FAO-56 depletion threshold,
        # not the remaining storage at which stress begins.
        if self.TAW is None:
            self.TAW = self._compute_TAW()
        raw = self.inputs.crop.p * self.TAW
        return np.asarray(raw, dtype=np.float32)

    def _compute_critical_storage(self) -> NDArray[np.float32]:
        # critical_storage_mm = TAW - RAW = (1 - p) * TAW.
        # This is the remaining soil water storage at the onset of stress.
        if self.TAW is None:
            self.TAW = self._compute_TAW()
        taw = np.asarray(self.TAW, dtype=np.float32)
        critical_storage_mm = np.full_like(taw, np.nan, dtype=np.float32)
        valid = np.isfinite(taw)
        critical_storage_mm[valid] = (1.0 - self.inputs.crop.p) * taw[valid]
        if not np.any(np.isfinite(critical_storage_mm)):
            raise ValueError("critical_storage_mm contains no valid finite values for the current crop domain.")
        return np.asarray(critical_storage_mm, dtype=np.float32)

    def _compute_TAW(self) -> NDArray[np.float32]:
        # TAW = (FC - WP) * root_depth * 1000, in mm.
        field_capacity = np.asarray(self.inputs.soil.field_capacity, dtype=np.float32)
        wilting_point = np.asarray(self.inputs.soil.wilting_point, dtype=np.float32)
        root_depth_m = np.asarray(self.inputs.crop.root_max_depth, dtype=np.float32)

        valid = np.isfinite(field_capacity) & np.isfinite(wilting_point) & (field_capacity > wilting_point)
        taw = np.full_like(field_capacity, np.nan, dtype=np.float32)
        taw[valid] = (field_capacity[valid] - wilting_point[valid]) * root_depth_m * 1000.0

        if not np.any(np.isfinite(taw)):
            raise ValueError("Soil parameters produced no valid finite TAW values.")
        if np.any(np.isfinite(taw) & (taw <= 0.0)):
            raise ValueError("TAW must be > 0 for a valid soil-water stress calculation on valid soil pixels.")
        return np.asarray(taw, dtype=np.float32)

    def _compute_ks(self) -> NDArray[np.float32]:
        # Ks = 1 if S >= critical_storage_mm; otherwise Ks = S / critical_storage_mm.
        # The onset-of-stress threshold is the remaining storage, not the depletion amount.
        if self.soil_moisture_mm is None:
            if self.inputs.soil.soil_moisture is None:
                raise ValueError("soil_moisture must be initialized before computing Ks.")
            self.soil_moisture_mm = self._volumetric_to_storage(self.inputs.soil.soil_moisture)
            self.soil_moisture_mm = np.clip(self.soil_moisture_mm, 0.0, self.TAW)

        if self.critical_storage_mm is None:
            self.critical_storage_mm = self._compute_critical_storage()

        soil_storage = np.asarray(self.soil_moisture_mm, dtype=np.float32)
        critical_storage = np.asarray(self.critical_storage_mm, dtype=np.float32)
        ks = np.ones_like(soil_storage, dtype=np.float32)
        valid = np.isfinite(soil_storage) & np.isfinite(critical_storage)
        stress_pixels = valid & (critical_storage > 0.0) & (soil_storage < critical_storage)
        ks[stress_pixels] = soil_storage[stress_pixels] / critical_storage[stress_pixels]
        ks = np.clip(ks, 0.0, 1.0)
        ks[~np.isfinite(soil_storage)] = np.nan
        return ks

    def _compute_deep_percolation(self) -> NDArray[np.float32]:
        # D = Fmax * (S - critical_storage_mm) / (TAW - critical_storage_mm)
        # when S >= critical_storage_mm; otherwise D = 0.
        if self.soil_moisture_mm is None:
            if self.inputs.soil.soil_moisture is None:
                raise ValueError("soil_moisture must be initialized before computing deep percolation.")
            self.soil_moisture_mm = self._volumetric_to_storage(self.inputs.soil.soil_moisture)
            self.soil_moisture_mm = np.clip(self.soil_moisture_mm, 0.0, self.TAW)

        if self.critical_storage_mm is None:
            self.critical_storage_mm = self._compute_critical_storage()

        soil_storage = np.asarray(self.soil_moisture_mm, dtype=np.float32)
        critical_storage = np.asarray(self.critical_storage_mm, dtype=np.float32)
        taw = np.asarray(self.TAW, dtype=np.float32)
        deep_percolation = np.zeros_like(soil_storage, dtype=np.float32)

        denominator = taw - critical_storage
        valid = np.isfinite(soil_storage) & np.isfinite(critical_storage) & np.isfinite(denominator)
        valid &= denominator > 0.0
        valid &= soil_storage >= critical_storage
        valid &= soil_storage <= taw

        deep_percolation[valid] = (
            self.inputs.soil.max_infiltration_rate[valid]
            * (soil_storage[valid] - critical_storage[valid])
            / denominator[valid]
        )
        deep_percolation = np.maximum(deep_percolation, 0.0)
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

    def _apply_active_crop_mask_to_iwr(
        self,
        blue_water_mm: NDArray[np.float32],
        phenology_state: NDArray[np.uint8],
    ) -> NDArray[np.float32]:
        blue_water_mm = np.asarray(blue_water_mm, dtype=np.float32)
        phenology_state = np.asarray(phenology_state, dtype=np.uint8)

        if blue_water_mm.shape != phenology_state.shape:
            raise ValueError(
                "blue_water_mm and phenology_state must have the same shape for seasonal IWR masking."
            )

        active_crop_mask = phenology_state > 0
        reported_iwr = blue_water_mm.copy()
        finite_pixels = np.isfinite(reported_iwr)
        reported_iwr[finite_pixels & (~active_crop_mask)] = 0.0
        reported_iwr[~finite_pixels] = np.nan
        return reported_iwr

    def _apply_irrigation_area_mask_to_iwr(
        self,
        iwr: NDArray[np.float32],
        irrigation_area_mask: NDArray[np.uint8],
    ) -> NDArray[np.float32]:
        iwr = np.asarray(iwr, dtype=np.float32)
        irrigation_area_mask = np.asarray(irrigation_area_mask, dtype=bool)

        if iwr.shape != irrigation_area_mask.shape:
            raise ValueError(
                "iwr and irrigation_area_mask must have the same shape for irrigated-demand masking."
            )

        reported_iwr = iwr.copy()
        finite_pixels = np.isfinite(reported_iwr)
        reported_iwr[finite_pixels & (~irrigation_area_mask)] = 0.0
        reported_iwr[~finite_pixels] = np.nan
        return reported_iwr

    def _compute_blue_water_requirement(
        self,
        et_potential_mm: NDArray[np.float32],
        et_actual_mm: NDArray[np.float32],
        apply_irrigation: bool | None = None,
    ) -> NDArray[np.float32]:
        if apply_irrigation is not None:
            warnings.warn(
                "'apply_irrigation' is deprecated for _compute_blue_water_requirement(); "
                "the function calculates theoretical blue-water demand only.",
                DeprecationWarning,
                stacklevel=2,
            )

        et_potential_mm = np.asarray(et_potential_mm, dtype=np.float32)
        et_actual_mm = np.asarray(et_actual_mm, dtype=np.float32)

        blue_water_mm = np.full_like(et_potential_mm, np.nan, dtype=np.float32)
        finite_pixels = np.isfinite(et_potential_mm) & np.isfinite(et_actual_mm)
        blue_water_mm[finite_pixels] = np.maximum(0.0, et_potential_mm[finite_pixels] - et_actual_mm[finite_pixels])

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
        ts = pd.Timestamp(time_value)
        month_lengths = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        day_index = sum(month_lengths[: ts.month - 1]) + ts.day - 1
        max_day = self.inputs.phenology.Kc_matrix.values.shape[0] - 1
        day_index = min(max(day_index, 0), max_day)
        return self.inputs.phenology.Kc_matrix.values[day_index].astype(np.float32)

    def _get_phenology_state_for_time(
        self, time_value
    ) -> NDArray[np.uint8]:
        ts = pd.Timestamp(time_value)

        # Three calendar dekads per month.
        # The final dekad includes every day from the 21st onward.
        dekad_index = (
            3 * (ts.month - 1)
            + min((ts.day - 1) // 10, 2)
        )

        return self.inputs.phenology.state.values[
            dekad_index
        ].astype(np.uint8)

    @staticmethod
    def _prepare_debug_plot_array(
        array: NDArray[np.float32] | NDArray[np.uint8],
        max_pixels: int = 250_000,
    ) -> NDArray[np.float32] | NDArray[np.uint8]:
        arr = np.asarray(array)
        if arr.ndim != 2 or arr.size <= max_pixels:
            return arr

        step = max(1, int(np.ceil(np.sqrt(arr.size / max_pixels))))
        if step == 1:
            return arr
        return arr[::step, ::step]

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
        debug_dir = Path(self.inputs.output_dir) / self.inputs.crop.name / "debug_png"
        debug_dir.mkdir(parents=True, exist_ok=True)

        max_pixels = max(1, self.inputs.debug_plot_max_pixels)
        fig, axes = plt.subplots(2, 4, figsize=(20, 10), constrained_layout=True)
        axes = axes.ravel()

        plots = [
            ("precipitation (mm)", self._prepare_debug_plot_array(precipitation_mm, max_pixels), "Blues"),
            ("kc (-)", self._prepare_debug_plot_array(kc_map, max_pixels), "viridis"),
            ("phenology_state", self._prepare_debug_plot_array(phenology_state, max_pixels), "phenology"),
            ("et0 (mm)", self._prepare_debug_plot_array(et0_mm, max_pixels), "YlOrRd"),
            ("actual_et (mm)", self._prepare_debug_plot_array(et_actual_mm, max_pixels), "OrRd"),
            ("deep_percolation (mm)", self._prepare_debug_plot_array(deep_percolation_mm, max_pixels), "PuBu"),
            ("soil_moisture_end (mm)", self._prepare_debug_plot_array(soil_moisture_mm, max_pixels), "YlGn"),
            ("blue water / IWR (mm)", self._prepare_debug_plot_array(iwr, max_pixels), "RdPu"),
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
        fig.savefig(debug_dir / f"{self.inputs.crop.name}_{day_label}_debug.png", dpi=100, bbox_inches="tight")
        plt.close(fig)

    def create_output_directory(self) -> tuple[Path, Path, Path, Path]:
        output_path = Path(self.inputs.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        curr_crop_path = output_path / self.inputs.crop.name
        curr_crop_path.mkdir(parents=True, exist_ok=True)

        iwr_path = curr_crop_path / "iwr"
        iwr_path.mkdir(parents=True, exist_ok=True)

        et_potential_path = curr_crop_path / "et_potential"
        et_potential_path.mkdir(parents=True, exist_ok=True)

        et_actual_path = curr_crop_path / "et_actual"
        et_actual_path.mkdir(parents=True, exist_ok=True)

        soil_moisture_path = curr_crop_path / "soil_moisture"
        soil_moisture_path.mkdir(parents=True, exist_ok=True)

        return iwr_path, et_potential_path, et_actual_path, soil_moisture_path

