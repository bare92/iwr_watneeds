from dataclasses import dataclass
from os import makedirs
import os
import glob
from numpy.typing import NDArray
import numpy as np
import iwr.tools
from owslib.wcs import WebCoverageService
from rasterio.warp import transform_bounds
import rasterio
from iwr.tools import weighted_average_rasters_by_depth
from rasterio.warp import reproject, Resampling
from scipy.ndimage import distance_transform_edt


SOIL_PARAMETERS = {
    1:  {"name": "Clay heavy",       "wp": 0.39, "fc": 0.54, "fmax": 5},
    2:  {"name": "Silty clay",       "wp": 0.32, "fc": 0.50, "fmax": 12},
    3:  {"name": "Clay",             "wp": 0.39, "fc": 0.54, "fmax": 7},
    4:  {"name": "Silty clay loam",  "wp": 0.23, "fc": 0.44, "fmax": 24},
    5:  {"name": "Clay loam",        "wp": 0.23, "fc": 0.39, "fmax": 24},
    6:  {"name": "Silt",             "wp": 0.09, "fc": 0.33, "fmax": 60},
    7:  {"name": "Silt loam",        "wp": 0.13, "fc": 0.33, "fmax": 82},
    8:  {"name": "Sandy clay",       "wp": 0.27, "fc": 0.39, "fmax": 14},
    9:  {"name": "Loam",             "wp": 0.15, "fc": 0.31, "fmax": 158},
    10: {"name": "Sandy clay loam",  "wp": 0.20, "fc": 0.32, "fmax": 36},
    11: {"name": "Sandy loam",       "wp": 0.10, "fc": 0.22, "fmax": 262},
    12: {"name": "Loamy sand",       "wp": 0.08, "fc": 0.16, "fmax": 500},
    13: {"name": "Sand",             "wp": 0.06, "fc": 0.13, "fmax": 720},
}


def _build_soil_property_rasters(texture_map: NDArray[np.uint8]) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    field_capacity = np.full(texture_map.shape, np.nan, dtype=np.float32)
    wilting_point = np.full(texture_map.shape, np.nan, dtype=np.float32)
    max_infiltration_rate = np.full(texture_map.shape, np.nan, dtype=np.float32)

    for soil_id, params in SOIL_PARAMETERS.items():
        mask = texture_map == soil_id
        field_capacity[mask] = params["fc"]
        wilting_point[mask] = params["wp"]
        max_infiltration_rate[mask] = params["fmax"]

    return field_capacity, wilting_point, max_infiltration_rate


def _fill_internal_nodata_holes(
    raster: NDArray[np.float32],
    valid_mask: NDArray[np.bool_] | None = None,
) -> NDArray[np.float32]:
    """Fill NaN values within a valid mask using nearest finite raster values."""
    array = np.asarray(raster, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("raster must be a 2D array")

    if valid_mask is None:
        valid_region = np.ones(array.shape, dtype=bool)
    else:
        valid_region = np.asarray(valid_mask, dtype=bool)
        if valid_region.shape != array.shape:
            raise ValueError("valid_mask must have the same shape as raster")

    fill_targets = valid_region & np.isnan(array)
    if not np.any(fill_targets):
        return array.copy()

    source_mask = valid_region & np.isfinite(array)
    if not np.any(source_mask):
        raise ValueError(
            "Cannot fill NaN values inside valid_mask because no finite source pixels exist within valid_mask."
        )

    nearest_indices = distance_transform_edt(
        ~source_mask,
        return_distances=False,
        return_indices=True,
    )

    filled = array.copy()
    filled[fill_targets] = array[
        nearest_indices[0][fill_targets],
        nearest_indices[1][fill_targets],
    ].astype(np.float32)
    return filled

@dataclass
class Soil:
     
    field_capacity: NDArray[np.float32]  # Volumetric water content at field capacity
    wilting_point: NDArray[np.float32]    # Volumetric water content at
    max_infiltration_rate: NDArray[np.float32] # mm/day
    texture_map: NDArray[np.uint8]
    soil_moisture_string: str = "field_capacity"
    soil_moisture_scalar: float = None
    soil_moisture_raster: NDArray[np.float32] = None
    clay_raster: NDArray[np.float32] = None
    sand_raster: NDArray[np.float32] = None
    silt_raster: NDArray[np.float32] = None

    soil_moisture: NDArray[np.float32] = None  # This will be set based on the initialization method

    @classmethod
    def from_texture(cls, texture_map: NDArray[np.uint8]):
        field_capacity, wilting_point, max_infiltration_rate = _build_soil_property_rasters(texture_map)

        return cls(
            texture_map=texture_map,
            field_capacity=field_capacity,
            wilting_point=wilting_point,
            max_infiltration_rate=max_infiltration_rate
        )

    def initialize_SM_from_string(self, value: str) -> None:
        if value == "field_capacity":
            self.soil_moisture = self.field_capacity
        elif value == "wilting_point":
            self.soil_moisture = self.wilting_point
        else:
            raise ValueError(
                "soil_moisture must be 'field_capacity', 'wilting_point'."
            )       
        pass

    def initialize_SM_from_raster(self, raster: NDArray[np.float32]) -> None:
        if raster.shape != self.field_capacity.shape:
            raise ValueError(
                "initial_soil_moisture raster dimensions do not match soil dimensions."
            )
        self.soil_moisture = raster
        pass

    def initialize_SM_from_scalar(self, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                "soil_moisture scalar must be between 0 and 1."
            )
        self.soil_moisture = value
        pass

    @classmethod
    def from_soilgrid(cls, reference_path, output_folder=None, fill_holes: bool = False) -> "Soil":

        try:
            from soilgrids import SoilGrids
        except Exception as exc:
            raise RuntimeError(
                "soilgrids is an optional dependency and could not be imported."
            ) from exc

        reference_georef = iwr.tools.get_raster_alignment_reference(reference_path)
        reference_bbox = iwr.tools.get_wgs84_extent_with_buffer(reference_georef)

        with rasterio.open(reference_path) as ref:
            ref_profile = ref.profile.copy()
            ref_transform = ref.transform
            ref_crs = ref.crs
            ref_shape = (ref.height, ref.width)
            ref_mask = ref.read_masks(1) > 0

        clay_raster = None
        sand_raster = None
        silt_raster = None

            ## Download soil data from ISRIC Soil grids
        target_dir = os.path.join(output_folder, 'soilgrids')
        os.makedirs(target_dir, exist_ok=True)

        # List of soilmaps to download
        soil_grids = SoilGrids()
        maps =['clay', 'sand', 'silt', 'soc']

        print("        *** DOWNLOADING ISRIC SOILGRIDS DATA ***")

        # Browsing through input soil maps
        for i in maps:
            print(i)
            # Skip WCS query entirely if all depth layers for this soil type are already downloaded
            existing_tifs = glob.glob(os.path.join(target_dir, i + '_*_mean.tif'))
            if len(existing_tifs) >= 6:
                print(f"             *** All {i} layers already downloaded, skipping WCS query ***")
                continue

            try:
                wcs = WebCoverageService('https://maps.isric.org/mapserv?map=/map/' + str(i) + '.map', version='2.0.1')
            except Exception as exc:
                print("             *** WARNING: SoilGrids service query failed. ***")
                print("             *** The ISRIC service is sometimes temporarily unresponsive. Please try running again. ***")
                raise RuntimeError(f"SoilGrids query failed for '{i}'.") from exc

            # Choosing the mean layers
            names = [k for k in wcs.contents.keys() if k.endswith('mean')]

            # Download via soilgrids (https://github.com/gantian127/soilgrids)
            for layer in names:
                print(layer)
                outfile = os.path.join(target_dir, layer + '.tif')

                # Skip download if file already exists
                if os.path.isfile(outfile):
                    print("             *** Skipping download as file already exists: " + outfile + " ***")
                    print("             *** If you want to download again, delete the file and run again ***")
                    continue

                data = soil_grids.get_coverage_data(
                    service_id= i,
                    coverage_id= layer,
                    crs= 'urn:ogc:def:crs:EPSG::4326',
                    west= reference_bbox["xmin"],
                    south= reference_bbox["ymin"],
                    east= reference_bbox["xmax"],
                    north= reference_bbox["ymax"],
                    width= reference_bbox["width"],
                    height= reference_bbox["height"],
                    output=  outfile)
                if data is None and not os.path.isfile(outfile):
                    print("             *** WARNING: SoilGrids download did not return data. ***")
                    print("             *** The ISRIC service is sometimes temporarily unresponsive. Please try running again. ***")
                    raise RuntimeError(f"SoilGrids download failed for '{layer}'.")


        ## Convert soil layers to mosaics for each depth layer
        print("Averaging soil layers and align to mask")
        target_dir_processed = os.path.join(output_folder, 'soilgrids_processed')
        os.makedirs(target_dir_processed, exist_ok=True)

        depth_aliases = {
            "0-5": 5.0,
            "5-15": 10.0,
            "15-30": 15.0,
            "30-60": 30.0,
            "60-100": 40.0,
            "100-200": 100.0,
        }
        ordered_depths = [
            "0-5",
            "5-15",
            "15-30",
            "30-60",
            "60-100",
            "100-200",
        ]

        ref_path = glob.glob(os.path.join(target_dir, '*0-5*.tif'))[0]
        with rasterio.open(ref_path) as src:
            ref = src.read(1)
            like_raster = np.zeros_like(ref)

        for soil_type in ['clay', 'sand', 'silt']:
            targetfile = os.path.join(target_dir_processed, 'soil_' + soil_type + '_averaged.tif')
            input_files = []
            for depth_key in ordered_depths:
                hits = sorted(glob.glob(os.path.join(target_dir, f"{soil_type}_*{depth_key}*mean.tif")))
                if len(hits) != 1:
                    raise FileNotFoundError(
                        f"Expected exactly one {soil_type} raster for depth {depth_key}, found {len(hits)}."
                    )
                input_files.append(hits[0])

            ref_profile.update(
                dtype=rasterio.float32,
                count=1,
                nodata=np.nan,
                compress="lzw",
                driver="GTiff",
            )
            if os.path.isfile(targetfile):
                print(f" Skipping {targetfile}, already exists.")
                with rasterio.open(targetfile) as src:
                    aligned = src.read(1).astype(np.float32)
                    if src.profile.get("transform") != ref_transform or src.shape != ref_shape or src.crs != ref_crs:
                        raise ValueError(f"Cached SoilGrids raster {targetfile} does not match the reference geometry.")
                if fill_holes:
                    aligned = np.where(ref_mask, aligned, np.nan)
                    aligned = _fill_internal_nodata_holes(aligned, valid_mask=ref_mask)
                    with rasterio.open(targetfile, "w", **ref_profile) as dst:
                        dst.write(aligned, 1)
            else:
                averaged, profile = weighted_average_rasters_by_depth(
                    raster_paths=input_files,
                    depth_list_cm=[depth_aliases[depth_key] for depth_key in ordered_depths],
                )

                averaged = averaged / 10.0  # Convert from g/kg to %
                averaged = np.where(np.isfinite(averaged) & (averaged == 0), np.nan, averaged)
                if fill_holes:
                    averaged = _fill_internal_nodata_holes(averaged)

                src_array = averaged.astype(np.float32)
                aligned = np.full(ref_shape, np.nan, dtype=np.float32)

                reproject(
                    source=src_array,
                    destination=aligned,
                    src_transform=profile["transform"],
                    src_crs=profile["crs"],
                    dst_transform=ref_transform,
                    dst_crs=ref_crs,
                    src_nodata=np.nan,
                    dst_nodata=np.nan,
                    resampling=Resampling.bilinear,
                )

                aligned = np.where(ref_mask, aligned, np.nan)

                if fill_holes:
                    aligned = _fill_internal_nodata_holes(aligned, valid_mask=ref_mask)

                with rasterio.open(targetfile, "w", **ref_profile) as dst:
                    dst.write(aligned, 1)

            if soil_type == 'clay':
                clay_raster = aligned
            elif soil_type == 'sand':
                sand_raster = aligned
            elif soil_type == 'silt':
                silt_raster = aligned

        if clay_raster is None or sand_raster is None or silt_raster is None:
            raise RuntimeError("Failed to build one or more SoilGrids rasters.")

        soil = cls(
            texture_map=np.zeros(ref_shape, dtype=np.uint8),
            field_capacity=np.full(ref_shape, np.nan, dtype=np.float32),
            wilting_point=np.full(ref_shape, np.nan, dtype=np.float32),
            max_infiltration_rate=np.full(ref_shape, np.nan, dtype=np.float32),
            clay_raster=clay_raster,
            sand_raster=sand_raster,
            silt_raster=silt_raster,
        )
        soil.derive_usda_texture()

        texture_target = os.path.join(target_dir_processed, 'soil_texture_map.tif')
        texture_profile = ref_profile.copy()
        texture_profile.update(dtype=rasterio.uint8, count=1, nodata=0, compress="lzw", driver="GTiff")
        if os.path.isfile(texture_target):
            print(f" Skipping {texture_target}, already exists.")
        else:
            with rasterio.open(texture_target, "w", **texture_profile) as dst:
                dst.write(soil.texture_map.astype(np.uint8), 1)

        soil.field_capacity, soil.wilting_point, soil.max_infiltration_rate = _build_soil_property_rasters(soil.texture_map)
        return soil


    def derive_usda_texture(self):
        """
        Classify USDA texture classes from percent sand/silt/clay.

        Parameters are expected in percent and should sum to ~100.
        Returns integer raster using USDA codes, with 0 reserved for nodata.
        """
        if self.sand_raster is None or self.silt_raster is None or self.clay_raster is None:
            raise ValueError("sand_raster, silt_raster, and clay_raster must be initialized first.")

        texture_codes = {
            "Sand": 13,
            "LoamySand": 12,
            "SandyLoam": 11,
            "Loam": 9,
            "SiltLoam": 7,
            "Silt": 6,
            "SandyClayLoam": 10,
            "ClayLoam": 5,
            "SiltyClayLoam": 4,
            "SandyClay": 8,
            "SiltyClay": 2,
            "Clay": 3,
        }

        self.texture_map = np.zeros(self.sand_raster.shape, dtype=np.uint8)

        sand = np.asarray(self.sand_raster, dtype=np.float32)
        silt = np.asarray(self.silt_raster, dtype=np.float32)
        clay = np.asarray(self.clay_raster, dtype=np.float32)

        valid_components = (
            np.isfinite(sand)
            & np.isfinite(silt)
            & np.isfinite(clay)
            & (sand >= 0.0)
            & (silt >= 0.0)
            & (clay >= 0.0)
        )
        total = sand + silt + clay
        valid_components &= (total > 0.0) & (total <= 150)

        self.texture_map = np.zeros(self.sand_raster.shape, dtype=np.uint8)

        sand_cls = valid_components & (silt + 1.5 * clay < 15)
        loamy_sand = valid_components & ((silt + 1.5 * clay) >= 15) & ((silt + 2.0 * clay) < 30)
        sandy_loam = valid_components & (
            ((clay >= 7) & (clay < 20) & (sand > 52) & ((silt + 2.0 * clay) >= 30))
            | ((clay < 7) & (silt < 50) & ((silt + 2.0 * clay) >= 30))
        )
        loam = valid_components & (clay >= 7) & (clay < 27) & (silt >= 28) & (silt < 50) & (sand <= 52)
        silt_loam = valid_components & (((silt >= 50) & (clay >= 12) & (clay < 27)) | ((silt >= 50) & (silt < 80) & (clay < 12)))
        silt_cls = valid_components & (silt >= 80) & (clay < 12)
        sandy_clay_loam = valid_components & (clay >= 20) & (clay < 35) & (silt < 28) & (sand > 45)
        clay_loam = valid_components & (clay >= 27) & (clay < 40) & (sand > 20) & (sand <= 45)
        silty_clay_loam = valid_components & (clay >= 27) & (clay < 40) & (sand <= 20)
        sandy_clay = valid_components & (clay >= 35) & (sand > 45)
        silty_clay = valid_components & (clay >= 40) & (silt >= 40)
        clay_cls = valid_components & (clay >= 40) & (sand <= 45) & (silt < 40)

        self.texture_map[sand_cls] = texture_codes["Sand"]
        self.texture_map[loamy_sand] = texture_codes["LoamySand"]
        self.texture_map[sandy_loam] = texture_codes["SandyLoam"]
        self.texture_map[loam] = texture_codes["Loam"]
        self.texture_map[silt_loam] = texture_codes["SiltLoam"]
        self.texture_map[silt_cls] = texture_codes["Silt"]
        self.texture_map[sandy_clay_loam] = texture_codes["SandyClayLoam"]
        self.texture_map[clay_loam] = texture_codes["ClayLoam"]
        self.texture_map[silty_clay_loam] = texture_codes["SiltyClayLoam"]
        self.texture_map[sandy_clay] = texture_codes["SandyClay"]
        self.texture_map[silty_clay] = texture_codes["SiltyClay"]
        self.texture_map[clay_cls] = texture_codes["Clay"]

        self.field_capacity, self.wilting_point, self.max_infiltration_rate = _build_soil_property_rasters(
            self.texture_map
        )

        return self.texture_map

        
