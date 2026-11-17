import os
import numpy as np
import pandas as pd
import iwr
import rasterio
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from iwr import aggregate_time_iwr, save_debug_raster_png
from iwr.tools import aggregate_crops_iwr
from iwr.tools import aggregate_time_iwr
from iwr.iwr_anomaly import compute_mean_and_interquartile_range
from iwr.plot import plot_iwr_vs_climatology
from plot import plot_yearly_iwr
from iwr.iwr_anomaly import compute_gamma_climatology
from iwr.phenology import Phenology
from iwr.crops import Crop
#from geoaquacrop_preprocess import soil

RUN_MODEL = True

check_folder = "/home/fremen/workspaces/RB/IWR_def/tmp_img"

# time extremes
START_DATE = "2010-01-01"
END_DATE = "2026-09-30"

# directories
static_folder = "/home/fremen/data/projects/VDA/00_static"
inputs_folder = "/home/fremen/data/projects/VDA/01_inputs"
outputs_folder = "/home/fremen/data/projects/VDA/02_outputs"
anomaly_folder = "/home/fremen/data/projects/VDA/05_archive"


# statics
working_grid_path = f"{static_folder}/vda_0nan_aligned.tif"
land_cover_grass_path = f"{static_folder}/grasslands_aligned.tif"
land_cover_vf_path = f"{static_folder}/vigneti_frutteti_aligned.tif"

# Load the valid mask from finite data only. Zero is valid coverage; NaN marks outside.
with rasterio.open(working_grid_path) as working_grid_src:
    working_grid = working_grid_src.read(1)
working_grid = np.isfinite(working_grid)
if not np.any(working_grid):
    raise ValueError(f"Reference working grid has no valid coverage: {working_grid_path}")

# generate fractional land cover for grasslandsand vf
with rasterio.open(land_cover_grass_path) as land_cover_grass_src:
    land_cover_grass = land_cover_grass_src.read(1)  # Read the first band

with rasterio.open(land_cover_vf_path) as land_cover_vf_src:
    land_cover_vf = land_cover_vf_src.read(1)  # Read the first band

grassland_dict = pd.DataFrame({
    "value": [1],
    "land_cover_name": ["grassland"]
})

vf_dict = pd.DataFrame({
    "value": [1],
    "land_cover_name": ["vigne_frutteti"]
})

fractional_grasslands, profile = iwr.aggregate_fractional(
    land_cover_path=land_cover_grass_path,
    class_mapping=grassland_dict,
    reference_grid=working_grid_path,
    output_path=f"{static_folder}/fractional_grasslands.tif"
)

fractional_vf, profile = iwr.aggregate_fractional(
    land_cover_path=land_cover_vf_path,
    class_mapping=vf_dict,
    reference_grid=working_grid_path,
    output_path=f"{static_folder}/fractional_vf.tif"
)

if not np.any(working_grid & (fractional_grasslands[0] > 0.0)):
    raise ValueError("Working domain and grassland crop fractions do not overlap.")
if not np.any(working_grid & (fractional_vf[0] > 0.0)):
    raise ValueError("Working domain and VF crop fractions do not overlap.")


# generate soil object from soilgrids

soil = iwr.Soil.from_soilgrid(
    reference_path=working_grid_path,
    output_folder=static_folder,
    fill_holes=True
)
soil.initialize_SM_from_string("field_capacity")  # Initialize soil moisture from string
# soil_moisture_raster_path = "/home/fremen/data/projects/VDA/02_outputs/grassland/soil_moisture/soil_moisture_2019_07_27.tif"
# with rasterio.open(soil_moisture_raster_path) as soil_moisture_src:
#     soil_moisture = soil_moisture_src.read(1)  # Read the first band
# soil.initialize_SM_from_raster(soil_moisture)



#soil = iwr.Soil.from_texture(soil_texture)

# load phenology from ASAP
pheno_paths = Phenology.create_asap_dataset(output_path=static_folder, reference_raster_path=working_grid_path)

# load phenology from ASAP
phenoe1 = rasterio.open(pheno_paths["phenoe1"]).read(1)
phenoe2 = rasterio.open(pheno_paths["phenoe2"]).read(1)
phenom1 = rasterio.open(pheno_paths["phenom1"]).read(1)
phenom2 = rasterio.open(pheno_paths["phenom2"]).read(1)
phenonseasons = rasterio.open(pheno_paths["phenonseasons"]).read(1)
phenos1 = rasterio.open(pheno_paths["phenos1"]).read(1)
phenos2 = rasterio.open(pheno_paths["phenos2"]).read(1)
phenosen1 = rasterio.open(pheno_paths["phenosen1"]).read(1)
phenosen2 = rasterio.open(pheno_paths["phenosen2"]).read(1)

phenology = Phenology.from_asap(
    valid_mask=working_grid,
    phenoe1=phenoe1,
    phenoe2=phenoe2,
    phenom1=phenom1,
    phenom2=phenom2,
    phenonseasons=phenonseasons,
    phenos1=phenos1,
    phenos2=phenos2,
    phenosen1=phenosen1,
    phenosen2=phenosen2
)

# prepare model scheduler for running
# model scheduler
schedule_df = iwr.build_forcing_dataframe(
    precipitation_folder=os.path.join(inputs_folder, "P"),
    pet_folder=os.path.join(inputs_folder, "PET"),
    start_date=START_DATE,
    end_date=END_DATE,
    precipitation_prefix="P_P_sim_",
    pet_prefix="PET_PET_sim_"
)

# create custom crop for grassland

grassland = Crop.custom(
    "grassland",
    name="grassland",
    root_max_depth=0.8,
    p=0.6,
    kc_max=0.8,
    kc_initial=0.4,
    kc_sen=0.45,
    crop_raster=fractional_grasslands[0],
)

# create custom crop for grassland

vf = Crop.custom(
    "vigne_frutteti",
    name="vigne_frutteti",
    root_max_depth=1.1,
    p=0.5,
    kc_max=0.95,
    kc_initial=0.37,
    kc_sen=0.5,
    crop_raster=fractional_vf[0],
)


# loop over different crops and run the model for each crop
for crop in [grassland, vf]:

    soil.initialize_SM_from_string("field_capacity")  # Initialize soil moisture from string
    # update phenology for current crop
    phenology.build_Kc_from_crop(crop)

    inputs = iwr.IWRInputs(
        scheduler=schedule_df,
        output_dir=outputs_folder,
        crop=crop, #object of Crop class
        soil=soil, #object of Soil class
        working_grid=working_grid,
        phenology=phenology,
        debug_mode=False
    )
    iwr_model = iwr.IWRModel(inputs)

   # iwr_model.run()  # Run the model

    # build dekadal iwr from daily outputs
    aggregated_iwr_folder = aggregate_time_iwr(
        iwr_folder=os.path.join(outputs_folder, crop.name, "iwr"),
        aggregation_window="dekad",  # or "month" for monthly aggregation
    )

    plot_yearly_iwr(
        iwr_folder=aggregated_iwr_folder,
        plot_out_folder="/home/fremen/data/projects/VDA/03_plots",
        crop_name=crop.name,
        crop_raster=crop.crop_raster,
    )

# Aggregate the crop-specific IWR rasters to a daily domain-wide value using the
# fractional crop cover as the weighting map.
aggregate_fractional_by_crop = {
    "grassland": fractional_grasslands[0],
    "vigne_frutteti": fractional_vf[0],
}

# aggregate_crops_iwr(
#     output_dir=outputs_folder,
#     model_output_dir={
#         "model_output_dir": outputs_folder,
#         "crop_names": ["grassland", "vigne_frutteti"],
#         "fractional_land_cover": aggregate_fractional_by_crop,
#     },
# )

# aggregated_iwr_folder_crops = aggregate_time_iwr(
#     iwr_folder=os.path.join(outputs_folder, "aggregated_iwr_crops"),
#     aggregation_window="dekad",  # or "month" for monthly aggregation
# )

# plot_yearly_iwr(
#     iwr_folder=aggregated_iwr_folder_crops,
#     plot_out_folder="/home/fremen/data/projects/VDA/03_plots/aggregated_crops",
#     crop_name="aggregated_crops",
#     crop_raster=np.sum([fractional_grasslands[0], fractional_vf[0]], axis=0),
# )

## anomaly computation


# compute_mean_and_interquartile_range(
#     data_folder="/home/fremen/data/projects/VDA/02_outputs/aggregated_iwr_crops_dekad",
#     output_folder=anomaly_folder,
#     start_date=START_DATE,
#     end_date=END_DATE)

plot_iwr_vs_climatology(
    mean_folder=os.path.join(anomaly_folder, "mean"),
    q1_folder=os.path.join(anomaly_folder, "q1"),
    q3_folder=os.path.join(anomaly_folder, "q3"),
    iwr_path=os.path.join(outputs_folder, "aggregated_iwr_crops_dekad"),
    start_date="2025-01-01",
    end_date="2025-12-31",
    aggregation_window="dekad",
    output_folder=os.path.join(outputs_folder, "plots"),
    filename="iwr_2025_vs_climatology.png",
    title="2025 IWR vs climatology",
)

# climatology = compute_gamma_climatology(
# data_folder="/path/to/your/iwr/folder",
# output_folder="/path/to/climatology/output",
# start_date="2010-01-01",
# end_date="2025-12-31",
# aggregation_window="dekad",   # or "month"
# )

# anomaly = climatology.compute_anomaly(
#     observation_file="/path/to/your/iwr/iwr_2024_01_10.tif",
#     output_file="/path/to/output/anomaly_2024_01_10.tif",
# )
