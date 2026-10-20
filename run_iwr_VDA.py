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
from plot import plot_yearly_iwr
from iwr.iwr_anomaly import compute_gamma_climatology
from iwr.phenology import Phenology
from iwr.crops import Crop
#from geoaquacrop_preprocess import soil

check_folder = "/home/fremen/workspaces/RB/IWR_def/tmp_img"

# time extremes
START_DATE = "2019-07-27"
END_DATE = "2026-09-30"

# directories
static_folder = "/home/fremen/data/projects/VDA/00_static"
inputs_folder = "/home/fremen/data/projects/VDA/01_inputs"
outputs_folder = "/home/fremen/data/projects/VDA/02_outputs"
anomaly_folder = "/home/fremen/data/projects/VDA/03_archive"


# statics
working_grid_path = f"{static_folder}/vda_0nan_aligned.tif"
land_cover_grass_path = f"{static_folder}/grasslands_aligned.tif"

# Load the valid mask
working_grid_src = rasterio.open(working_grid_path)
working_grid = working_grid_src.read(1)  # Read the first band

# generate fractional land cover for grasslands
with rasterio.open(land_cover_grass_path) as land_cover_grass_src:
    land_cover_grass = land_cover_grass_src.read(1)  # Read the first band

grassland_dict = pd.DataFrame({
    "value": [1],
    "land_cover_name": ["grassland"]
})

fractional_grasslands, profile = iwr.aggregate_fractional(
    land_cover_path=land_cover_grass_path,
    class_mapping=grassland_dict,
    reference_grid=working_grid_path,
    output_path=f"{static_folder}/fractional_grasslands.tif"
)


# generate soil object from soilgrids

soil = iwr.Soil.from_soilgrid(
    reference_path=working_grid_path,
    output_folder=static_folder,
    fill_holes=True
)
#soil.initialize_SM_from_string("field_capacity")  # Initialize soil moisture from string
soil_moisture_raster_path = "/home/fremen/data/projects/VDA/02_outputs/grassland/soil_moisture/soil_moisture_2019_07_27.tif"
with rasterio.open(soil_moisture_raster_path) as soil_moisture_src:
    soil_moisture = soil_moisture_src.read(1)  # Read the first band
soil.initialize_SM_from_raster(soil_moisture)



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
# update phenology for current crop
phenology.build_Kc_from_crop(grassland)

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

inputs = iwr.IWRInputs(
    scheduler=schedule_df,
    output_dir=outputs_folder,
    crop=grassland, #object of Crop class
    soil=soil, #object of Soil class
    working_grid=working_grid,
    phenology=phenology,
    debug_mode=False
)
iwr_model = iwr.IWRModel(inputs)

iwr_model.run()  # Run the model

# build dekadal iwr from daily outputs
aggregated_iwr_folder = aggregate_time_iwr(
    iwr_folder=os.path.join(outputs_folder, grassland.name, "iwr"),
    aggregation_window="dekad",  # or "month" for monthly aggregation
)

plot_yearly_iwr(
    iwr_folder=aggregated_iwr_folder,
    plot_out_folder="/home/fremen/data/projects/VDA/03_plots"
)

# compute_mean_and_interquartile_range(
#     data_folder="/home/fremen/data/projects/DAO/02_outputs/aggregated_iwr_dekad",
#     output_folder=anomaly_folder,
#     start_date=START_DATE,
#     end_date=END_DATE)


# obj = compute_gamma_climatology(
#     data_folder="/home/fremen/data/projects/DAO/02_outputs/aggregated_iwr_dekad",
#     output_folder=anomaly_folder,
#     start_date="2011-01-01",
#     end_date="2024-12-31",
#     aggregation_window="dekad",   # or "month"
# )

# clim = obj.compute_climatology()

# anom = obj.compute_anomaly(
#     "/home/fremen/data/projects/DAO/02_outputs/aggregated_iwr_dekad/iwr_2022_01_10.tif",
#     output_file="/home/fremen/data/projects/DAO/03_plots/anomaly_2022_01_10.tif",
