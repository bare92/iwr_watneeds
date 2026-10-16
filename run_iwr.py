import os
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
#from geoaquacrop_preprocess import soil

check_folder = "/home/fremen/workspaces/RB/IWR_def/tmp_img"

# time extremes
START_DATE = "2010-01-01"
END_DATE = "2025-12-31"

# directories
static_folder = "/home/fremen/data/projects/DAO/00_static"
inputs_folder = "/home/fremen/data/projects/DAO/01_inputs"
outputs_folder = "/home/fremen/data/projects/DAO/02_outputs"
anomaly_folder = "/home/fremen/data/projects/DAO/03_archive"

# model scheduler
schedule_df = iwr.build_forcing_dataframe(
    precipitation_folder=os.path.join(inputs_folder, "P"),
    pet_folder=os.path.join(inputs_folder, "PET"),
    start_date=START_DATE,
    end_date=END_DATE,
    precipitation_prefix="P_P_sim_",
    pet_prefix="PET_PET_sim_"
)
    

# statics
working_grid_path = f"{static_folder}/working_grid_3035_1km_precip_valid.tif"
soil_texture_path = f"{static_folder}/soil_texture_aligned.tif"
crop_fraction_path = f"{static_folder}/italy_areas_irrigated_DAO_1km_clipped_no_i.tif"
irrigation_mask_path = f"{static_folder}/distretti_irrigui_SIGRIAN_3035_1km_mask_aligned.tif"  # Optional irrigation mask

# inputs
eto_path = f"{inputs_folder}/PET.tif"
precipitation_path = f"{inputs_folder}/P.tif"

# Load the multi-band raster once
crop_fractions = rasterio.open(crop_fraction_path)

# Load the valid mask and soil texture rasters
working_grid_src = rasterio.open(working_grid_path)
working_grid = working_grid_src.read(1)  # Read the first band
irrigation_mask_src = rasterio.open(irrigation_mask_path)
irrigation_mask = irrigation_mask_src.read(1)  # Read the first band

with rasterio.open(soil_texture_path) as soil_texture_src:
    soil_texture = soil_texture_src.read(1)  # Read the first band

# define soil object

soil = iwr.Soil.from_soilgrid(reference_path=working_grid_path, output_folder=static_folder)



#soil = iwr.Soil.from_texture(soil_texture)
soil.initialize_SM_from_string("field_capacity")  # Initialize soil moisture from string
# load phenology from ASAP
phenoe1 = rasterio.open(f"{static_folder}/phenoe1_v04_aligned.tif").read(1)
phenoe2 = rasterio.open(f"{static_folder}/phenoe2_v04_aligned.tif").read(1)
phenom1 = rasterio.open(f"{static_folder}/phenom1_v04_aligned.tif").read(1)
phenom2 = rasterio.open(f"{static_folder}/phenom2_v04_aligned.tif").read(1)
phenonseasons = rasterio.open(f"{static_folder}/phenonseasons_v04_aligned.tif").read(1)
phenos1 = rasterio.open(f"{static_folder}/phenos1_v04_aligned.tif").read(1)
phenos2 = rasterio.open(f"{static_folder}/phenos2_v04_aligned.tif").read(1)
phenosen1 = rasterio.open(f"{static_folder}/phenosen1_v04_aligned.tif").read(1)
phenosen2 = rasterio.open(f"{static_folder}/phenosen2_v04_aligned.tif").read(1)

phenology = iwr.Phenology.from_spam(
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


#print(phenology)


# Map band indices to crop namesS
CROP_BANDS = {
    1: "barley",
    2: "citrus",
    3: "fodder",
    4: "fruits",
    5: "maize",
    6: "oats",
    7: "olives",
    8: "other_cereals",
    9: "other_industrial",
    10: "other_roots",
    11: "potatoes",
    12: "pulses",
    13: "rapeseed",
    14: "rice",
    15: "rye",
    16: "sugarbeet",
    17: "sunflower",
    18: "triticale",
    19: "vegetables",
    20: "vineyard",
}



# plot_yearly_iwr(
#     iwr_folder="/home/fremen/data/projects/DAO/02_outputs/aggregated_iwr",
#     plot_out_folder="/home/fremen/data/projects/DAO/03_plots",
#     reference_value=1311,
# )




for band_index, crop_name in CROP_BANDS.items():
#band_index = 20
    crop_name = CROP_BANDS[band_index]
    crop_raster = crop_fractions.read(band_index)
    crop = iwr.Crop.from_name(crop_name, crop_raster=crop_raster)

    phenology.build_Kc_from_crop(crop)

    inputs = iwr.IWRInputs(
        scheduler=schedule_df,
        output_dir=outputs_folder,
        crop=crop, #object of Crop class
        soil=soil, #object of Soil class
        working_grid=working_grid,
        phenology=phenology,
        debug_mode=False,  # Enable debug mode to save intermediate outputs
        irrigation_mask=irrigation_mask  # Pass the irrigation mask to the model
    )
    iwr_model = iwr.IWRModel(inputs)

    iwr_model.run()  # Run the model

# def run_iwr(band_index, crop_name):
#     crop_raster = crop_fractions.read(band_index)
#     crop = iwr.Crop.from_name(crop_name, crop_raster=crop_raster)

#     phenology.build_Kc_from_crop(crop)

#     inputs = iwr.IWRInputs(
#         scheduler=schedule_df,
#         output_dir=outputs_folder,
#         crop=crop,  # object of Crop class
#         soil=soil,  # object of Soil class
#         working_grid=working_grid,
#         phenology=phenology,
#         debug_mode=False,  # Enable debug mode to save intermediate outputs
#         irrigation_mask=irrigation_mask  # Pass the irrigation mask to the model
#     )
#     iwr_model = iwr.IWRModel(inputs)
#     iwr_model.run()  # Run the model

# parallel = Parallel(n_jobs=8, backend="threading")(
#     delayed(run_iwr)(band_index, crop_name)
#     for band_index, crop_name in CROP_BANDS.items()
 # Use all available CPU cores
# ------------------------------------------------------------------
# Aggregate crop-specific IWR using crop fractions
# ------------------------------------------------------------------

# print("Aggregating crop-specific IWR...")

aggregate_crops_iwr(
    output_dir=outputs_folder,
    crop_fractions_path=crop_fraction_path,
    scheduler=schedule_df,
    crop_band_map=CROP_BANDS,
)

print("IWR aggregation completed.")

aggregate_time_iwr(
    iwr_folder="/home/fremen/data/projects/DAO/02_outputs/aggregated_iwr",
    aggregation_window="dekad",  # or "month" for monthly aggregation
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
