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

# time extremes
START_DATE = "2010-01-01"
END_DATE = "2025-12-31"

# directories
static_folder = "/home/fremen/data/projects/DAO/00_static"
inputs_folder = "/home/fremen/data/projects/DAO/01_inputs"
outputs_folder = "/home/fremen/data/projects/DAO/02_outputs_single_crop"
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

# statics
working_grid_path = f"{static_folder}/vda_0nan_aligned.tif"
land_cover_grass_path = f"{static_folder}/grasslands_aligned.tif"
land_cover_vf_path = f"{static_folder}/vigneti_frutteti_aligned.tif"

# inputs
eto_path = f"{inputs_folder}/PET.tif"
precipitation_path = f"{inputs_folder}/P.tif"

# Load the multi-band raster once
crop_fractions = rasterio.open(crop_fraction_path)



