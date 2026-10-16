from dataclasses import dataclass
from numpy.typing import NDArray
import numpy as np


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

@dataclass
class Soil:
    texture_map: NDArray[np.uint8]
    field_capacity: NDArray[np.float32]  # Volumetric water content at field capacity
    wilting_point: NDArray[np.float32]    # Volumetric water content at
    max_infiltration_rate: NDArray[np.float32] # mm/day
    soil_moisture_string: str = "field_capacity"
    soil_moisture_scalar: float = None
    soil_moisture_raster: NDArray[np.float32] = None

    soil_moisture: NDArray[np.float32] = None  # This will be set based on the initialization method

    @classmethod
    def from_texture(cls, texture_map: NDArray[np.uint8]):
        field_capacity = np.zeros_like(texture_map, dtype=np.float32)
        wilting_point = np.zeros_like(texture_map, dtype=np.float32)
        max_infiltration_rate = np.zeros_like(texture_map, dtype=np.float32)
        

        for soil_id, params in SOIL_PARAMETERS.items():
            mask = texture_map == soil_id
            field_capacity[mask] = params["fc"]
            wilting_point[mask] = params["wp"]
            max_infiltration_rate[mask] = params["fmax"]

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
