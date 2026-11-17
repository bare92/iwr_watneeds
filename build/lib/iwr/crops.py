from dataclasses import dataclass
import numpy as np

CROP_LIBRARY = {

    "wheat": {
        "name": "wheat",
        "root_max_depth": 1.0,
        "kc_initial": 0.30,
        "kc_max": 1.15,
        "kc_sen": 0.40,
        "p": 0.55,
    },

    "maize": {
        "name": "maize",
        "root_max_depth": 1.7,
        "kc_initial": 0.30,
        "kc_max": 1.20,
        "kc_sen": 0.60,
        "p": 0.55,
    },

    "rice": {
        "name": "rice",
        "root_max_depth": 0.6,
        "kc_initial": 1.05,
        "kc_max": 1.20,
        "kc_sen": 0.90,
        "p": 0.20,
    },

    "barley": {
        "name": "barley",
        "root_max_depth": 1.3,
        "kc_initial": 0.30,
        "kc_max": 1.15,
        "kc_sen": 0.40,
        "p": 0.55,
    },

    "citrus": {
        "name": "citrus",
        "root_max_depth": 1.3,
        "kc_initial": 0.70,
        "kc_max": 0.65,
        "kc_sen": 0.70,
        "p": 0.50,
    },

    "fodder": {
        "name": "fodder",
        "root_max_depth": 1.2,
        "kc_initial": 0.40,
        "kc_max": 0.95,
        "kc_sen": 0.90,
        "p": 0.55,
    },

    "fruits": {
        "name": "fruits",
        "root_max_depth": 1.0,
        "kc_initial": 0.45,
        "kc_max": 0.95,
        "kc_sen": 0.75,
        "p": 0.50,
    },

    "oats": {
        "name": "oats",
        "root_max_depth": 1.3,
        "kc_initial": 0.30,
        "kc_max": 1.15,
        "kc_sen": 0.40,
        "p": 0.55,
    },

    "olives": {
        "name": "olives",
        "root_max_depth": 1.5,
        "kc_initial": 0.65,
        "kc_max": 0.70,
        "kc_sen": 0.70,
        "p": 0.65,
    },

    "other_cereals": {
        "name": "other_cereals",
        "root_max_depth": 1.3,
        "kc_initial": 0.30,
        "kc_max": 1.10,
        "kc_sen": 0.40,
        "p": 0.55,
    },

    "other_industrial": {
        "name": "other_industrial",
        "root_max_depth": 1.2,
        "kc_initial": 0.35,
        "kc_max": 1.15,
        "kc_sen": 0.35,
        "p": 0.55,
    },

    "other_roots": {
        "name": "other_roots",
        "root_max_depth": 0.6,
        "kc_initial": 0.50,
        "kc_max": 1.10,
        "kc_sen": 0.95,
        "p": 0.45,
    },

    "potatoes": {
        "name": "potatoes",
        "root_max_depth": 0.6,
        "kc_initial": 0.50,
        "kc_max": 1.15,
        "kc_sen": 0.75,
        "p": 0.35,
    },

    "pulses": {
        "name": "pulses",
        "root_max_depth": 0.9,
        "kc_initial": 0.40,
        "kc_max": 1.15,
        "kc_sen": 0.35,
        "p": 0.45,
    },

    "rapeseed": {
        "name": "rapeseed",
        "root_max_depth": 1.3,
        "kc_initial": 0.35,
        "kc_max": 1.15,
        "kc_sen": 0.35,
        "p": 0.60,
    },

    "rye": {
        "name": "rye",
        "root_max_depth": 1.3,
        "kc_initial": 0.30,
        "kc_max": 1.15,
        "kc_sen": 0.40,
        "p": 0.55,
    },

    "sugarbeet": {
        "name": "sugarbeet",
        "root_max_depth": 1.0,
        "kc_initial": 0.35,
        "kc_max": 1.20,
        "kc_sen": 0.70,
        "p": 0.55,
    },

    "sunflower": {
        "name": "sunflower",
        "root_max_depth": 1.5,
        "kc_initial": 0.35,
        "kc_max": 1.15,
        "kc_sen": 0.35,
        "p": 0.45,
    },

    "triticale": {
        "name": "triticale",
        "root_max_depth": 1.3,
        "kc_initial": 0.30,
        "kc_max": 1.10,
        "kc_sen": 0.40,
        "p": 0.55,
    },

    "vegetables": {
        "name": "vegetables",
        "root_max_depth": 0.6,
        "kc_initial": 0.60,
        "kc_max": 1.05,
        "kc_sen": 0.85,
        "p": 0.45,
    },

    "vineyard": {
        "name": "vineyard",
        "root_max_depth": 1.2,
        "kc_initial": 0.30,
        "kc_max": 0.70,
        "kc_sen": 0.45,
        "p": 0.45,
    },
}

@dataclass
class Crop:
    name: str
    root_max_depth: float
    kc_initial: float
    kc_max: float
    kc_sen: float
    p: float
    crop_raster: np.ndarray = None  # raster representation of the crop fraction

    def __post_init__(self):
        if self.root_max_depth <= 0:
            raise ValueError("root_max_depth must be > 0")
        if self.kc_initial < 0 or self.kc_max < 0 or self.kc_sen < 0:
            raise ValueError("Kc values must be >= 0")
        if not 0 <= self.p <= 1:
            raise ValueError("p must be between 0 and 1")

    @classmethod
    def from_name(cls, crop_name: str, crop_raster: np.ndarray | None = None):
        if crop_name not in CROP_LIBRARY:
            raise ValueError(f"Unknown crop: {crop_name}")
        data = CROP_LIBRARY[crop_name].copy()
        data["crop_raster"] = crop_raster
        return cls(**data)

    @classmethod
    def custom(cls, crop_name: str, **overrides):
        data = CROP_LIBRARY[crop_name].copy()
        data.update(overrides)
        return cls(**data)