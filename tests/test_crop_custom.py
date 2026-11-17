import numpy as np
import pytest

from iwr.crops import Crop


def test_crop_custom_can_build_brand_new_crop() -> None:
    crop = Crop.custom(
        "grassland",
        root_max_depth=0.8,
        kc_initial=0.4,
        kc_max=0.8,
        kc_sen=0.45,
        p=0.6,
        crop_raster=np.ones((2, 2), dtype=np.float32),
    )

    assert crop.name == "grassland"
    assert crop.root_max_depth == 0.8
    assert crop.kc_initial == 0.4
    assert crop.kc_max == 0.8
    assert crop.kc_sen == 0.45
    assert crop.p == 0.6
    assert crop.crop_raster.shape == (2, 2)


def test_crop_custom_still_supports_predefined_crops() -> None:
    crop = Crop.custom("maize", kc_max=1.25)

    assert crop.name == "maize"
    assert crop.root_max_depth == 1.7
    assert crop.kc_max == 1.25


def test_crop_custom_requires_core_parameters_for_new_crop() -> None:
    with pytest.raises(ValueError, match="Missing required crop parameters"):
        Crop.custom("new_crop", root_max_depth=1.0, kc_initial=0.3)