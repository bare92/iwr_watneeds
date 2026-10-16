import numpy as np

from iwr.phenology import Phenology


def test_phenology_daily_access_handles_leap_year():
    valid_mask = np.ones((2, 2), dtype=np.uint8)
    zeros = np.zeros_like(valid_mask)
    ones = np.ones_like(valid_mask)

    phenology = Phenology(
        valid_mask=valid_mask,
        phenoe1=zeros,
        phenoe2=zeros,
        phenom1=zeros,
        phenom2=zeros,
        phenonseasons=ones,
        phenos1=zeros,
        phenos2=zeros,
        phenosen1=zeros,
        phenosen2=zeros,
    )

    daily_state = phenology._state_to_daily()
    assert daily_state.shape[0] == 366
    assert phenology.Kc_matrix.values.shape[0] == 366
    assert phenology.Kc_matrix.values[365].shape == (2, 2)
