from types import SimpleNamespace

import numpy as np
import pandas as pd

from iwr.model import IWRModel
from iwr.phenology import Phenology


def _make_identity_phenology() -> Phenology:
    valid_mask = np.ones((1, 2), dtype=np.uint8)
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

    state = np.arange(36, dtype=np.uint8)[:, None, None]
    state = np.repeat(state, 2, axis=2)
    phenology.state.values[...] = state
    return phenology


def _make_active_phenology() -> Phenology:
    valid_mask = np.ones((1, 2), dtype=np.uint8)
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

    state = np.zeros((36, 1, 2), dtype=np.uint8)
    for month in range(5, 10):
        state[3 * (month - 1) : 3 * month, :, :] = 1
    phenology.state.values[...] = state
    return phenology


def _model_with_calendars() -> IWRModel:
    model = object.__new__(IWRModel)
    model.inputs = SimpleNamespace(phenology=_make_identity_phenology())
    return model


def test_calendar_lookup_indices_match_calendar_dekads() -> None:
    model = _model_with_calendars()

    expected_indices = {
        pd.Timestamp("2023-01-01"): 0,
        pd.Timestamp("2023-01-10"): 0,
        pd.Timestamp("2023-01-11"): 1,
        pd.Timestamp("2023-01-20"): 1,
        pd.Timestamp("2023-01-21"): 2,
        pd.Timestamp("2023-01-31"): 2,
        pd.Timestamp("2023-02-01"): 3,
        pd.Timestamp("2023-02-21"): 5,
        pd.Timestamp("2023-02-28"): 5,
        pd.Timestamp("2024-02-29"): 5,
        pd.Timestamp("2023-03-01"): 6,
        pd.Timestamp("2023-04-01"): 9,
        pd.Timestamp("2023-05-01"): 12,
        pd.Timestamp("2023-07-01"): 18,
        pd.Timestamp("2023-09-30"): 26,
        pd.Timestamp("2023-12-31"): 35,
    }

    for timestamp, expected_index in expected_indices.items():
        actual_index = int(model._get_phenology_state_for_time(timestamp)[0, 0])
        assert actual_index == expected_index, f"{timestamp}: expected {expected_index}, got {actual_index}"


def test_phenology_daily_lookup_matches_reference_calendar_for_2023_and_2024() -> None:
    phenology = _make_identity_phenology()
    daily_state = phenology._state_to_daily()

    model = object.__new__(IWRModel)
    model.inputs = SimpleNamespace(phenology=phenology)

    for year in (2023, 2024):
        for timestamp in pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D"):
            reference = pd.Timestamp(year=2024, month=timestamp.month, day=timestamp.day)
            expected_index = reference.dayofyear - 1
            actual_index = int(model._get_phenology_state_for_time(timestamp)[0, 0])
            expected_value = int(daily_state[expected_index, 0, 0])
            assert actual_index == expected_value, (
                f"Mismatch for {timestamp.isoformat()}: expected calendar value {expected_value} "
                f"from daily state index {expected_index}, got {actual_index}"
            )


def test_july_reporting_uses_actual_calendar_lookup_and_keeps_inactive_dates_zero() -> None:
    model = object.__new__(IWRModel)
    model.inputs = SimpleNamespace(phenology=_make_active_phenology())

    july_et_potential = np.array([[8.0, 8.0]], dtype=np.float32)
    july_et_actual = np.array([[4.0, 2.0]], dtype=np.float32)
    july_blue_water = np.maximum(0.0, july_et_potential - july_et_actual)

    july_state = model._get_phenology_state_for_time(pd.Timestamp("2024-07-15"))
    july_reported = model._apply_active_crop_mask_to_iwr(july_blue_water, july_state)

    assert np.any(july_et_actual < july_et_potential)
    assert int(july_state[0, 0]) == 1
    np.testing.assert_allclose(july_reported[0], np.array([4.0, 6.0], dtype=np.float32), rtol=0.0, atol=1e-6)

    december_state = model._get_phenology_state_for_time(pd.Timestamp("2024-12-15"))
    december_blue_water = np.array([[5.0, 7.0]], dtype=np.float32)
    december_reported = model._apply_active_crop_mask_to_iwr(december_blue_water, december_state)

    assert int(december_state[0, 0]) == 0
    np.testing.assert_allclose(december_reported, np.array([[0.0, 0.0]], dtype=np.float32), rtol=0.0, atol=1e-6)

    invalid_blue_water = np.array([[np.nan, 5.0]], dtype=np.float32)
    invalid_state = np.array([[1, 1]], dtype=np.uint8)
    invalid_reported = model._apply_active_crop_mask_to_iwr(invalid_blue_water, invalid_state)
    assert np.isnan(invalid_reported[0, 0])
    np.testing.assert_allclose(invalid_reported[0, 1], 5.0, rtol=0.0, atol=1e-6)
