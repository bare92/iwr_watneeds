import numpy as np
import pytest

from iwr.tools import _normalize_crop_fractions


def test_crop_fraction_normalization_keeps_unit_fractions() -> None:
    fractions = np.array([[[0.0, 0.25], [0.5, np.nan]]], dtype=np.float32)

    normalized = _normalize_crop_fractions(fractions)

    np.testing.assert_allclose(normalized, fractions, equal_nan=True)


def test_crop_fraction_normalization_converts_percent_scale() -> None:
    fractions = np.array([[[0.0, 25.0], [50.0, np.nan]]], dtype=np.float32)

    normalized = _normalize_crop_fractions(fractions)

    expected = np.array([[[0.0, 0.25], [0.5, np.nan]]], dtype=np.float32)
    np.testing.assert_allclose(normalized, expected, equal_nan=True)


def test_crop_fraction_normalization_rejects_negative_values() -> None:
    fractions = np.array([[[-0.1, 0.25]]], dtype=np.float32)

    with pytest.raises(ValueError, match="values < 0"):
        _normalize_crop_fractions(fractions)


def test_crop_fraction_normalization_rejects_values_above_percent_scale() -> None:
    fractions = np.array([[[125.0, 25.0]]], dtype=np.float32)

    with pytest.raises(ValueError, match="values > 1"):
        _normalize_crop_fractions(fractions)