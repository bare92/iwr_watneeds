import numpy as np

from iwr.model import IWRModel


def _build_model() -> IWRModel:
    model = IWRModel.__new__(IWRModel)
    model._last_balance_correction_stats = None
    return model


def test_runoff_closure_no_runoff() -> None:
    model = _build_model()
    soil_storage_candidate = np.array([[70.0]], dtype=np.float32)
    taw = np.array([[100.0]], dtype=np.float32)

    runoff_mm, soil_storage_new = model._compute_runoff_from_excess_storage(soil_storage_candidate, taw)

    np.testing.assert_allclose(runoff_mm, [[0.0]])
    np.testing.assert_allclose(soil_storage_new, [[70.0]])


def test_runoff_closure_generates_explicit_runoff() -> None:
    model = _build_model()
    soil_storage_candidate = np.array([[110.0]], dtype=np.float32)
    taw = np.array([[100.0]], dtype=np.float32)

    runoff_mm, soil_storage_new = model._compute_runoff_from_excess_storage(soil_storage_candidate, taw)

    np.testing.assert_allclose(runoff_mm, [[10.0]])
    np.testing.assert_allclose(soil_storage_new, [[100.0]])


def test_runoff_closure_at_taw() -> None:
    model = _build_model()
    soil_storage_candidate = np.array([[100.0]], dtype=np.float32)
    taw = np.array([[100.0]], dtype=np.float32)

    runoff_mm, soil_storage_new = model._compute_runoff_from_excess_storage(soil_storage_candidate, taw)

    np.testing.assert_allclose(runoff_mm, [[0.0]])
    np.testing.assert_allclose(soil_storage_new, [[100.0]])


def test_runoff_closure_preserves_mass() -> None:
    model = _build_model()
    s_previous = np.array([[90.0]], dtype=np.float32)
    peff = np.array([[30.0]], dtype=np.float32)
    et_actual = np.array([[5.0]], dtype=np.float32)
    deep_percolation = np.array([[5.0]], dtype=np.float32)
    taw = np.array([[100.0]], dtype=np.float32)

    soil_storage_candidate = s_previous + peff - et_actual - deep_percolation
    runoff_mm, soil_storage_new = model._compute_runoff_from_excess_storage(soil_storage_candidate, taw)

    np.testing.assert_allclose(runoff_mm, [[10.0]])
    np.testing.assert_allclose(soil_storage_new, [[100.0]])
    np.testing.assert_allclose(s_previous + peff, et_actual + deep_percolation + runoff_mm + soil_storage_new)


def test_runoff_closure_vectorized_raster() -> None:
    model = _build_model()
    soil_storage_candidate = np.array(
        [
            [70.0, 110.0],
            [100.0, np.nan],
        ],
        dtype=np.float32,
    )
    taw = np.array(
        [
            [100.0, 100.0],
            [100.0, 100.0],
        ],
        dtype=np.float32,
    )

    runoff_mm, soil_storage_new = model._compute_runoff_from_excess_storage(soil_storage_candidate, taw)

    expected_runoff = np.array(
        [
            [0.0, 10.0],
            [0.0, np.nan],
        ],
        dtype=np.float32,
    )
    expected_storage = np.array(
        [
            [70.0, 100.0],
            [100.0, np.nan],
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(runoff_mm, expected_runoff, equal_nan=True)
    np.testing.assert_allclose(soil_storage_new, expected_storage, equal_nan=True)


def test_runoff_closure_large_rainfall_becomes_runoff() -> None:
    model = _build_model()
    s_previous = np.array([[95.0]], dtype=np.float32)
    peff = np.array([[40.0]], dtype=np.float32)
    et_actual = np.array([[5.0]], dtype=np.float32)
    deep_percolation = np.array([[5.0]], dtype=np.float32)
    taw = np.array([[100.0]], dtype=np.float32)

    soil_storage_candidate = s_previous + peff - et_actual - deep_percolation
    runoff_mm, soil_storage_new = model._compute_runoff_from_excess_storage(soil_storage_candidate, taw)

    np.testing.assert_allclose(soil_storage_candidate, [[125.0]])
    np.testing.assert_allclose(runoff_mm, [[25.0]])
    np.testing.assert_allclose(soil_storage_new, [[100.0]])


def test_runoff_closure_preserves_small_excess_mass() -> None:
    model = _build_model()
    soil_storage_candidate = np.array([[100.000008]], dtype=np.float64)
    taw = np.array([[100.0]], dtype=np.float64)

    runoff_mm, soil_storage_new = model._compute_runoff_from_excess_storage(soil_storage_candidate, taw)

    np.testing.assert_allclose(runoff_mm, [[0.000008]], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(soil_storage_new, [[100.0]], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(soil_storage_candidate, soil_storage_new + runoff_mm, rtol=0.0, atol=1e-12)