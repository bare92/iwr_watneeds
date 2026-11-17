import numpy as np

from iwr.model import IWRModel


def test_debug_plot_array_downsampling_reduces_large_rasters():
    large = np.arange(10000, dtype=np.float32).reshape(100, 100)

    downsampled = IWRModel._prepare_debug_plot_array(large, max_pixels=1000)

    assert downsampled.size <= 1000
    assert downsampled.shape[0] <= large.shape[0]
    assert downsampled.shape[1] <= large.shape[1]
