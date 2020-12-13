import unittest
import pandas as pd
from poisson_bootstrap import get_resampled_data


class TestPoissonBootstrap(unittest.TestCase):
    def test_get_resampled_data(self):
        actual_resampled_data = get_resampled_data(pd.Series([1, 4]), 3, 10)
        expected_resampled_data = pd.DataFrame({
            "data": [1] * 3 + [4] * 3, "resample_id": [0, 1, 2] * 2, "count_by_resample_id": [1, 2, 0, 0, 1, 0]}
        )
        pd.util.testing.assert_frame_equal(expected_resampled_data, actual_resampled_data)


if __name__ == '__main__':
    unittest.main()
