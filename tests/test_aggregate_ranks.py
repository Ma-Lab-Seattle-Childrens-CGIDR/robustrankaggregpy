import numpy as np
import pandas as pd

from robustrankaggregpy.aggregate_ranks import create_rank_matrix


def test_rank_matrix():
    test_rank_lists = [
        ["q", "o", "l", "m"],
        ["i", "y", "g", "v", "o", "f", "r", "t", "j", "u"],
        ["w", "n", "z", "k", "r", "x", "f", "g", "b", "t", "d", "o"],
    ]
    # Test rank lists (based on RobustRankAggreg Example)
    test_full_rank_mat = pd.DataFrame(
        {
            "q": [0.25, np.nan, np.nan],
            "o": [0.50, 0.5, 1.00000000],
            "l": [0.75, np.nan, np.nan],
            "m": [1.00, np.nan, np.nan],
            "i": [np.nan, 0.1, np.nan],
            "y": [np.nan, 0.2, np.nan],
            "g": [np.nan, 0.3, 0.66666667],
            "v": [np.nan, 0.4, np.nan],
            "f": [np.nan, 0.6, 0.58333333],
            "r": [np.nan, 0.7, 0.41666667],
            "t": [np.nan, 0.8, 0.83333333],
            "j": [np.nan, 0.9, np.nan],
            "u": [np.nan, 1.0, np.nan],
            "w": [np.nan, np.nan, 0.08333333],
            "n": [np.nan, np.nan, 0.16666667],
            "z": [np.nan, np.nan, 0.25000000],
            "k": [np.nan, np.nan, 0.33333333],
            "x": [np.nan, np.nan, 0.50000000],
            "b": [np.nan, np.nan, 0.75000000],
            "d": [np.nan, np.nan, 0.91666667],
        }
    ).T.sort_index()
    test_not_full_rank_mat = pd.DataFrame(
        {
            "q": [0.05, 1.00, 1.00],
            "o": [0.10, 0.25, 0.60],
            "l": [0.15, 1.00, 1.00],
            "m": [0.20, 1.00, 1.00],
            "i": [1.00, 0.05, 1.00],
            "y": [1.00, 0.10, 1.00],
            "g": [1.00, 0.15, 0.40],
            "v": [1.00, 0.20, 1.00],
            "f": [1.00, 0.30, 0.35],
            "r": [1.00, 0.35, 0.25],
            "t": [1.00, 0.40, 0.50],
            "j": [1.00, 0.45, 1.00],
            "u": [1.00, 0.50, 1.00],
            "w": [1.00, 1.00, 0.05],
            "n": [1.00, 1.00, 0.10],
            "z": [1.00, 1.00, 0.15],
            "k": [1.00, 1.00, 0.20],
            "x": [1.00, 1.00, 0.30],
            "b": [1.00, 1.00, 0.45],
            "d": [1.00, 1.00, 0.55],
        }
    ).T.sort_index()
    actual_rank_mat_not_full = create_rank_matrix(
        test_rank_lists, full=False
    ).sort_index()
    actual_rank_mat_full = create_rank_matrix(test_rank_lists, full=True).sort_index()
    print(actual_rank_mat_not_full)
    print(actual_rank_mat_full)
    # Check that the actual and test rank matrices are the same
    pd.testing.assert_frame_equal(
        test_not_full_rank_mat, actual_rank_mat_not_full, check_exact=False
    )
    pd.testing.assert_frame_equal(
        test_full_rank_mat, actual_rank_mat_full, check_exact=False
    )
