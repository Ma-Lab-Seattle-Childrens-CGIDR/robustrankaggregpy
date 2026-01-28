from typing import cast

import numpy as np
import pandas as pd
import pytest

from robustrankaggregpy.aggregate_ranks import (
    create_rank_matrix,
    q_stuart,
    sum_stuart,
    stuart,
    FloatMatrix1D,
    FloatMatrix2D,
)


# Create a Pytest fixture providing a common set of rank lists
@pytest.fixture
def rank_lists():
    # glist <- list(
    #   c("q", "o", "l", "m"),
    #   c("i", "y", "g", "v", "o", "f", "r", "t", "j", "u"),
    #   c("w", "n", "z", "k", "r", "x", "f", "g", "b", "t", "d", "o")
    #   )
    return [
        ["q", "o", "l", "m"],
        ["i", "y", "g", "v", "o", "f", "r", "t", "j", "u"],
        ["w", "n", "z", "k", "r", "x", "f", "g", "b", "t", "d", "o"],
    ]


@pytest.fixture
def rank_matrix(rank_lists):
    return create_rank_matrix(rank_lists)


def test_rank_matrix(rank_lists):
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
    actual_rank_mat_not_full = create_rank_matrix(rank_lists, full=False).sort_index()
    actual_rank_mat_full = create_rank_matrix(rank_lists, full=True).sort_index()
    # Check that the actual and test rank matrices are the same
    pd.testing.assert_frame_equal(
        test_not_full_rank_mat, actual_rank_mat_not_full, check_exact=False
    )
    pd.testing.assert_frame_equal(
        test_full_rank_mat, actual_rank_mat_full, check_exact=False
    )


def test_sum_stuart():
    test_v: FloatMatrix1D = cast(
        FloatMatrix1D,
        np.array(
            [
                0.8888889,
                0.5555556,
                0.2222222,
                0.6666667,
                0.5555556,
                0.1111111,
                0.3333333,
                1.0000000,
            ]
        ),
    )
    test_r = 0.3
    # Value based on RobustRankAggreg
    expected_sum = 0.2853258
    actual_sum = sum_stuart(test_v, test_r)
    assert actual_sum == pytest.approx(expected_sum)


def test_q_stuart(rank_matrix):
    # Based on RobustRankAggreg implementation
    # NOTE: This is after the rank matrix was sorted by index,
    # and across the rows
    expected_q_stuart = np.array(
        [
            0.833625,
            0.908875,
            0.27675,
            0.223875,
            0.142625,
            0.833625,
            0.488,
            0.385875,
            0.488,
            0.271,
            0.05425,
            0.142625,
            0.26125,
            0.484,
            0.875,
            0.488,
            0.142625,
            0.657,
            0.271,
            0.385875,
        ]
    )
    # Get the sorted rank matrix
    sorted_rank_matrix: FloatMatrix2D = cast(
        FloatMatrix2D,
        np.apply_along_axis(np.sort, 1, rank_matrix.sort_index().to_numpy()),
    )
    for row_idx in range(sorted_rank_matrix.shape[0]):
        row = cast(FloatMatrix1D, sorted_rank_matrix[row_idx, :])
        actual_q_stuart = q_stuart(row)
        assert actual_q_stuart == pytest.approx(expected_q_stuart[row_idx])


def test_stuart(rank_matrix):
    expected_stuart = np.array(
        [
            0.833625,
            0.908875,
            0.27675,
            0.223875,
            0.142625,
            0.833625,
            0.488,
            0.385875,
            0.488,
            0.271,
            0.05425,
            0.142625,
            0.26125,
            0.484,
            0.875,
            0.488,
            0.142625,
            0.657,
            0.271,
            0.385875,
        ]
    )
    actual_stuart = stuart(rank_matrix.sort_index().to_numpy())
    np.testing.assert_almost_equal(actual_stuart, expected_stuart)
