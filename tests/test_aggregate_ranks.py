from typing import cast

import numpy as np
import pandas as pd
import pytest

from robustrankaggregpy.aggregate_ranks import (
    aggregate_ranks,
    beta_scores,
    threshold_beta_score,
    create_rank_matrix,
    rank_matrix_from_df,
    q_stuart,
    sum_stuart,
    stuart,
    rho_scores,
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


def test_rank_matrix_from_df():
    test_data = pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "B": [2, 1, np.nan, 4, 3],
            "C": [3, 3, 2, np.nan, np.nan],
        },
        index=pd.Index(["a", "b", "c", "d", "e"]),
    )
    expected_not_full_ascending = pd.DataFrame(
        {
            "A": [0.2, 0.4, 0.6, 0.8, 1.0],
            "B": [0.4, 0.2, 1.0, 0.8, 0.6],
            "C": [0.6, 0.6, 0.2, 1.0, 1.0],
        },
        index=pd.Index(["a", "b", "c", "d", "e"]),
    )
    expected_full_ascending = pd.DataFrame(
        {
            "A": [0.2, 0.4, 0.6, 0.8, 1.0],
            "B": [0.4, 0.2, np.nan, 0.8, 0.6],
            "C": [0.6, 0.6, 0.2, np.nan, np.nan],
        },
        index=pd.Index(["a", "b", "c", "d", "e"]),
    )
    expected_not_full_not_ascending = pd.DataFrame(
        {
            "A": [1.0, 0.8, 0.6, 0.4, 0.2],
            "B": [0.6, 0.8, 1.0, 0.2, 0.4],
            "C": [0.4, 0.4, 0.6, 1.0, 1.0],
        },
        index=pd.Index(["a", "b", "c", "d", "e"]),
    )
    expected_full_not_ascending = pd.DataFrame(
        {
            "A": [1.0, 0.8, 0.6, 0.4, 0.2],
            "B": [0.6, 0.8, np.nan, 0.2, 0.4],
            "C": [0.4, 0.4, 0.6, np.nan, np.nan],
        },
        index=pd.Index(["a", "b", "c", "d", "e"]),
    )
    pd.testing.assert_frame_equal(
        expected_not_full_ascending,
        rank_matrix_from_df(test_data, full=False, ascending=True),
    )
    pd.testing.assert_frame_equal(
        expected_full_ascending,
        rank_matrix_from_df(test_data, full=True, ascending=True),
    )
    pd.testing.assert_frame_equal(
        expected_not_full_not_ascending,
        rank_matrix_from_df(test_data, full=False, ascending=False),
    )
    pd.testing.assert_frame_equal(
        expected_full_not_ascending,
        rank_matrix_from_df(test_data, full=True, ascending=False),
    )


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


def test_beta_scores():
    rank_vector_no_na = np.array([5, 2, 3, 6, 1, 4, 7]) / 7
    rank_vector_na = np.array([np.nan, 5, 2, 3, 6, np.nan, 1, 4, 7]) / 9
    expected_beta_no_na = np.array(
        [0.6600833, 0.6395149, 0.6406551, 0.6531001, 0.6792299, 0.7364861, 1.0000000]
    )
    expected_beta_na = np.array(
        [
            0.5615376,
            0.4834529,
            0.4293553,
            0.3799615,
            0.3273333,
            0.2633745,
            0.1721824,
            np.nan,
            np.nan,
        ]
    )
    np.testing.assert_allclose(beta_scores(rank_vector_no_na), expected_beta_no_na)
    # NOTE: Small floating point difference, increasing tolerance slightly
    np.testing.assert_allclose(
        beta_scores(rank_vector_na), expected_beta_na, rtol=1.1e-7
    )


def test_threshold_beta_scores():
    scores_no_na: FloatMatrix1D = cast(
        FloatMatrix1D, np.array([0.1, 0.4, 0.2, 0.01, 0.17, 0.25, 0.43])
    )
    scores_na: FloatMatrix1D = cast(
        FloatMatrix1D,
        np.array([0.1, 0.4, 0.2, 0.01, 0.17, 0.25, 0.43, np.nan, np.nan, np.nan]),
    )
    sigma_no_na = np.ones(scores_no_na.shape)
    sigma_no_na.fill(0.5)
    sigma_na = np.ones(scores_na.shape)
    sigma_na.fill(0.5)

    expected_thresholded_no_na = np.array(
        [
            0.067934652,
            0.149694400,
            0.100520069,
            0.033344000,
            0.012878418,
            0.018841600,
            0.002718186,
        ]
    )
    expected_thresholded_na = np.array(
        [
            0.09561792,
            0.26390107,
            0.23413055,
            0.12087388,
            0.07812691,
            0.16623862,
            0.08057631,
            1.00000000,
            1.00000000,
            1.00000000,
        ]
    )

    np.testing.assert_allclose(
        threshold_beta_score(scores=scores_no_na, sigma=sigma_no_na),
        expected_thresholded_no_na,
    )

    np.testing.assert_allclose(
        threshold_beta_score(scores=scores_na, sigma=sigma_na),
        expected_thresholded_na,
    )


def test_rho_scores():
    test_r1: FloatMatrix1D = cast(
        FloatMatrix1D,
        np.array(
            [
                0.07919443,
                0.34070727,
                0.04911215,
                0.57717496,
                0.22727211,
                0.37171813,
                0.85979700,
                0.86185242,
                0.08082894,
                0.90504901,
                0.03027985,
                0.51429599,
                0.88828221,
                0.75317915,
                0.99846283,
            ]
        ),
    )
    test_r2: FloatMatrix1D = cast(
        FloatMatrix1D,
        np.array(
            [
                0.200770238,
                0.889720872,
                0.787357739,
                0.927841279,
                0.019845115,
                0.677542569,
                0.224317168,
                0.254374503,
                0.309581592,
                0.719109173,
                0.124438156,
                0.031504511,
                0.031702428,
                0.036366031,
                0.003931667,
            ]
        ),
    )
    expected_rho_1 = 0.4237422
    expected_tho_2 = 0.002108783
    assert rho_scores(test_r1) == pytest.approx(expected_rho_1)
    assert rho_scores(test_r2) == pytest.approx(expected_tho_2)


def test_aggregate_ranks():
    rank_lists = [
        ["u", "f", "c", "w"],
        ["h", "f", "j", "y", "e", "q", "p", "k", "r", "v"],
        ["q", "e", "f", "d", "k", "c", "x", "j", "m", "r", "t", "z"],
    ]
    ranked_elements = 26

    # RRA aggregation
    expected_rra = pd.Series(
        {
            "f": 0.004608557,
            "e": 0.290168411,
            "u": 0.333010924,
            "h": 0.333010924,
            "q": 0.333010924,
            "c": 0.405553027,
            "j": 0.677287210,
            "k": 0.677287210,
            "r": 0.989986345,
            "w": 1.000000000,
            "y": 1.000000000,
            "p": 1.000000000,
            "v": 1.000000000,
            "d": 1.000000000,
            "x": 1.000000000,
            "m": 1.000000000,
            "t": 1.000000000,
            "z": 1.000000000,
        }
    )
    actual_rra = aggregate_ranks(rank_lists=rank_lists, ranked_elements=ranked_elements)
    pd.testing.assert_series_equal(
        expected_rra, actual_rra, check_exact=False, check_like=True
    )
