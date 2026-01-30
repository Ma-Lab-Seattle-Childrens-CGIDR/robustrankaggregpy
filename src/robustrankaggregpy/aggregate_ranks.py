"""
Functions for performing rank aggregation
"""

from functools import reduce
from typing import cast, Hashable, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import special, stats


def create_rank_matrix(
    rank_lists: list[list[Hashable]],
    ranked_elements: Optional[int | list[int]] = None,
    full: bool = False,
) -> pd.DataFrame:
    """
    Create a rank matrix by converting a list of rank lists into a dataframe

    Parameters
    ----------
    rank_lists : list of lists of Hashable
        The rank lists to create the rank matrix from. Each list
        should be in order starting from rank 1 (the first element)
    ranked_elements : int or list of int, optional
        The total number of elements being ranked, if not provided
        will use the number of unique elements found across all rank lists.
        If a single int, that is used as the total number of elements for
        all the lists, if a list should be the same length as rank_lists, and
        each list will have a different total number of elements specified.
    full : bool, default=True
        Whether the given ranks are complete

    Returns
    -------
    rank_matrix : pd.DataFrame
        The rank matrix describing the ranks across the lists, with
        a column for each input list, and a row for each unique element
        found in any list
    """
    # Find the unique elements across all the lists
    unique_elems: set[Hashable] = reduce(lambda x, y: x | set(y), rank_lists, set())
    num_lists = len(rank_lists)
    if ranked_elements is None:
        ranked_elements = len(unique_elems)
    # Create an index that will be used for the columns of the rank matrix
    rank_list_index = pd.Index(range(num_lists))
    # Create the Empty rank matrix, fill with NaN if the ranks are full
    # fill with 1.0 if the ranks are not full
    if not full:
        fill_elem: float = 1.0
        # Convert the total_elems to a series with the same index as the
        # rank_mat
        if isinstance(ranked_elements, int):
            ranked_elements = [ranked_elements] * num_lists
    else:
        fill_elem = np.nan
        ranked_elements = list(map(len, rank_lists))
    rank_matrix = pd.DataFrame(
        fill_elem,
        index=pd.Index(unique_elems),
        columns=rank_list_index,
    )
    # Fill in the rank matrix
    for idx in range(num_lists):
        rank_matrix.loc[rank_lists[idx], idx] = (
            pd.Series(
                range(1, len(rank_lists[idx]) + 1), index=pd.Index(rank_lists[idx])
            )
            / ranked_elements[idx]
        )
    return rank_matrix


# region Stuart-Aerts
# NOTE: This seems to recalculate the factorial for each row, there
# should be a way to memoize that...

# Define some useful Types
FloatMatrix1D = np.ndarray[Tuple[int], np.dtype[np.float32 | np.float64]]
FloatMatrix2D = np.ndarray[Tuple[int, int], np.dtype[np.float32 | np.float64]]
IntMatrix1D = np.ndarray[Tuple[int], np.dtype[np.int32 | np.int64]]


def sum_stuart(v: FloatMatrix1D, r: float) -> float:
    """
    Helper function for Stuart-Aerts method

    Parameters
    ----------
    v : 1-D numpy array of floats
        The array to compute the Stuart-Aerts sum for
    r : float
        The rank ratio to compute the Stuart-Aerts sum for

    Returns
    -------
    sum
        The Stuart-Aerts sum of v with rank ratio r
    """
    k = len(v)
    l_k = np.arange(1, k + 1)
    ones: FloatMatrix1D = (-1) ** (l_k + 1)
    f: FloatMatrix1D = special.factorial(l_k)
    p: FloatMatrix1D = cast(FloatMatrix1D, r**l_k)
    return ones @ (np.flip(v) * p / f)


def q_stuart(row: FloatMatrix1D) -> float:
    """
    Calculate the Q-statistic for a single row of a matrix

    Parameters
    ----------
    row : 1-D numpy NDArray
        The row to calculate the Q-statistic for

    Returns
    -------
    q : float
        The Q-statistic for the row
    """
    # Get the number of non-NaN entries
    non_na_count: int = np.sum(~np.isnan(row))
    v: FloatMatrix1D = np.ones((non_na_count + 1,))
    for k in range(non_na_count):
        v[k + 1] = sum_stuart(v[0 : (k + 1)], row[non_na_count - k - 1])
    return special.factorial(non_na_count) * v[non_na_count]


def stuart(rank_matrix: FloatMatrix2D):
    """
    Compute the Stuart ranks for each row in a 2-D numpy array

    Parameters
    ----------
    rank_matrix : 2-D numpy NDArray
        The rank matrix to compute the Stuart-Aerts ranks for

    Returns
    -------
    ranks : 1-D numpy NDArray
        The ranks of the rows in the rank_matrix
    """
    rank_matrix: FloatMatrix2D = cast(
        FloatMatrix2D, np.apply_along_axis(np.sort, 1, rank_matrix)
    )
    return np.apply_along_axis(q_stuart, 1, rank_matrix)


# endregion Stuart-Aerts


# region RRA
def beta_scores(rank_vector: FloatMatrix1D) -> FloatMatrix1D:
    """
    Calculate beta scores for a normalized rank vector

    Parameters
    ----------
    rank_vector : FloatMatrix1D
        Vector of normalized ranks (aka rank ratios), should be values in the range [0,1]

    Returns
    -------
    scores : FloatMatrix1D
        p-values calculated using the beta distribution
    """
    count: int = np.sum(~np.isnan(rank_vector))
    pvec: FloatMatrix1D = np.empty(rank_vector.shape, dtype=rank_vector.dtype)
    pvec.fill(np.nan)
    sorted_rank_vector: FloatMatrix1D = cast(FloatMatrix1D, np.sort(rank_vector))
    # Shape parameters for the beta distribution
    a = np.arange(count) + 1
    b = np.flip(a)
    pvec[:count] = stats.beta.cdf(sorted_rank_vector[:count], a, b)
    return pvec


def threshold_beta_score(
    scores: FloatMatrix1D,
    k: Optional[IntMatrix1D] = None,
    n: Optional[int] = None,
    sigma: Optional[FloatMatrix1D] = None,
):
    """
    Threshold the Beta Scores, used when the lists being
    aggregated are only the top sigma portion of the rankings

    Parameters
    ----------
    scores : FloatMatrix1D
        Beta scores to threshold
    k : IntMatrix1D, optional
    n : int, optional
    sigma : FloatMatrix1D, optional
        The thresholds

    Returns
    -------
    FloatMatrix1D
        The thresholded beta scores
    """
    if k is None:
        k: IntMatrix1D = np.arange(scores.shape[0], dtype=int) + 1
    if n is None:
        n = scores.shape[0]
    if sigma is None:
        sigma: FloatMatrix1D = np.ones((n,), dtype=scores.dtype)

    # Check the input parameters
    if len(sigma) != n:
        raise ValueError(
            f"Length of Sigma must match n, but sigma length is {len(sigma)}, and n is {n}"
        )
    if len(scores) != n:
        raise ValueError(
            f"Length of the scores must match n, but scores has length {len(scores)}, and n is {n}"
        )
    if np.min(sigma) < 0.0 or np.max(sigma) > 1.0:
        raise ValueError(
            f"Elements of sigma must be in rane [0,1], but actual range was [{np.min(sigma)},{np.max(sigma)}]"
        )
    if any((~np.isnan(scores)) & (scores > sigma)):
        raise ValueError("Elements of scores must be smaller than elements of sigma")

    # Get a vector with no NaN
    x: FloatMatrix1D = cast(FloatMatrix1D, np.sort(scores[~np.isnan(scores)]))
    # Sort the sigma vector in descending order
    sigma: FloatMatrix1D = cast(FloatMatrix1D, np.flip(np.sort(sigma)))
    # Create the thresholded beta vector, filled with NaNs for now
    beta: FloatMatrix1D = np.empty((len(k)), dtype=scores.dtype)
    beta.fill(np.nan)
    # For each value of K
    for idx in range(len(k)):
        if k[idx] > n:
            beta[idx] = 0
            continue
        if k[idx] > len(x):
            beta[idx] = 1
            continue
        if sigma[n - 1] >= x[k[idx] - 1]:
            beta[idx] = stats.beta.cdf(x[k[idx] - 1], k[idx], n + 1 - k[idx])
            continue

        # Find the last element such that sigma[n0] <= x[k[idx]]
        n0 = np.searchsorted(sigma, x[k[idx] - 1], side="left")

        # Compute beta(n,k) for n=n0 and k=1..k[idx]
        b = np.zeros((k[idx],), dtype=scores.dtype)
        b[0] = 1.0
        if n0 == 0:
            pass
        elif k[idx] > n0:
            a_param = np.arange(1, n0 + 1)
            b[1:n0] = stats.beta.cdf(x[k[idx] - 1], a_param, np.flip(a_param))
        else:
            a_param = np.arange(1, k[idx])
            b_param = n0 + 2 - a_param
            b[1:] = stats.beta.cdf(x[k[idx] - 1], a_param, b_param)

        z = sigma[(n0 + 1) : n + 1]
        for j in range(n - n0):
            b[1 : (k[idx])] = (1 - z[j]) * b[1 : k[idx]] + z[j] * b[0 : k[idx] - 1]
        beta[idx] = b[k[idx]]
    return beta


def correct_beta_pvalues(pvalues: FloatMatrix1D, k: int) -> float:
    return min(np.min(pvalues * k), 1.0)


def correct_beta_pvalues_exact(pvalues: FloatMatrix1D, k: int) -> float:
    rm = np.empty((len(pvalues), k), dtype=pvalues.dtype)
    a = np.arange(1, k + 1, dtype=int)
    b = np.flip(a)
    for idx, p in pvalues:
        rm[idx, :] = stats.beta.ppf(p, a=a, b=b)
    return 1 - stuart(1 - rm)


def rho_scores(
    r: FloatMatrix1D, top_cutoff: Optional[FloatMatrix1D] = None, exact: bool = False
) -> float:
    """
    Calculate the rho scores for a row of the rank matrix

    Parameters
    ----------
    r : FloatMatrix1D
        Normalized ranks to calculate the rho scores for (must be in range [0,1])
    top_cutoff : FloatMatrix1D, optional
        Cutoff values used to limit the number of elements in the input lists
    exact : bool
        Whether to calculate exact p-values (which is computationally expensive
        and unstable, and does not provide a lot of benefit in most cases)

    Returns
    -------
    rho_score : float
        The rho score of the normalized rank vector
    """
    if top_cutoff is None:
        x = beta_scores(r)
    else:
        r = r[~np.isnan(r)]
        r[r == 1.0] = np.nan
        x = threshold_beta_score(r, sigma=top_cutoff)
    if exact:
        rho = correct_beta_pvalues_exact(np.nanmin(x), np.sum(~np.isnan(x)))
    else:
        rho = correct_beta_pvalues(np.nanmin(x), np.sum(~np.isnan(x)))
    return rho


# endregion RRA


# region Main Function
def aggregate_ranks(
    rank_lists: Optional[list[list[Hashable]]] = None,
    rank_matrix: Optional[pd.DataFrame] = None,
    ranked_elements: Optional[int] = None,
    method: Literal["rra", "min", "geom-mean", "mean", "median", "stuart"] = "rra",
    full: bool = False,
    exact: bool = False,
    top_cutoff: Optional[FloatMatrix1D | np.typing.ArrayLike] = None,
) -> pd.Series:
    """
    Aggregate ranked lists

    Parameters
    ----------
    rank_lists : list of list of Hashable
        The ranked lists to aggregate, each list should be
        ordered from lowest rank to highest (i.e. from rank 1 to n)
    rank_matrix : pd.DataFrame, optional
        The ranking in a matrix format (like that from create_rank_matrix),
        by default the rank_lists will be automatically converted into this form
    ranked_elements : int, optional
        The number of ranked elements, by default it is calculated as the number of unique
        elements in the input rankings
    method : Literal 'rra' or 'min' or 'geom-mean' or 'mean' or 'median' or 'stuart', default='rra'
        The method to use for aggregating the ranks
    full : bool, default=False
        Whether the full rankings are given
    exact : bool, default=False
        Whether the exact p-value should be calculated based on the rho scores
    top_cutoff : FloatMatrix1D or ArrayLike, optional
        The cutoff values used to limit the number of elements in each of the input
        lists, should be the proportion of the list which is provided (so if
        there are 1000 elements being ranked, but the first list is
        limited to 100, the second to 200, and the third to 900,
        this should be [0.1, 0.2, 0.9]).

    Returns
    -------
    aggregated_ranks : pd.Series
        The aggregated ranks, the index is the elements in rank order, and the
        values are the associated score or p-value

    Note
    ----
    The top_cutoff parameter is the proportion
    """
    # Create the rank matrix
    if rank_matrix is None:
        if rank_lists is None:
            raise ValueError(
                "Either rank_lists or rank_matrix must be provided, but received neither"
            )
        else:
            rank_matrix = create_rank_matrix(
                rank_lists=rank_lists, ranked_elements=ranked_elements, full=full
            )
    # Convert the sigma values if needed
    if top_cutoff is not None and not isinstance(top_cutoff, np.ndarray):
        top_cutoff = np.array(top_cutoff)
    # Get the number of elements to be ranked
    if ranked_elements is None:
        ranked_elements = len(rank_matrix.index)
    # Get the numpy array representing the rank matrix
    rank_matrix_array = rank_matrix.to_numpy()
    match method:
        case "min":
            aggregated_matrix = np.apply_along_axis(np.nanmin, 1, rank_matrix_array)
        case "median":
            aggregated_matrix = np.apply_along_axis(np.nanmedian, 1, rank_matrix_array)
        case "mean":
            # The RobustRankAggreg library calculates a probability here
            # using a CLT approximation of a sum of normals
            # I don't think this adds a lot, and it conflicts with
            # the other methods (like median, min, etc.)
            # So I am currently removing it, but may be added back later as
            # an option
            aggregated_matrix = np.apply_along_axis(np.nanmean, 1, rank_matrix_array)
        case "geom-mean":
            aggregated_matrix = np.apply_along_axis(
                stats.gmean, 1, rank_matrix_array, nan_policy="omit"
            )
        case "rra":
            aggregated_matrix = np.apply_along_axis(
                rho_scores, 1, rank_matrix_array, top_cutoff=top_cutoff, exact=exact
            )
        case "stuart":
            aggregated_matrix = stuart(rank_matrix_array)
        case _:
            raise ValueError(
                f"Invalid method, expected one of 'min', "
                f"'median', 'mean', 'geom-mean', 'stuart', or 'rra',"
                f"but reveived {method}"
            )
    return pd.Series(aggregated_matrix, index=rank_matrix.index).sort_values(
        ascending=True
    )


# endregion Main Function
