"""
Functions for performing rank aggregation
"""

from functools import reduce
from typing import cast, Hashable, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import special


def create_rank_matrix(
    rank_lists: list[list[Hashable]],
    total_elems: Optional[int | list[int]] = None,
    full: bool = False,
) -> pd.DataFrame:
    """
    Create a rank matrix by converting a list of rank lists into a dataframe

    Parameters
    ----------
    rank_lists : list of lists of Hashable
        The rank lists to create the rank matrix from. Each list
        should be in order starting from rank 1 (the first element)
    total_elems : int or list of int, optional
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
    if total_elems is None:
        total_elems = len(unique_elems)
    # Create an index that will be used for the columns of the rank matrix
    rank_list_index = pd.Index(range(num_lists))
    # Create the Empty rank matrix, fill with NaN if the ranks are full
    # fill with 1.0 if the ranks are not full
    if not full:
        fill_elem: float = 1.0
        # Convert the total_elems to a series with the same index as the
        # rank_mat
        if isinstance(total_elems, int):
            total_elems = [total_elems] * num_lists
    else:
        fill_elem = np.nan
        total_elems = list(map(len, rank_lists))
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
            / total_elems[idx]
        )
    return rank_matrix


# region Stuart-Aerts
# NOTE: This seems to recalculate the factorial for each row, there
# should be a way to memoize that...

# Define some useful Types
FloatMatrix1D = np.ndarray[Tuple[int], np.dtype[np.float32 | np.float64]]
FloatMatrix2D = np.ndarray[Tuple[int, int], np.dtype[np.float32 | np.float64]]


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
