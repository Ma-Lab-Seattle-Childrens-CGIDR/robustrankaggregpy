"""
Functions for performing rank aggregation
"""

from functools import reduce

from typing import Hashable, Optional

import numpy as np
import pandas as pd


def create_rank_matrix(
    rank_lists: list[list[Hashable]],
    total_elems: Optional[int | list[int]] = None,
    full: bool = True,
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
