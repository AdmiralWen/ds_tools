"""
Ranking Evaluation Metrics
Author: Brandon Wen

Contains metrics for evaluating ranking models and search indices.

mean_reciprocal_rank: Computes the mean reciprocal rank (MRR) of the evaluation dataframe.
mean_recall_at_k: Computes the mean recall at k of the evaluation dataframe.
"""

import pandas as pd
from typing import List
from statistics import mean

# MRR
def mean_reciprocal_rank(
        eval_df: pd.DataFrame,
        search_rankings: List,
        id_field: str
) -> float:
    """
    Calculates the Mean Reciprocal Rank (MRR) of the evaluation dataframe.

    Args:
        eval_df (pd.DataFrame): evaluation dataframe
        search_rankings (List): list containing the results from the index
    Returns:
        mean(rec_rank_list) (float): mrr
    """
    rec_rank_list = []
    for rel_id, sr in zip(eval_df[id_field], search_rankings):
        rec_rank_list.append(_first_relevant_rr(rel_id, sr))
    return mean(rec_rank_list)

def _first_relevant_rr(rel_ids: List, search_rslt: List) -> float:
    """
    Helper function for MRR. Returns the reciprocal rank of the first relevant id.
    If the search results do not contain any relevant id, mrr=0.

    Args:
        rel_ids (List): list of relevant ids
        search_result (List): list of search results
    Returns:
        first_relevant_rr (float): reciprocal rank of first relevant id
    """
    relevant_ranks = [
        search_rslt.index(i)+1 for i in rel_ids if i in search_rslt
    ]
    if not relevant_ranks:
        first_relevant_rr = 0.0
    else:
        first_relevant_rr = 1/min(relevant_ranks)
    return first_relevant_rr

# Recall at k
def mean_recall_at_k(
        eval_df: pd.DataFrame,
        search_rankings: List,
        id_field: str,
        k: int
) -> float:
    """
    Calculates the Mean Recall at k for the given evaluation dataframe.
    It is the proportion of relevant items found in the top k results.

    Args:
        eval_df (pd.DataFrame): evaluation dataframe
        search_rankings (List): list containing the results from the index
        k (int): top k results to use for evaluation
    Returns:
        mean(reck_scores) (float): mean recall score at k.
    """
    reck_scores = []
    for rel_id, sr in zip(eval_df[id_field], search_rankings):
        reck_scores.append(
            len(
                set(sr[:k]) & set(rel_id)
            )/min(len(rel_id), k) if rel_id else 0
        )
    return mean(reck_scores)
