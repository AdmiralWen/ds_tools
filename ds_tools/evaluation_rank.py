"""
Ranking Evaluation Metrics
Author: Brandon Wen

Contains metrics for evaluating ranking models and search indices. All evaluation functions in this module
assumes that you have your evaluation data prepared as a pandas dataframe, where each row represents a query
and the columns contain a column for the correct/relevant results (the ground truth) and a column for the
search results (the "predicted"). For example:

query|search_results           |relevant_results
-----|-------------------------|-----------------
'cat'|"['cat', 'cats', 'dogs']"|"['cat', 'cats']"
'dog'|"['cat', 'cats', 'dog']" |"['dog', 'dogs']"

Evaluation metrics:
    - mean_reciprocal_rank
    - mean_recall_at_k
    - mean_precision_at_k
    - map_at_k (mean average precision at k)
"""

import pandas as pd
from typing import List
from statistics import mean
from math import log2

# MRR
def mean_reciprocal_rank(eval_df: pd.DataFrame, relevant_results_field: str, search_results_field: str) -> float:
    """
    Calculates the Mean Reciprocal Rank (MRR) of the evaluation dataframe.

    Args:
        eval_df (pd.DataFrame): The evaluation dataframe
        relevant_results_field (str): Name of the field containing correct results ("ground-truth")
        search_results_field (str): Name of the field containing search results ("predicted")
    Returns:
        mean(rec_rank_list) (float): Mean reciprocal rank
    """
    rec_rank_list = []
    for rr, sr in zip(eval_df[relevant_results_field], eval_df[search_results_field]):
        rec_rank_list.append(_first_relevant_rr(rr, sr))
    return mean(rec_rank_list)

def _first_relevant_rr(relevant_items: List, search_rslt: List) -> float:
    """
    Helper function for MRR. Returns the reciprocal rank of the first relevant item from the search
    result. If the search results do not contain any relevant id, mrr=0.

    Args:
        relevant_items (List): list of relevant items (the ground truth)
        search_rslt (List): list of search results
    Returns:
        first_relevant_rr (float): reciprocal rank of first relevant id
    """
    relevant_ranks = [
        search_rslt.index(i)+1 for i in relevant_items if i in search_rslt
    ]
    if not relevant_ranks:
        first_relevant_rr = 0.0
    else:
        first_relevant_rr = 1/min(relevant_ranks)
    return first_relevant_rr

# Mean Recall at K
def mean_recall_at_k(
    eval_df: pd.DataFrame,
    relevant_results_field: str,
    search_results_field: str,
    k: int
) -> float:
    """
    Calculates the Mean Recall at k for the given evaluation dataframe. It is the proportion of relevant
    items found in the top k search results, out of the total number of relevant results (up to k).

    Args:
        eval_df (pd.DataFrame): The evaluation dataframe
        relevant_results_field (str): Name of the field containing correct results ("ground-truth")
        search_results_field (str): Name of the field containing search results ("predicted")
        k (int): The top k results to use for evaluation
    Returns:
        mean(reck_scores) (float): Mean recall at k
    """
    reck_scores = []
    for rr, sr in zip(eval_df[relevant_results_field], eval_df[search_results_field]):
        reck_scores.append(len(set(sr[:k]) & set(rr))/min(len(rr), k) if rr else 0)
    return mean(reck_scores)

# Mean Precision at K
def mean_precision_at_k(
    eval_df: pd.DataFrame,
    relevant_results_field: str,
    search_results_field: str,
    k: int
) -> float:
    """
    Calculates the Mean Precision at k for the given evaluation dataframe. It is the proportion of relevant
    items found in the top k search results, out of the total number of search results (up to k).

    Args:
        eval_df (pd.DataFrame): The evaluation dataframe
        relevant_results_field (str): Name of the field containing correct results ("ground-truth")
        search_results_field (str): Name of the field containing search results ("predicted")
        k (int): The top k results to use for evaluation
    Returns:
        mean(preck_scores) (float): Mean precision at k
    """
    preck_scores = []
    for rr, sr in zip(eval_df[relevant_results_field], eval_df[search_results_field]):
        preck_scores.append(len(set(sr[:k]) & set(rr))/min(len(sr), k) if sr else 0)
    return mean(preck_scores)

# Mean Average Precision at K
def map_at_k(
    eval_df: pd.DataFrame,
    relevant_results_field: str,
    search_results_field: str,
    k: int
) -> float:
    """
    Calculates the Mean Average Precision at k for the given evaluation dataframe. For the explanation of
    MAP, please see https://www.evidentlyai.com/ranking-metrics/mean-average-precision-map.

    Args:
        eval_df (pd.DataFrame): The evaluation dataframe
        relevant_results_field (str): Name of the field containing correct results ("ground-truth")
        search_results_field (str): Name of the field containing search results ("predicted")
        k (int): The top k results to use for evaluation
    Returns:
        mean(apk_scores) (float): MAP at k
    """
    apk_scores = []
    for rr, sr in zip(eval_df[relevant_results_field], eval_df[search_results_field]):
        apk_scores.append(_average_precision_at_k(rr, sr, k))
    return mean(apk_scores)

def _average_precision_at_k(relevant_items: List, search_rslt: List, k: int) -> float:
    '''
    Helper function for the map_at_k(). Returns the average precision at k for a given list of
    relevant items and search results.

    Args:
        relevant_items (List): list of relevant items (the ground truth)
        search_rslt (List): list of search results
        k (int): top k results to use for evaluation
    Returns:
        average_precision_at_k (float): average precision at k for the search results
    '''
    relevant_results = 0
    running_sum = 0

    for i, sr in enumerate(search_rslt[:k]):
        if sr in relevant_items:
            relevant_results += 1
            running_sum += relevant_results/(i+1)

    return running_sum/len(relevant_items)

# Mean NDCG at K
def mean_ndcg_at_k(
    eval_df: pd.DataFrame,
    relevant_results_field: str,
    search_results_field: str,
    k: int
) -> float:
    """
    Calculates the Mean NDCG at k for the given evaluation dataframe. For the explanation of
    NDCG, please see https://www.evidentlyai.com/ranking-metrics/ndcg-metric.

    Args:
        eval_df (pd.DataFrame): The evaluation dataframe
        relevant_results_field (str): Name of the field containing correct results ("ground-truth")
        search_results_field (str): Name of the field containing search results ("predicted")
        k (int): The top k results to use for evaluation
    Returns:
        mean(ndcg_scores) (float): Mean NDCG at k
    """
    ndcg_scores = []
    for rr, sr in zip(eval_df[relevant_results_field], eval_df[search_results_field]):
        ndcg_scores.append(_dcg_at_k(rr, sr, k)/_dcg_at_k(rr, rr, k))
    return mean(ndcg_scores)

def _dcg_at_k(relevant_items: List, search_rslt: List, k: int) -> float:
    '''
    Helper function for ndcg_at_k(). Computes the DCG (discounted cumulative gain) at k for a given search result.

    Args:
        relevant_items (List): list of relevant items (the ground truth)
        search_rslt (List): list of search results
        k (int): top k results to use for evaluation
    Returns:
        dcg_at_k (float): discounted cumulative gain at k for the search results
    '''
    cg_list = [(search_rslt[i] in relevant_items)/log2(i+2) for i in range(min(len(search_rslt), k))]
    return sum(cg_list)
