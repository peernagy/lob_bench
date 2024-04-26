"""
"""

import itertools
from typing import Callable, Iterable, Optional, Union
import numpy as np
import pandas as pd
import data_loading


def flatten(l, ltypes=(list, tuple)):
    if isinstance(l, np.ndarray):
        return l.flatten()
    ltype = type(l)
    l = list(l)
    i = 0
    while i < len(l):
        while isinstance(l[i], ltypes):
            if not l[i]:
                l.pop(i)
                i -= 1
                break
            else:
                l[i:i + 1] = l[i]
        i += 1
    return ltype(l)

def get_subseqs(
        seq: data_loading.Lobster_Sequence,
        *,
        num_subseqs: Optional[int] = None,
        subseq_len: Optional[int] = None,
        time_interval: Optional[str] = None
    ) -> data_loading.Lobster_Sequence:
    """
    """
    # only one of the kwargs is allowed
    assert sum(kw is not None for kw in [num_subseqs, subseq_len, time_interval]) == 1
    assert len(seq.num_gen_series) == 1, "Sequences are already split into subsequences."

    m_real = seq.m_real
    b_real = seq.b_real
    m_gen = tuple(seq.m_gen)
    b_gen = seq.b_gen

    if num_subseqs is not None:
        subseq_len = len(m_real) // num_subseqs

    if subseq_len is not None:
        m_real = _split_df(m_real, subseq_len)
        b_real = _split_df(b_real, subseq_len)
        m_gen = tuple(_split_df(m, subseq_len) for m in m_gen)
        b_gen = tuple(_split_df(b, subseq_len) for b in b_gen)

        return data_loading.Lobster_Sequence(
            date=seq.date,
            m_real=m_real,
            b_real=b_real,
            m_gen=m_gen,
            b_gen=b_gen,
            m_cond=seq._m_cond, # don't materialize
            b_cond=seq._b_cond,
            num_gen_series=(seq.num_gen_series[0], num_subseqs)
        )

    elif time_interval is not None:
        m_real, b_real = _split_by_time_interval(m_real, b_real, time_interval)
        # TODO: check for different lengths?
        m_gen, b_gen = zip(*(_split_by_time_interval(m, b, time_interval) for m, b in zip(m_gen, b_gen)))
        num_subseqs = len(m_gen[0])

        return data_loading.Lobster_Sequence(
            date=seq.date,
            m_real=m_real,
            b_real=b_real,
            m_gen=m_gen,
            b_gen=b_gen,
            m_cond=seq._m_cond, # don't materialize
            b_cond=seq._b_cond,
            # TODO: potential bug: this dimension can be different for different gen seqs
            num_gen_series=(seq.num_gen_series[0], num_subseqs)
        )
    
def _split_df(
        df: pd.DataFrame,
        subseq_len: int,
    ) -> list[pd.DataFrame]:
    """
    """
    return [df[i: i+subseq_len] for i in range(0, df.shape[0], subseq_len)]

def _split_by_time_interval(
        messages: pd.DataFrame,
        book: pd.DataFrame,
        time_interval: str
    ) -> tuple[pd.DataFrame]:
        num_message_cols = len(messages.columns)
        df = pd.concat([messages, book], axis=1)
        groups = tuple(group for _, group in df.groupby(pd.Grouper(
            key='time',
            freq=time_interval,
            label='right', closed='left', origin='start_day'
        )))
        m, b = zip(*((group.iloc[:, :num_message_cols], group.iloc[:, num_message_cols: ]) for group in groups))
        return m, b
        # m_fn, b_fn = lambda: m, lambda: b
        # return m_fn, b_fn

""" Scoring Functions """

def score_cond(
        seqs: Iterable[data_loading.Lobster_Sequence],
        scoring_fn: Callable[[pd.DataFrame, pd.DataFrame], float],
    ):
    """
    """
    scores = np.array([scoring_fn(seq.m_cond, seq.b_cond) for seq in seqs])
    return scores

def score_real(
        seqs: Iterable[data_loading.Lobster_Sequence],
        scoring_fn: Callable[[pd.DataFrame, pd.DataFrame], float],
    ) -> list:
    """
    """
    return _score_data(seqs, scoring_fn, score_real=True)

def score_gen(
        seqs: Iterable[data_loading.Lobster_Sequence],
        scoring_fn: Callable[[pd.DataFrame, pd.DataFrame], float],
    ) -> list:
    """ 
    """
    return _score_data(seqs, scoring_fn, score_real=False)

def _score_data(
        seqs: Iterable[data_loading.Lobster_Sequence],
        scoring_fn: Callable[[pd.DataFrame, pd.DataFrame], float],
        score_real: bool,
    ) -> float:
    """
    """
    scores = []
    for seq in seqs:
        score_i = _score_seq(seq, scoring_fn, score_real)
        scores.append(score_i)
    return scores

def _score_seq(
        seq: data_loading.Lobster_Sequence,
        scoring_fn: Callable[[pd.DataFrame, pd.DataFrame], float],
        score_real: bool,
    ) -> float:
    """
    """
    if score_real:
        messages = seq.m_real
        book = seq.b_real
    else:
        messages = seq.m_gen
        book = seq.b_gen
    if isinstance(messages, data_loading.Lazy_Tuple) or isinstance(messages, tuple):
        score = tuple(scoring_fn(m_real_i, b_real_i) for m_real_i, b_real_i in zip(messages, book))
    else:
        score = scoring_fn(messages, book)
    return score

def group_by_score(
        scores_real: Iterable,
        scores_gen: Optional[Iterable[Iterable]] = [],
        *,
        n_bins: Optional[int] = None,
        quantiles: Optional[list[float]] = None,
        thresholds: Optional[list[float]] = None,
    ) -> list:
    """
    """
    all_scores = np.concatenate((flatten(scores_real), flatten(scores_gen)))
    
    min_score, max_score = all_scores.min(), all_scores.max()
    if n_bins is not None:
        thresholds = np.linspace(min_score, max_score, n_bins+1)
    elif quantiles is not None:
        thresholds = np.concatenate([[min_score], np.quantile(all_scores, quantiles), [max_score]])
    elif thresholds is not None:
        thresholds = np.concatenate([[min_score], thresholds, [max_score]])
    else:
        raise ValueError("Must provide either n_bins, quantiles, or thresholds.")
    
    # print(thresholds)
    
    # assert isinstance(scores_gen[0], Iterable), "scores_gen must be an iterable of iterables."
    # subsequences
    if isinstance(scores_real[0], list):
        groups_real = [
            np.searchsorted(thresholds, sr, side='right') - 1
            for sr in scores_real
        ]
        groups_gen = [
            np.searchsorted(thresholds, sg_subseq, side='right') - 1
            for sg_i in scores_gen
            for sg_subseq in sg_i
        ]
    # single (real) sequence
    else:
        groups_real = np.searchsorted(thresholds, scores_real, side='right') - 1
        groups_gen = [
            np.searchsorted(thresholds, sg_i, side='right') - 1
            for sg_i in scores_gen
        ]
    
    return groups_real, groups_gen
