"""
"""

import itertools
from typing import Callable, Iterable, Optional, Union
import numpy as np
import pandas as pd
import data_loading


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

    if subseq_len is not None:
        num_subseqs = np.ceil(len(m_real) / subseq_len)

    if num_subseqs is not None:
        m_real = np.array_split(m_real, num_subseqs)
        b_real = np.array_split(b_real, num_subseqs)
        m_gen = tuple(np.array_split(m, num_subseqs) for m in m_gen)
        b_gen = tuple(np.array_split(b, num_subseqs) for b in b_gen)

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
    scores = []
    for seq in seqs:
        m_real = seq.m_real
        b_real = seq.b_real
        if isinstance(m_real, tuple):
            scores_i = tuple(scoring_fn(m_real_i, b_real_i) for m_real_i, b_real_i in zip(m_real, b_real))
            scores.append(scores_i)
        else:
            scores.append(scoring_fn(m_real, b_real))
    return scores

def group_by_score(
        scores: Iterable,
        scores_gen: Optional[Iterable[Iterable]] = [],
        *,
        n_quantiles: Optional[Union[int, list[float]]] = None,
        group_thresholds: Optional[float] = None,

    ) -> list:
    """
    """
    all_scores = np.array(list(itertools.chain(*scores, *scores_gen)))

