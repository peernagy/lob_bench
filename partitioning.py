"""
"""

import itertools
import warnings
from typing import Callable, Iterable, Optional, Union
import numpy as np
import pandas as pd
import data_loading


# ---------------------------------------------------------------------------
# Worker-pool configuration (set via set_n_workers before scoring calls)
# ---------------------------------------------------------------------------
_N_WORKERS = 1


def set_n_workers(n: int) -> None:
    """Set the number of parallel workers for sequence-level scoring."""
    global _N_WORKERS
    _N_WORKERS = max(1, n)


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
    # return ltype(l)
    # return np.array(tuple(l)).flatten()
    return np.hstack(l)#, casting='safe')

def _get_duplicates(x):
    return np.insert(x[1:] == x[:-1], 0, np.array(False))

def _remove_multiple_duplicates(x: np.array):
    is_duplicated = _get_duplicates(x)
    is_duplicated_multiple = np.insert(is_duplicated[1:] & is_duplicated[:-1], 0, np.array(False))
    return x[~is_duplicated_multiple]

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
            real_id=0,
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
            real_id=0,
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
        """
        """
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

def _align_messages_book(
        messages: Optional[pd.DataFrame],
        book: Optional[pd.DataFrame],
    ) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], bool]:
    if messages is None or book is None:
        return None, None, False
    if len(messages) == len(book):
        return messages, book, False
    min_len = min(len(messages), len(book))
    if min_len == 0:
        return None, None, True
    return messages.iloc[:min_len], book.iloc[:min_len], True

def _score_cond_one(seq, scoring_fn):
    """Score one sequence's conditional data (helper for joblib dispatch)."""
    messages, book, trimmed = _align_messages_book(seq.m_cond, seq.b_cond)
    if trimmed:
        warnings.warn(
            "Conditional messages/book lengths differ; trimming to min length.",
            RuntimeWarning,
            stacklevel=2,
        )
    if messages is None or book is None:
        return np.nan
    return scoring_fn(messages, book)


def score_cond(
        seqs: Iterable[data_loading.Lobster_Sequence],
        scoring_fn: Callable[[pd.DataFrame, pd.DataFrame], float],
    ):
    """
    """
    if _N_WORKERS > 1:
        from joblib import Parallel, delayed
        seqs_list = list(seqs)
        scores = Parallel(n_jobs=_N_WORKERS, backend='loky')(
            delayed(_score_cond_one)(seq, scoring_fn) for seq in seqs_list
        )
        scores = np.array(scores)
    else:
        scores = []
        warned_sequences = set()
        for seq_idx, seq in enumerate(seqs):
            messages, book, trimmed = _align_messages_book(seq.m_cond, seq.b_cond)
            if trimmed and seq_idx not in warned_sequences:
                warnings.warn(
                    "Conditional messages/book lengths differ; trimming to min length.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                warned_sequences.add(seq_idx)
            if messages is None or book is None:
                scores.append(np.nan)
                continue
            scores.append(scoring_fn(messages, book))
        scores = np.array(scores)
    return scores

def score_real_gen(
        seqs: Iterable[data_loading.Lobster_Sequence],
        scoring_fn: Callable[[pd.DataFrame, pd.DataFrame], float],
    ) -> list:
    """
    """
    scores_real = _score_data(seqs, scoring_fn, score_real=True)
    scores_gen = _score_data(seqs, scoring_fn, score_real=False)
    return scores_real, scores_gen

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
    if _N_WORKERS > 1:
        from joblib import Parallel, delayed
        seqs_list = list(seqs)
        scores = Parallel(n_jobs=_N_WORKERS, backend='loky')(
            delayed(_score_seq)(seq, scoring_fn, score_real) for seq in seqs_list
        )
    else:
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
        if isinstance(messages[0], data_loading.Lazy_Tuple) \
        or isinstance(messages[0], tuple) \
        or isinstance(messages[0], list):

            score = tuple(
                tuple(scoring_fn(m_real_i, b_real_i) for m_real_i, b_real_i in zip(m_subseq, b_subseq)) \
                for m_subseq, b_subseq in zip(messages, book)
            )
        else:
            score = tuple(scoring_fn(m_real_i, b_real_i) for m_real_i, b_real_i in zip(messages, book))
    else:
        score = scoring_fn(messages, book)
    return score

def group_by_score(
    scores_real: Iterable,
    scores_gen: Optional[Iterable[Iterable]] = None,
    *,
    bin_method: Optional[str] = None,
    n_bins: Optional[int] = None,
    quantiles: Optional[list[float]] = None,
    thresholds: Optional[list[float]] = None,
    return_thresholds: bool = False,
    discrete: bool = False,
) -> tuple[list[float], list[float]]:
    """ TODO: think about quantile behaviour when large numbers of discrete values
                occur, resulting in multiple quantiles to be the same. Currently,
                this will put all the data up until the first non-unique threshold
                in the same group/bin.
    """

    if scores_gen is None:
        scores_gen = []

    all_scores = np.concatenate(
        (
            flatten(scores_real),
            flatten(scores_gen)
        ),
        casting='safe'
    ).astype(float)

    min_score, max_score = all_scores.min(), all_scores.max()
    if discrete:
        thresholds = np.unique(all_scores)
    else:
        if bin_method is not None:
            # ignore nan scores
            all_scores = all_scores[(~np.isnan(all_scores)) & (~np.isinf(all_scores))]
            thresholds = np.histogram_bin_edges(all_scores, bins=bin_method)
        elif n_bins is not None:
            # thresholds = np.linspace(min_score, max_score, n_bins+1)
            thresholds = np.linspace(min_score, max_score, n_bins+1)[1:-1]
        elif quantiles is not None:
            # thresholds = np.concatenate([[min_score], np.quantile(all_scores, quantiles), [max_score]])
            thresholds = np.quantile(all_scores, quantiles)
            # remove thresholds occuring more than 2x
            thresholds = _remove_multiple_duplicates(thresholds)
            # add a very small delta to the last repeated threshold to make grouping work
            thresholds[_get_duplicates(thresholds)] += 1e-2
        elif thresholds is not None:
            # thresholds = np.concatenate([[min_score], thresholds, [max_score]])
            pass
        else:
            raise ValueError("Must provide either bin_method, n_bins, quantiles, or thresholds.")

    # single (real) sequence
    if (len(scores_real) == 0) or (not hasattr(scores_real[0], '__iter__')):
        # groups_real = np.searchsorted(thresholds, scores_real, side='right') - 1
        groups_real = np.searchsorted(thresholds, scores_real, side='right')
        groups_gen = [
            # np.searchsorted(thresholds, sg_i, side='right') - 1
            np.searchsorted(thresholds, sg_i, side='right')
            for sg_i in scores_gen
        ]
    # subsequences
    else:
        groups_real = [
            # np.searchsorted(thresholds, sr, side='right') - 1
            np.searchsorted(thresholds, sr, side='right')
            for sr in scores_real
        ]
        groups_gen = [
            # tuple(np.searchsorted(thresholds, sg_subseq, side='right') - 1 for sg_subseq in sg_i)
            tuple(np.searchsorted(thresholds, sg_subseq, side='right') for sg_subseq in sg_i)
            for sg_i in scores_gen
        ]

    if return_thresholds:
        thresholds = np.concatenate([[min_score], thresholds, [max_score]])
        return groups_real, groups_gen, thresholds
    else:
        return groups_real, groups_gen

def group_by_subseq(
        subseqs: Iterable[data_loading.Lobster_Sequence],
    ) -> list:
    """
    """
    groups_real = [
        np.arange(len(s.m_real)) for s in subseqs
    ]
    groups_gen = [
        tuple(np.arange(len(m)) for m in s.m_gen) for s in subseqs
    ]
    return groups_real, groups_gen

def get_score_table(
        scores_real: Optional[Iterable],
        scores_gen: Optional[Iterable[Iterable]],
        groups_real: Optional[Iterable],
        groups_gen: Optional[Iterable[Iterable]],
    ) -> pd.DataFrame:
    """
    """
    # use groups = scores if not provided
    if (groups_real is None):
        groups_real = scores_real
    if (groups_gen is None):
        groups_gen = scores_gen

    # REAL DATA
    if scores_real is not None:
        scores_real_flat = flatten(scores_real)
        groups_real_flat = flatten(groups_real)
        assert len(scores_real_flat) == len(groups_real_flat), f"Length mismatch: {len(scores_real_flat)} != {len(groups_real_flat)}"
        real_data = [(sg, g, 'real') for sg, g in zip(scores_real_flat, groups_real_flat)]

        if hasattr(scores_real[0], '__iter__'):
            real_data = [
                (sr, g, 'real') \
                    for scores_i, groups_i in zip(scores_real, groups_real) \
                        for sr, g in zip(scores_i, groups_i)
            ]
        else:
            real_data = [
                (sr, g, 'real') \
                    for sr, g in zip(scores_real, groups_real) \
            ]
    else:
        real_data = []

    # GENERATED DATA
    if scores_gen is not None:
        # scores_gen_flat = flatten(scores_gen)
        # groups_gen_flat = flatten(groups_gen)
        # assert len(scores_gen_flat) == len(groups_gen_flat), f"Length mismatch: {len(scores_gen_flat)} != {len(groups_gen_flat)}"
        # gen_data = [(sg, g, 'generated') for sg, g in zip(scores_gen_flat, groups_gen_flat)]

        if hasattr(scores_gen[0][0], '__iter__'):
            gen_data = [
                (sg, g, 'generated') \
                    for scores_i, groups_i in zip(scores_gen, groups_gen) \
                        for scores_ij, groups_ij in zip(scores_i, groups_i) \
                            for sg, g in zip(scores_ij, groups_ij)

            ]
        else:
            gen_data = [
                (sg, g, 'generated') \
                    for scores_i, groups_i in zip(scores_gen, groups_gen) \
                        for sg, g in zip(scores_i, groups_i) \
            ]
    else:
        gen_data = []

    data = real_data + gen_data
    df = pd.DataFrame(data, columns=['score', 'group', 'type']).explode('score')
    return df
