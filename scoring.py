import numpy as np
import pandas as pd
from typing import Callable
import plotting
import partitioning
import data_loading


def score_data(
        loader: data_loading.Simple_Loader,
        scoring_fn: Callable[[pd.DataFrame, pd.DataFrame], float],
        *,
        return_plot_fn=False,
        discrete=False,
        **kwargs,
    ):
    """
    """
    scores_real, scores_gen = partitioning.score_real_gen(loader, scoring_fn)

    groups_real, groups_gen, thresholds = partitioning.group_by_score(
        scores_real, scores_gen,
        return_thresholds=True,
        discrete=discrete,
        **kwargs
    )
    score_df = partitioning.get_score_table(scores_real, scores_gen, groups_real, groups_gen)

    if return_plot_fn:
        plot_fn = lambda: plotting.hist(
            scores_real,
            scores_gen,
            bins=np.unique(thresholds)
        )
        return score_df, plot_fn
    else:
        return score_df

def calc_metric(
        loader,
        scoring_fn,
        metric_fn,
        **kwargs,
    ):
    """
    """
    score_df = score_data(loader, scoring_fn, **kwargs)
    metric = metric_fn(score_df)
    return metric
