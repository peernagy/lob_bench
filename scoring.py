import numpy as np
import pandas as pd
from typing import Callable, Iterable, List
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
        plot_fn = lambda title: plotting.hist(
            scores_real,
            scores_gen,
            bins=np.unique(thresholds),
            title=title,
        )
        return score_df, plot_fn
    else:
        return score_df

def calc_metric(
        loader: data_loading.Simple_Loader,
        scoring_fn: Callable,
        metric_fn: Callable | Iterable[Callable],
        **kwargs,
    ) -> tuple[float, Callable]:
    """
    """
    score_df, plot_fn = score_data(loader, scoring_fn, return_plot_fn=True, **kwargs)
    # TODO: integrate bootstrapping into metric fn
    #       either by making all metric jns jax compatible and vmapping
    #       ... or by adding a bootstrapping layer on top of the metric
    #       --> for loop over bootstrap samples
    if hasattr(metric_fn, "__iter__"):
        metric = [m(score_df) for m in metric_fn]
    else:
        metric = metric_fn(score_df)
    return metric, plot_fn
