import numpy as np
import pandas as pd
from typing import Callable, Iterable, List, Optional
import plotting
import partitioning
import data_loading


def get_kwargs(score_config):
    kwargs = {
        "discrete": score_config.get("discrete", False),
        "quantiles": score_config.get("quantiles", None),
        "n_bins": score_config.get("n_bins", None),
        "thresholds": score_config.get("thresholds", None),
    }

    # default to quartiles if no quantiles, n_bins or thresholds are specified
    if (kwargs["quantiles"] is None) \
        and (kwargs["n_bins"] is None) \
        and (kwargs["thresholds"] is None):
            kwargs["quantiles"] = [0.25, 0.5, 0.75]

    return kwargs

def score_data(
        loader: data_loading.Simple_Loader,
        scoring_fn: Callable[[pd.DataFrame, pd.DataFrame], float],
        *,
        return_plot_fn=False,
        # discrete=False,
        **kwargs,
    ):
    """
    TODO: make grouping optional -> only return scored in df
    """
    scores_real, scores_gen = partitioning.score_real_gen(loader, scoring_fn)

    groups_real, groups_gen, thresholds = partitioning.group_by_score(
        scores_real, scores_gen,
        return_thresholds=True,
        # discrete=discrete,
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

def score_data_cond(
        loader: data_loading.Simple_Loader,
        scoring_fn: Callable[[pd.DataFrame, pd.DataFrame], float],
        scoring_fn_cond: Callable[[pd.DataFrame, pd.DataFrame], float],
        *,
        return_plot_fn: bool = False,
        score_kwargs: dict = {},
        score_cond_kwargs: dict = {},
    ):
    """
    """
    # calc scores to be conditioned on
    scores_real, scores_gen = partitioning.score_real_gen(loader, scoring_fn_cond)
    groups_real, groups_gen, thresholds = partitioning.group_by_score(
        scores_real, scores_gen,
        return_thresholds=True,
        **score_cond_kwargs
    )
    print('score_real', scores_real)
    print('score_gen', scores_gen)
    print('thresholds', thresholds)

    # calc scores of interest (to be conditioned on prior scores)
    eval_real, eval_gen = partitioning.score_real_gen(loader, scoring_fn)
    score_df = partitioning.get_score_table(eval_real, eval_gen, groups_real, groups_gen)

    # TODO: do second grouping for each of the first groups! (score_cond_kwargs)

    if return_plot_fn:
        # TODO: conditional plots
        # plot_fn = lambda title: plotting.hist(
        #     scores_real,
        #     scores_gen,
        #     bins=np.unique(thresholds),
        #     title=title,
        # )
        plot_fn = lambda: None
        return score_df, plot_fn
    else:
        return score_df


def calc_metric(
        loader: data_loading.Simple_Loader,
        scoring_fn: Callable[[pd.DataFrame, pd.DataFrame], float],
        metric_fn: Callable[[pd.DataFrame], float] | \
                   Iterable[Callable[[pd.DataFrame], float]],
        scoring_fn_cond: Optional[
                         Callable[[pd.DataFrame, pd.DataFrame], float]
                        ] = None,
        score_kwargs: dict = {},
        score_cond_kwargs: dict = {},
    ) -> tuple[float, pd.DataFrame, Callable]:
    """
    """
    # unconditional scoring
    if scoring_fn_cond is None:
        score_df, plot_fn = score_data(
            loader, scoring_fn, return_plot_fn=True, **score_kwargs)
    # conditional scoring
    else:
        score_df, plot_fn = score_data_cond(
            loader, scoring_fn, scoring_fn_cond,
            return_plot_fn=True, score_kwargs=score_kwargs,
            score_cond_kwargs=score_cond_kwargs
        )

    # TODO: integrate bootstrapping into metric fn
    #       either by making all metric fns jax compatible and vmapping
    #       ... or by adding a bootstrapping layer on top of the metric
    #       --> for loop over bootstrap samples
    if hasattr(metric_fn, "__iter__"):
        metric = [m(score_df) for m in metric_fn]
    else:
        metric = metric_fn(score_df)
    return metric, score_df, plot_fn

def run_benchmark(
        loader: data_loading.Simple_Loader,
        scoring_config_dict: dict,
        default_metric: Callable[[pd.DataFrame], float],
    ) -> tuple[
        dict[str, float],
        dict[str, pd.DataFrame],
        dict[str, Callable]
    ]:
    """
    """

    scores = {}
    score_dfs = {}
    plot_fns = {}

    for score_name, score_config in scoring_config_dict.items():

        if score_config.get("eval", None) is not None:
            score_cond_config = score_config["cond"]
            score_config = score_config["eval"]
            score_cond_kwargs = get_kwargs(score_cond_config)
            score_fn_cond = score_cond_config["fn"]
        else:
            score_cond_kwargs = {}
            score_fn_cond = None

        score_kwargs = get_kwargs(score_config)

        # print('score_cond_kwargs:', score_cond_kwargs)
        metric_fns = score_config.get("metric_fns", default_metric)

        scores[score_name], score_dfs[score_name], plot_fns[score_name] = \
            calc_metric(
                loader,
                score_config["fn"],
                metric_fns,
                score_fn_cond,
                score_kwargs=score_kwargs,
                score_cond_kwargs=score_cond_kwargs,
            )
        
    return scores, score_dfs, plot_fns