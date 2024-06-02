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
        "group_scores": score_config.get("group_scores", True),
    }

    # default to quartiles if no quantiles, n_bins or thresholds are specified
    if (kwargs["quantiles"] is None) \
        and (kwargs["n_bins"] is None) \
        and (kwargs["thresholds"] is None):
            # kwargs["quantiles"] = [0.25, 0.5, 0.75]
            # fd: Freedman Diaconis Estimator
            # see: https://numpy.org/doc/stable/reference/generated/numpy.histogram_bin_edges.html#numpy.histogram_bin_edges
            kwargs["bin_method"] = 'fd'

    return kwargs


def score_data(
        loader: data_loading.Simple_Loader,
        scoring_fn: Callable[[pd.DataFrame, pd.DataFrame], float],
        *,
        group_scores: bool = True,
        return_plot_fn=False,
        **kwargs,
    ):
    """
    """
    scores_real, scores_gen = partitioning.score_real_gen(loader, scoring_fn)

    if group_scores:
        groups_real, groups_gen, thresholds = partitioning.group_by_score(
            scores_real, scores_gen,
            return_thresholds=True,
            **kwargs
        )
        thresholds = np.unique(thresholds)
    else:
        groups_real, groups_gen, thresholds = None, None, 'auto'

    score_df = partitioning.get_score_table(scores_real, scores_gen, groups_real, groups_gen)

    if return_plot_fn:
        plot_fn = lambda title, ax: plotting.hist(
            scores_real,
            scores_gen,
            bins=thresholds,
            title=title,
            xlabel=title,
            ax=ax,
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
    # print('score_real', scores_real)
    # print('score_gen', scores_gen)
    # print('thresholds', thresholds)

    # calc scores of interest (to be conditioned on prior scores)
    eval_real, eval_gen = partitioning.score_real_gen(loader, scoring_fn)
    score_df = partitioning.get_score_table(eval_real, eval_gen, groups_real, groups_gen)

    # TODO: do optional second grouping for each of the first groups! (score_cond_kwargs)

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


def score_prediction_horizons(
        loader: data_loading.Simple_Loader,
        scoring_fn: Callable[[pd.DataFrame, pd.DataFrame], float],
        horizon_length: int,
        group_scores: bool = True,
        **kwargs
    ):
    """
    Computes a table unconditional scores of all real data with 
    """
    # unconditional scoring of all real data
    scores_real = partitioning.score_real(loader, scoring_fn)
    # score_df_real = partitioning.get_score_table(scores_real, None, None, None)

    # split generated data into subsequences based on generation horizon
    subseqs = tuple(partitioning.get_subseqs(s, subseq_len=horizon_length) for s in loader)
    # score the generated subsequences
    scores_gen = partitioning.score_gen(subseqs, scoring_fn)
    # enumerate the subsequences (===prediction horizon)
    _, groups_gen = partitioning.group_by_subseq(subseqs)
    score_df_gen = partitioning.get_score_table(
        scores_real=None,
        scores_gen=scores_gen,
        groups_real=None,
        groups_gen=groups_gen,
    )
    # print('score_df_gen')
    # display(score_df_gen)

    score_dfs = []
    for horizon_group, subtable in score_df_gen.groupby('group'):
        # print(horizon_group)
        # display(subtable)
        # compute score groups for each prediction horizon for generated data
        scores_gen = subtable.score.values
        # print('scores_real:', scores_real)

        if group_scores:
            groups_real, groups_gen, thresholds = partitioning.group_by_score(
                scores_real, (scores_gen,),
                return_thresholds=True, **kwargs)
        else:
            groups_real, groups_gen = None, None

        # print('groups_gen', groups_gen)
        score_df = partitioning.get_score_table(
            scores_real, (scores_gen,), groups_real, groups_gen)
        score_dfs.append(score_df)

    return score_dfs


def compute_metrics(
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

    if hasattr(metric_fn, "__iter__"):
        metric = [m(score_df) for m in metric_fn]
    else:
        metric = metric_fn(score_df)
    return metric, score_df, plot_fn


def compute_divergence_metrics(
        loader: data_loading.Simple_Loader,
        scoring_fn: Callable[[pd.DataFrame, pd.DataFrame], float],
        metric_fn: Callable[[pd.DataFrame], float] | \
                   Iterable[Callable[[pd.DataFrame], float]],
        horizon_length: int,
        **kwargs,
    ):
    """
    """
    score_dfs_horizon = score_prediction_horizons(
        loader,
        scoring_fn,
        horizon_length,
        # quantiles=[0.25, 0.5, 0.75],
        **kwargs,
    )
    # L1 scores for each horizon distribution
    loss_horizons = [metric_fn(sdf) for sdf in score_dfs_horizon]
    plot_fn = lambda title, ax: plotting.error_divergence_plot(
        loss_horizons,
        horizon_length, 
        title=title, 
        xlabel='Prediction Horizon [messages]',
        ylabel='L1 score',
        ax=ax,
    )
    return loss_horizons, score_dfs_horizon, plot_fn


def run_benchmark(
        loader: data_loading.Simple_Loader,
        scoring_config_dict: dict,
        default_metric: Callable[[pd.DataFrame], float],
        divergence: bool = False,
        divergence_horizon: int = 50,
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

        if divergence:
            scores[score_name], score_dfs[score_name], plot_fns[score_name] = \
                compute_divergence_metrics(
                    loader,
                    score_config["fn"],
                    metric_fns,
                    divergence_horizon,
                    **get_kwargs(score_config)
                )

        else:
            scores[score_name], score_dfs[score_name], plot_fns[score_name] = \
                compute_metrics(
                    loader,
                    score_config["fn"],
                    metric_fns,
                    score_fn_cond,
                    score_kwargs=score_kwargs,
                    score_cond_kwargs=score_cond_kwargs,
                )
        
    return scores, score_dfs, plot_fns

def summary_stats(scores, bootstrap=True, ci_alpha=0.05, n_bootstrap=100):
    loss_vals = np.array([s[0] for s in scores.values()])
    aggr_mean, aggr_median, aggr_iqm = _calc_summary_stats(loss_vals)
    # print(aggr_mean, aggr_median, aggr_iqm)
    
    if bootstrap:
        losses_bootstrap = np.array(
            [[np.random.choice(s[2]) for s in scores.values()] for _ in range(n_bootstrap)])
        bs_mean, bs_median, bs_iqm = _calc_summary_stats(losses_bootstrap)
        q = np.array([ci_alpha/2 * 100, 100 - ci_alpha/2*100])
        ci_mean = np.percentile(bs_mean, q)
        ci_median = np.percentile(bs_median, q)
        ci_iqm = np.percentile(bs_iqm, q)
        return (aggr_mean, ci_mean), (aggr_median, ci_median), (aggr_iqm, ci_iqm)
    return aggr_mean, aggr_median, aggr_iqm


def _calc_summary_stats(loss_vals):
    aggr_mean = np.mean(loss_vals, axis=-1)
    aggr_median = np.median(loss_vals, axis=-1)

    q25, q75 = np.percentile(loss_vals, [25, 75], axis=-1, keepdims=True)
    # iqr = q75 - q25
    aggr_iqm = np.ma.masked_array(
        loss_vals,
        mask=(loss_vals < q25) | (loss_vals > q75)
    ).mean(axis=-1).__array__()

    return aggr_mean, aggr_median, aggr_iqm