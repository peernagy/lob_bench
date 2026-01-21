import numpy as np
import pandas as pd
from typing import Callable, Iterable, List, Optional,Union
from tqdm import tqdm
import plotting
import partitioning
import data_loading


def get_kwargs(score_config, conditional=False):
    kwargs = {
        "discrete": score_config.get("discrete", False),
        "bin_method": score_config.get("bin_method", None),
        "quantiles": score_config.get("quantiles", None),
        "n_bins": score_config.get("n_bins", None),
        "thresholds": score_config.get("thresholds", None),
    }
    if not conditional:
        kwargs["group_scores"] = score_config.get("group_scores", True)

    # default to quartiles if no quantiles, n_bins or thresholds are specified
    if kwargs["bin_method"] is None \
        and (kwargs["quantiles"] is None) \
        and (kwargs["n_bins"] is None) \
        and (kwargs["thresholds"] is None):
            if conditional:
                # kwargs["quantiles"] = [0.25, 0.5, 0.75]
                # use deciles to condition on by default
                kwargs["quantiles"] = np.arange(0.1, 1.0, 0.1)
            else:
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
    Grouping is done based on the scoring_fn_cond
    and scores are based on the scoring_fn.
    Conditional analysis can then be done using the returned score_df
    by grouping on 'group' and calculating metrics on 'score'.
    The additional colum 'subgroup' gives the bin of the conditional scores,
    where the binning is done based on the scoring_fn and score_kwargs.
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

    score_df_cond = partitioning.get_score_table(scores_real, scores_gen, groups_real, groups_gen)

    # second grouping for each of the first groups (binning the conditional scores):
    score_df['subgroup'] = -1
    score_df['score_cond'] = score_df_cond.score
    sub_dfs = [df[1] for df in score_df.groupby('group')]
    new_dfs = []
    for df in sub_dfs:
        real_scores = df.loc[df.type == 'real', 'score']
        gen_scores = df.loc[df.type == 'generated', 'score']
        groups_real, groups_gen = partitioning.group_by_score(
            real_scores.values,
            gen_scores.values,
            **score_kwargs
        )
        df = df.copy()
        # df['subgroup'] = -1
        df.loc[real_scores.index, 'subgroup'] = groups_real
        df.loc[gen_scores.index, 'subgroup'] = groups_gen
        new_dfs.append(df)
    score_df = pd.concat(new_dfs)

    if return_plot_fn:
        plot_fn = lambda var_eval, var_cond, bins, binwidth: plotting.facet_grid_hist(
            score_df,
            var_eval=var_eval,
            var_cond=var_cond,
            filter_groups_below_weight=0.01,
            bins=bins,
            binwidth=binwidth,
        )
        return score_df, plot_fn
    else:
        return score_df


def score_data_context(
    loader: data_loading.Simple_Loader,
    scoring_fn: Callable[[pd.DataFrame, pd.DataFrame], float],
    scoring_fn_context: Callable[[pd.DataFrame, pd.DataFrame], float],
    *,
    score_context_kwargs: dict = {},
):
    """
    Context-aware scoring that evaluates performance within market regimes.
    
    Regimes are defined from CONDITIONAL data using scoring_fn_context.
    The evaluation metric is computed from real/generated data using scoring_fn,
    and compared within each regime.
    
    Returns:
        score_df: DataFrame with columns [score, group, type]
        where 'group' is the regime ID from conditional data
    """
    # Step 1: Define regimes from CONDITIONAL data
    scores_cond = partitioning.score_cond(loader, scoring_fn_context)
    _, _, thresholds = partitioning.group_by_score(
        scores_cond, scores_cond,
        return_thresholds=True,
        **score_context_kwargs
    )
    
    # Step 2: Assign real/gen data to the same regimes using thresholds from conditional
    scores_real, scores_gen = partitioning.score_real_gen(loader, scoring_fn_context)
    groups_real, groups_gen = partitioning.group_by_score(
        scores_real, scores_gen,
        thresholds=thresholds
    )[:2]
    
    # Step 3: Evaluate metric of interest within each regime
    eval_real, eval_gen = partitioning.score_real_gen(loader, scoring_fn)
    score_df = partitioning.get_score_table(eval_real, eval_gen, groups_real, groups_gen)
    
    return score_df


def compute_metrics_context(
    loader: data_loading.Simple_Loader,
    scoring_fn: Callable[[pd.DataFrame, pd.DataFrame], float],
    metric_fn: dict[str, Callable[[pd.DataFrame], float]],
    scoring_fn_context: Callable[[pd.DataFrame, pd.DataFrame], float],
    score_context_kwargs: dict = {},
    ci_alpha: float = 0.01,
) -> tuple[dict, pd.DataFrame]:
    """
    Compute metrics separately for each contextual regime (e.g., spread regime).
    Returns per-regime metrics without aggregation, exposing performance degradation
    in specific market conditions.
    
    Returns:
        metric: dict[metric_name: dict[regime_id: (point_est, ci, bootstrapped_losses)]]
        score_df: DataFrame with columns [score, group, type]
    """
    score_df = score_data_context(
        loader, scoring_fn, scoring_fn_context,
        score_context_kwargs=score_context_kwargs
    )
    
    # Calculate metrics separately for each regime
    metric = {}
    regimes = sorted(score_df['group'].unique())
    
    for m_name, mfn in metric_fn.items():
        metric[m_name] = {}
        
        for regime_id in regimes:
            regime_df = score_df[score_df['group'] == regime_id].copy()
            
            if len(regime_df) == 0:
                continue
            
            # Calculate metric for this regime
            result = mfn(regime_df)
            
            # Handle both conditional (tuple) and unconditional (scalar) metric functions
            if isinstance(result, tuple) and len(result) == 3:
                point_est, _, bootstrapped = result
            else:
                # For unconditional metrics that return scalar, wrap in expected format
                point_est = result
                bootstrapped = np.array([result])
            
            # Calculate confidence intervals for this regime
            q = np.array([ci_alpha/2 * 100, 100 - ci_alpha/2*100])
            ci = np.percentile(bootstrapped, q)
            
            metric[m_name][regime_id] = (point_est, ci, bootstrapped)
    
    return metric, score_df


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
    # metric_fn: Callable[[pd.DataFrame], float] | \
    #            Iterable[Callable[[pd.DataFrame], float]],
    metric_fn: dict[str, Callable[[pd.DataFrame], float]],
    scoring_fn_cond: Optional[
                        Callable[[pd.DataFrame, pd.DataFrame], float]
                    ] = None,
    score_kwargs: dict = {},
    score_cond_kwargs: dict = {},
) -> tuple[float, pd.DataFrame, Callable]:
    # unconditional scoring
    if scoring_fn_cond is None:
        score_df, plot_fn = score_data(
            loader, scoring_fn, return_plot_fn=True, **score_kwargs)
        metric = {m_name: m(score_df) for m_name, m in metric_fn.items()}

    # conditional scoring
    else:
        score_df, plot_fn = score_data_cond(
            loader, scoring_fn, scoring_fn_cond,
            return_plot_fn=True, score_kwargs=score_kwargs,
            score_cond_kwargs=score_cond_kwargs
        )

        # calc. loss for each of the conditional distributions
        lens = []
        losses = []
        # len_and_losses = []
        for name, group in score_df.groupby('group'):
            # len_and_losses.append([len(group), metric_fn(group)[0]])
            lens.append(len(group))
            # use the subgroup for binning now:
            group.group = group.subgroup
            losses.append(
                np.stack(
                    tuple(mfn(group)[2] for mfn in metric_fn.values()),
                    axis=-1
                )
            )

        # calculate weights by normalizing the number of observations
        weights = np.array(lens, dtype=float)
        weights /= weights.sum()
        losses = np.array(losses).T
        # print('weights:', weights, 'losses:', losses.shape, losses)
        # sum over all groups
        # shape: (num metrics, n_bootstrap + 1, num groups)
        #     -> (num metrics, n_bootstrap + 1)
        metric = np.nansum(losses * weights, axis=-1)
        # print('metric:', metric)

        # get the percentiles of the bootstrapped loss values
        # TODO: make this an argument
        ci_alpha = 0.01
        q = np.array([ci_alpha/2 * 100, 100 - ci_alpha/2*100])
        # shape: (num metrics, n_bootstrap + 1)
        ci = np.nanpercentile(metric, q, axis=-1).T
        # print('ci', ci)
        # metric = (metric[:, 0], ci, metric)
        metric = {
            m_name: (m[0], ci_, m) for m_name, ci_, m
                in zip(metric_fn.keys(), ci, metric)
        }
        # print('metric:', metric)

    return metric, score_df, plot_fn


def _replace_gen_with_real_scores(horizon_df):
    df_real1 = horizon_df[horizon_df["type"] == 'real'].copy()
    df_real2 = df_real1.copy()
    # pretend real data is generated to calculate baseline divergence
    df_real2['type'] = 'generated'
    df_ = pd.concat([df_real1, df_real2])
    return df_


def calc_baseline_errors_by_score(
    scoring_dfs_horizon_all_scores: dict[str, list[pd.DataFrame]],
    metric_fn: Callable[[pd.DataFrame], float],
) -> dict[str, list[float]]:
    baseline_errors_by_score = {
        score_name: [
            # TODO: generalize this to other metrics
            metric_fn(
                _replace_gen_with_real_scores(df),
                # we want one-sided: 0.99
                # --> upper limit of the two-sided intervals for 0.98
                ci_alpha=0.98
            ) for df in scoring_dfs_horizon
        ] for score_name, scoring_dfs_horizon in scoring_dfs_horizon_all_scores.items()
    }
    return baseline_errors_by_score


def compute_divergence_metrics(
        loader: data_loading.Simple_Loader,
        scoring_fn: Callable[[pd.DataFrame, pd.DataFrame], float],
        metric_fn: Union[Callable[[pd.DataFrame], float] , \
                   Iterable[Callable[[pd.DataFrame], float]]],
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
    contextual: bool = False,
    divergence_horizon: int = 50,
) -> tuple[
    dict[str, float],
    dict[str, pd.DataFrame],
    dict[str, Callable]
]:

    scores = {}
    score_dfs = {}
    plot_fns = {}

    for score_name, score_config in scoring_config_dict.items():
        print("Calculating scores and metrics for: ", score_name, end="\r\n")

        # contextual scoring
        if contextual:
            score_context_fn = score_config.get("context_fn", None)
            score_context_config = score_config.get("context_config", {})
            score_context_kwargs = get_kwargs(score_context_config, conditional=True)
            metric_fns = score_config.get("metric_fns", default_metric)
            
            if score_context_fn is None:
                raise ValueError(f"contextual=True but 'context_fn' not found in config for {score_name}")
            
            scores[score_name], score_dfs[score_name] = \
                compute_metrics_context(
                    loader,
                    score_config["fn"],
                    metric_fns,
                    score_context_fn,
                    score_context_kwargs=score_context_kwargs,
                )

        # conditional scoring
        elif score_config.get("eval", None) is not None:
            score_cond_config = score_config["cond"]
            score_config = score_config["eval"]
            score_cond_kwargs = get_kwargs(score_cond_config, conditional=True)
            score_kwargs = get_kwargs(score_config, conditional=True)
            score_fn_cond = score_cond_config["fn"]
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

        # unconditional scoring
        else:
            score_kwargs = get_kwargs(score_config)
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
                        None,
                        score_kwargs=score_kwargs,
                        score_cond_kwargs={},
                    )

    return scores, score_dfs, plot_fns


def summary_stats(
        scores,
        bootstrap: bool = True,
        ci_alpha: float = 0.01,
        n_bootstrap: int = 1000,
        rng_np: np.random.Generator = np.random.default_rng(12345),
    ):
    return_dict = {}
    # for each metric:
    values_list = list(scores.values())
    for i, metric_name in enumerate(values_list[0].keys()):
        loss_vals = np.array([s[metric_name][0] for s in scores.values()])

        # loss_vals = np.array(
        #     [[score[2] for score in sdict.values()[0]] for sdict in scores.values()])
        aggr_mean, aggr_median, aggr_iqm = _calc_summary_stats(loss_vals)
        # print(aggr_mean, aggr_median, aggr_iqm)

        if bootstrap:
            # draw single score from each bootstrap sample per metric
            losses_bootstrap = np.array([
                [rng_np.choice(s[metric_name][2])
                    for s in scores.values()]
                for _ in range(n_bootstrap)
            ])
            # print(losses_bootstrap)
            bs_mean, bs_median, bs_iqm = _calc_summary_stats(losses_bootstrap)
            q = np.array([ci_alpha/2 * 100, 100 - ci_alpha/2*100])
            ci_mean = np.percentile(bs_mean, q)
            ci_median = np.percentile(bs_median, q)
            ci_iqm = np.percentile(bs_iqm, q)

            # return (aggr_mean, ci_mean), (aggr_median, ci_median), (aggr_iqm, ci_iqm)
            return_dict[metric_name] = (
                (aggr_mean, ci_mean),
                (aggr_median, ci_median),
                (aggr_iqm, ci_iqm),
            )
        # return aggr_mean, aggr_median, aggr_iqm
        else:
            return_dict[metric_name] = aggr_mean, aggr_median, aggr_iqm

    return return_dict

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
