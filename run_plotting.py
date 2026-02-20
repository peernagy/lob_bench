import os
os.environ["PYPLOTLY_BROWSER"] = "none"

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import argparse
from datetime import datetime
import pathlib
import scoring
import metrics
import plotting
from run_bench import load_results

###
import traceback
import warnings
import sys


def warn_with_traceback(message, category, filename, lineno, file=None, line=None):

    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))

warnings.showwarning = warn_with_traceback


def _parse_score_filename(filename: str):
    stem = filename.rsplit("/", 1)[-1].replace('.pkl', '')
    parts = stem.split('_')
    timestamp = f"{parts[-2]}_{parts[-1]}" if len(parts) >= 2 else "unknown"
    core = parts[1:-2]  # drop leading "scores" and trailing timestamp
    if not core:
        return "unknown", "unknown", "unknown", timestamp

    if len(core) >= 2 and core[0] == "time" and core[1] == "lagged":
        score_type = "time_lagged"
        idx = 2
    else:
        score_type = core[0]
        idx = 1

    stock = core[idx] if len(core) > idx else "unknown"
    model_parts = core[idx + 1:] if len(core) > idx + 1 else []
    model = "_".join(model_parts) if model_parts else "unknown"
    if model and model[0].isdigit():
        model = model[1:]

    return score_type, stock, model, timestamp


def _load_all_scores(files):
    all_scores = {}
    all_dfs = {}
    all_timestamps = {}
    for f in files:
        score_type, stock, model, timestamp = _parse_score_filename(f)
        
        print(f"  Loading: {f.rsplit('/', 1)[-1]}")
        print(
            "    Parsed as - Type: {score_type}, Stock: {stock}, Model: {model}, "
            "Timestamp: {timestamp}".format(
                score_type=score_type,
                stock=stock,
                model=model,
                timestamp=timestamp,
            )
        )
        
        scores, scores_dfs = load_results(f)
        print(f"    Loaded {len(scores)} score metrics")
        
        if stock not in all_scores:
            all_scores[stock] = {}
            all_dfs[stock] = {}
            all_timestamps[stock] = {}
        
        # Store with timestamp to handle multiple runs
        # Use latest timestamp for each stock/model combination
        if model not in all_scores[stock] or timestamp > all_timestamps[stock].get(model, ""):
            all_scores[stock][model] = scores
            all_dfs[stock][model] = scores_dfs
            all_timestamps[stock][model] = timestamp
    
    return all_scores, all_dfs, all_timestamps


def _scores_to_df(scores):
    rows = []
    for stock, model_scores in scores.items():
        for model, model_score in model_scores.items():
            for score_name, metric_score in model_score.items():
                # standard format for cond / uncond scores
                if isinstance(metric_score, dict):
                    col_names = ['stock', 'model', 'metric', 'score', 'mean', 'ci_low', 'ci_high']
                    for metric_name, metric_val in metric_score.items():
                        mean, ci, bootstr_vals = metric_val
                        rows.append((stock, model, metric_name, score_name, mean, ci[0], ci[1]))
                # expect iterable as used by divergence scores
                elif isinstance(metric_score, list):
                    col_names = ['stock', 'model', 'score', 'interval', 'mean', 'ci_low', 'ci_high']
                    for i, (mean, ci, bootstr_vals) in enumerate(metric_score):
                        rows.append((stock, model, score_name, i, mean, ci[0], ci[1]))
                else:
                    col_names = ['stock', 'model', 'score', 'mean', 'ci_low', 'ci_high']
                    mean, ci, bootstr_vals = metric_score
                    rows.append((stock, model, score_name, mean, ci[0], ci[1]))

    return pd.DataFrame(rows, columns=col_names)


def _tag_scores(scores: dict, tag: str) -> dict:
    return {f"{tag}:{name}": val for name, val in scores.items()}


def _aggregate_divergence_scores(score_list: list, ci_alpha: float = 0.01) -> dict:
    bootstraps = []
    for score_val in score_list:
        if not score_val or len(score_val) < 3:
            continue
        boot = np.asarray(score_val[2]).ravel()
        if boot.size:
            bootstraps.append(boot)
    if not bootstraps:
        return {}
    flat = np.concatenate(bootstraps)
    q = np.array([ci_alpha / 2 * 100, 100 - ci_alpha / 2 * 100])
    ci = np.percentile(flat, q)
    return {"divergence": (np.nanmean(flat), ci, flat)}


def _summary_stats_flexible(
    scores,
    bootstrap: bool = True,
    ci_alpha: float = 0.01,
    n_bootstrap: int = 1000,
    rng_np: np.random.Generator = np.random.default_rng(12345),
):
    return_dict = {}
    metric_names = sorted({m for s in scores.values() for m in s.keys()})
    for metric_name in metric_names:
        metric_scores = [s[metric_name] for s in scores.values() if metric_name in s]
        if not metric_scores:
            continue

        point_vals = np.asarray([s[0] for s in metric_scores], dtype=float)
        point_vals = point_vals[np.isfinite(point_vals)]
        n_points = point_vals.size
        if n_points == 0:
            continue

        q25, q75 = np.percentile(point_vals, [25, 75])
        iqm_vals = point_vals[(point_vals >= q25) & (point_vals <= q75)]
        aggr_mean = float(np.mean(point_vals))
        aggr_median = float(np.median(point_vals))
        aggr_iqm = float(np.mean(iqm_vals)) if iqm_vals.size else float(np.mean(point_vals))

        if bootstrap:
            losses_bootstrap = np.array([
                rng_np.choice(point_vals, size=n_points, replace=True)
                for _ in range(n_bootstrap)
            ])
            bs_mean, bs_median, bs_iqm = scoring._calc_summary_stats(losses_bootstrap)
            q = np.array([ci_alpha / 2 * 100, 100 - ci_alpha / 2 * 100])
            ci_mean = np.percentile(bs_mean, q)
            ci_median = np.percentile(bs_median, q)
            ci_iqm = np.percentile(bs_iqm, q)
            return_dict[metric_name] = (
                (aggr_mean, ci_mean),
                (aggr_median, ci_median),
                (aggr_iqm, ci_iqm),
            )
        else:
            return_dict[metric_name] = aggr_mean, aggr_median, aggr_iqm

    return return_dict


def run_plotting(
    args,
    score_dir: str,
    plot_dir: str,
    model_name: str,
) -> None:
    # load all saved stats
    print("[*] Loading data...")
    uncond_files = sorted(glob(score_dir + "/scores_uncond_*.pkl"))
    cond_files = sorted(glob(score_dir + "/scores_cond_*.pkl"))
    context_files = sorted(glob(score_dir + "/scores_context_*.pkl"))
    time_lagged_files = sorted(glob(score_dir + "/scores_time_lagged_*.pkl"))
    div_files = sorted(glob(score_dir + "/scores_div_*.pkl"))
    if len(div_files) > 0:
        div_horizon_length = int(div_files[0].split("_")[-3])

    # load all scores
    all_scores_uncond = {}
    all_scores_cond = {}
    all_scores_context = {}
    all_scores_time_lagged = {}
    all_scores_div = {}
    all_dfs_uncond = {}
    all_dfs_cond = {}
    all_dfs_context = {}
    all_dfs_time_lagged = {}
    all_dfs_div = {}
    all_timestamps_uncond = {}
    all_timestamps_time_lagged = {}
    
    if len(uncond_files) > 0:
        all_scores_uncond, all_dfs_uncond, all_timestamps_uncond = _load_all_scores(uncond_files)
    if len(cond_files) > 0:
        all_scores_cond, all_dfs_cond, _ = _load_all_scores(cond_files)
    if len(context_files) > 0:
        all_scores_context, all_dfs_context, _ = _load_all_scores(context_files)
    if len(time_lagged_files) > 0:
        all_scores_time_lagged, all_dfs_time_lagged, all_timestamps_time_lagged = _load_all_scores(time_lagged_files)
    if len(div_files) > 0:
        all_scores_div, all_dfs_div, _ = _load_all_scores(div_files)

    # Combined summary plots across all score types
    combined_scores = {}
    for stock, score_stock in all_scores_uncond.items():
        for model, score_model in score_stock.items():
            combined_scores.setdefault(stock, {}).setdefault(model, {}).update(
                _tag_scores(score_model, "uncond")
            )
    for stock, score_stock in all_scores_cond.items():
        for model, score_model in score_stock.items():
            combined_scores.setdefault(stock, {}).setdefault(model, {}).update(
                _tag_scores(score_model, "cond")
            )
    for stock, score_stock in all_scores_time_lagged.items():
        for model, score_model in score_stock.items():
            combined_scores.setdefault(stock, {}).setdefault(model, {}).update(
                _tag_scores(score_model, "time_lagged")
            )
    for stock, score_stock in all_scores_context.items():
        for model, score_model in score_stock.items():
            combined_scores.setdefault(stock, {}).setdefault(model, {}).update(
                _tag_scores(score_model, "context")
            )
    for stock, score_stock in all_scores_div.items():
        for model, score_model in score_stock.items():
            if model.upper() == "REAL":
                continue
            div_scores = {}
            for score_name, score_list in score_model.items():
                aggregated = _aggregate_divergence_scores(score_list)
                if aggregated:
                    div_scores[f"div:{score_name}"] = aggregated
            combined_scores.setdefault(stock, {}).setdefault(model, {}).update(div_scores)

    if combined_scores:
        print("[*] Plotting combined summary stats")
        summary_stats_all = {
            stock: {
                model: _summary_stats_flexible(scores, bootstrap=True)
                for model, scores in combined_scores[stock].items()
            } for stock in combined_scores
        }
        print(summary_stats_all)
        plotting.summary_plot(
            summary_stats_all,
            save_path=f"{plot_dir}/summary_stats_all.png"
        )

        if args.summary_only:
            print("[*] Summary-only mode enabled; skipping remaining plots")
            return

        # COMPARISON PLOTS: bar plots and spider plots
        timestamps_uncond = [max(ts.values()) for ts in all_timestamps_uncond.values() if ts]
        if not timestamps_uncond or not all_scores_uncond:
            print("[*] No unconditional timestamps found; skipping comparison plots")
        else:
            print("[*] Plotting comparison plots")
            # Create comparison subdirectory with timestamp
            latest_timestamp = max(timestamps_uncond)
            comparison_dir = f"{plot_dir}/comparison/{latest_timestamp}"
            pathlib.Path(comparison_dir).mkdir(parents=True, exist_ok=True)
            
            # Bar plot of unconditional scores comparing all models
            data = _scores_to_df(all_scores_uncond)
            for stock in data.stock.unique():
                for metric in data.metric.unique():
                    print(f"[*] Plotting {stock} {metric} bar plots")
                    plotting.loss_bars(
                        data,
                        stock,
                        metric,
                        save_path=f"{comparison_dir}/bar_{stock}_{metric}.png"
                    )
                    print(f"[*] Plotting {stock} {metric} spider plots")
                    plotting.spider_plot(
                        all_scores_uncond[stock],
                        metric,
                        title=f"{metric.capitalize()} Loss ({stock})",
                        plot_cis=False,
                        save_path=f"{comparison_dir}/spider_{stock}_{metric}.png"
                    )

    if len(time_lagged_files) > 0:
        # COMPARISON PLOTS: bar plots and spider plots for time-lagged
        print("[*] Plotting time-lagged comparison plots")
        # Create comparison subdirectory with timestamp
        latest_timestamp_tl = max([max(ts.values()) if ts else "unknown" for ts in all_timestamps_time_lagged.values()])
        comparison_dir_tl = f"{plot_dir}/comparison/{latest_timestamp_tl}"
        pathlib.Path(comparison_dir_tl).mkdir(parents=True, exist_ok=True)
        
        # Filter out empty scores before converting to dataframe
        all_scores_time_lagged_filtered = {}
        for stock, models in all_scores_time_lagged.items():
            all_scores_time_lagged_filtered[stock] = {m: s for m, s in models.items() if s}
        
        if all_scores_time_lagged_filtered and any(all_scores_time_lagged_filtered.values()):
            data_time_lagged = _scores_to_df(all_scores_time_lagged_filtered)
        else:
            data_time_lagged = pd.DataFrame()  # Empty dataframe
        
        if not data_time_lagged.empty:
            for stock in data_time_lagged.stock.unique():
                for metric in data_time_lagged.metric.unique():
                    print(f"[*] Plotting {stock} {metric} time-lagged bar plots")
                    plotting.loss_bars(
                        data_time_lagged,
                        stock,
                        metric,
                        save_path=f"{comparison_dir_tl}/bar_time_lagged_{stock}_{metric}.png"
                    )
                    print(f"[*] Plotting {stock} {metric} time-lagged spider plots")
                    plotting.spider_plot(
                        all_scores_time_lagged_filtered[stock],
                        metric,
                        title=f"{metric.capitalize()} Loss - Time-Lagged ({stock})",
                        plot_cis=False,
                        save_path=f"{comparison_dir_tl}/spider_time_lagged_{stock}_{metric}.png"
                    )

    if len(div_files) > 0:
        # divergence plots
        print("[*] Plotting divergence plots")
        for stock, score_stock in tqdm(all_scores_div.items(), position=0, desc="Stock"):

            # baseline errors for each score by bootstrapping
            # loss for two real data samples and plot the 99% CI as a lower bound
            baseline_errors_by_score = all_scores_div[stock].pop("REAL", None)
            if baseline_errors_by_score is not None:
                baseline_errors_by_score = {
                    k: np.array([e[1][1] for e in v])
                    for k, v in baseline_errors_by_score.items()
                }

            # new plot for each stock but layer all models on top
            axs = None
            for i_model, (model, score_model) in tqdm(
                enumerate(score_stock.items()),
                position=1, desc="Model", leave=False
            ):
                # only plot baseline errors once
                if i_model > 0 or baseline_errors_by_score is None:
                    baseline_errors_by_score = {score_name: None for score_name in score_model.keys()}

                plot_fns_uncond = {
                    score_name: plotting.get_div_plot_fn(
                        score_,
                        horizon_length=div_horizon_length,
                        color=f"C{i_model}",
                        model_name=model,
                        baseline_errors=baseline_errors_by_score.get(score_name, None)
                    )
                    for score_name, score_ in score_model.items()
                        # skip OFI scores (averaged over 100 messages)
                        if not score_name.startswith("ofi")
                }
                # only save once when the last model is plotted
                if i_model == len(score_stock) - 1:
                    save_path = f'{plot_dir}/divergence_{stock}.png'
                else:
                    save_path = None
                axs = plotting.hist_subplots(
                    plot_fns_uncond,
                    axs=axs,
                    figsize=(10, 22),
                    suptile=f"L1 Divergence {stock} {model}",
                    save_path=save_path
                )

            plt.close()

    if len(uncond_files) and args.histograms > 0:
        # UNCONDITIONAL score histograms
        print("[*] Plotting unconditional histograms")
        for stock, score_stock in tqdm(all_dfs_uncond.items(), position=0, desc="Stock"):
            for model, score_model in tqdm(score_stock.items(), position=1, desc="Model", leave=False):
                # unconditional scores
                plot_fns_uncond = {
                    score_name: plotting.get_plot_fn_uncond(score_df)
                        for score_name, score_df in score_model.items()
                }
                print(f"[*] Obtained plot functions for {stock} {model} unconditional histograms")
                plotting.hist_subplots(
                    plot_fns_uncond,
                    figsize=(10, 22),
                    suptile=f"{stock} {model}",
                    save_path=f"{plot_dir}/hist_{stock}_{model}.png",
                    plot_legend=False,
                )
                plt.close()
    
    if len(cond_files) > 0 and args.histograms:
        # CONDITIONAL score histograms
        print("[*] Plotting conditional histograms")
        for stock, score_stock in tqdm(all_dfs_cond.items(), position=0, desc="Stock"):
            for model, score_model in tqdm(score_stock.items(), position=1, desc="Model", leave=False):
                for score_name, score_df in score_model.items():
                    var_eval, var_cond = score_name.split(" | ", 1)
                    print(f"[*] Plotting {stock} {model} cond histograms for {var_eval} | {var_cond}")
                    binwidth = 100 if var_eval == "spread" else None
                    plotting.facet_grid_hist(
                        score_df,
                        var_eval=var_eval,
                        var_cond=var_cond,
                        filter_groups_below_weight=0.01,
                        bins='auto',
                        binwidth=binwidth,
                        stock=stock,
                        model=model,
                    )
                    plt.savefig(
                        f"{plot_dir}/hist_cond_{stock}_{model}_{var_eval}_{var_cond}.png",
                        bbox_inches="tight", dpi=300
                    )
                    plt.close()

    if len(time_lagged_files) > 0 and args.histograms > 0:
        # TIME-LAGGED score histograms
        print("[*] Plotting time-lagged histograms")
        for stock, score_stock in tqdm(all_dfs_time_lagged.items(), position=0, desc="Stock"):
            for model, score_model in tqdm(score_stock.items(), position=1, desc="Model", leave=False):
                # Get timestamp for this stock/model
                timestamp = all_timestamps_time_lagged.get(stock, {}).get(model, "unknown")
                
                for score_name, score_df in score_model.items():
                    # Backward compatibility: rename score_lagged to score_cond if present
                    if 'score_lagged' in score_df.columns and 'score_cond' not in score_df.columns:
                        score_df = score_df.rename(columns={'score_lagged': 'score_cond'})
                    
                    # Parse score name: "ask_volume | spread (t-lag=500)" -> var_eval, var_cond
                    if " | " in score_name:
                        var_eval, var_cond_full = score_name.split(" | ", 1)
                        # Remove lag info from var_cond for cleaner display
                        var_cond = var_cond_full.split(" (")[0] if " (" in var_cond_full else var_cond_full
                    else:
                        var_eval = score_name
                        var_cond = "lagged"
                    
                    print(f"[*] Plotting {stock} {model} time-lagged histograms for {var_eval} | {var_cond}")
                    binwidth = 100 if var_eval == "spread" else None
                    plotting.facet_grid_hist(
                        score_df,
                        var_eval=var_eval,
                        var_cond=var_cond,
                        filter_groups_below_weight=0.01,
                        bins='auto',
                        binwidth=binwidth,
                        stock=stock,
                        model=model,
                    )
                    plt.savefig(
                        f"{plot_dir}/hist_time_lagged_{stock}_{model}_{var_eval}_{var_cond}_{timestamp}.png",
                        bbox_inches="tight", dpi=300
                    )
                    plt.close()

    if len(context_files) > 0:
        # CONTEXTUAL score histograms and regime analysis
        print("[*] Plotting contextual regime histograms")
        for stock, score_stock in tqdm(all_dfs_context.items(), position=0, desc="Stock"):
            for model, score_model in tqdm(score_stock.items(), position=1, desc="Model", leave=False):
                for score_name, score_df in score_model.items():
                    # Get timestamp from model files to use in filename
                    context_files_stock = [f for f in context_files if f"_{stock}_" in f]
                    timestamp = "unknown"
                    if context_files_stock:
                        timestamp = context_files_stock[0].split("_")[-1].replace(".pkl", "")
                    
                    # Use facet_grid_hist similar to conditional/time-lagged scoring
                    binwidth = 100 if "spread" in score_name else None
                    plotting.facet_grid_hist(
                        score_df,
                        var_eval=score_name,
                        var_cond="regime",
                        filter_groups_below_weight=0.01,
                        bins='auto',
                        binwidth=binwidth,
                        stock=stock,
                        model=model,
                    )
                    plt.savefig(
                        f"{plot_dir}/hist_context_{stock}_{model}_{score_name}_{timestamp}.png",
                        bbox_inches="tight", dpi=300
                    )
                    plt.close()

    print("[*] Done")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--score_dir", type=str, default="./results/scores")
    parser.add_argument("--plot_dir", default="./results/plots", type=str)
    parser.add_argument("--show_plots", action="store_true")
    parser.add_argument("--model_name",default="large_model_sample",type=str)
    parser.add_argument("--summary_only", action="store_true", default=False,
                        help="Only generate summary plots and skip other plots")
    parser.add_argument("--histograms", action="store_true", default=False,
                        help="Plot histograms of scores")
    args = parser.parse_args()

    run_plotting(args,args.score_dir, args.plot_dir,args.model_name)
    if args.show_plots:
        plt.show()
