from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import argparse
from datetime import datetime
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


def _load_all_scores(files):
    all_scores = {}
    all_dfs = {}
    for f in files:
        stock = f.rsplit("/", 1)[-1].split("_")[2]
        model = f.rsplit("/", 1)[-1].split("_")[3]
        # remove sorting number at the beginning of the model name
        # if present
        # TODO: generalize this for more then 10 models
        if model[0].isdigit():
            model = model[1:]
        scores, scores_dfs = load_results(f)
        if stock not in all_scores:
            all_scores[stock] = {}
        if stock not in all_dfs:
            all_dfs[stock] = {}
        all_scores[stock][model] = scores
        all_dfs[stock][model] = scores_dfs
    return all_scores, all_dfs


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
    all_scores_cond = {}
    if len(uncond_files) > 0:
        all_scores_uncond, all_dfs_uncond = _load_all_scores(uncond_files)
    if len(cond_files) > 0:
        all_scores_cond, all_dfs_cond = _load_all_scores(cond_files)
    if len(context_files) > 0:
        all_scores_context, all_dfs_context = _load_all_scores(context_files)
    if len(time_lagged_files) > 0:
        all_scores_time_lagged, all_dfs_time_lagged = _load_all_scores(time_lagged_files)
    if len(div_files) > 0:
        all_scores_div, all_dfs_div = _load_all_scores(div_files)

    if len(uncond_files) > 0:
        # SUMMARY PLOTS
        print("[*] Plotting summary stats")
        summary_stats = {
            stock: {
                model: scoring.summary_stats(
                    scores | all_scores_cond.get(stock, {}).get(model, {}),
                    bootstrap=True
                )
                for model, scores in all_scores_uncond[stock].items()
            } for stock in all_scores_uncond
        }
        print(summary_stats)

        plotting.summary_plot(
            summary_stats,
            save_path=f"{plot_dir}/summary_stats_comp.png"
        )

        # COMPARISON PLOTS: bar plots and spider plots
        print("[*] Plotting comparison plots")
        # Bar plot of unconditional scores comparing all models
        data = _scores_to_df(all_scores_uncond)
        for stock in data.stock.unique():
            for metric in data.metric.unique():
                print(f"[*] Plotting {stock} {metric} bar plots")
                plotting.loss_bars(
                    data,
                    stock,
                    metric,
                    save_path=f"{plot_dir}/bar_{stock}_{metric}.png"
                )
                print(f"[*] Plotting {stock} {metric} spider plots")
                plotting.spider_plot(
                    all_scores_uncond[stock],
                    metric,
                    title=f"{metric.capitalize()} Loss ({stock})",
                    plot_cis=False,
                    save_path=f"{plot_dir}/spider_{stock}_{metric}.png"
                )

    if len(time_lagged_files) > 0:
        # TIME-LAGGED SUMMARY PLOTS
        print("[*] Plotting time-lagged summary stats")
        summary_stats_time_lagged = {
            stock: {
                model: scoring.summary_stats(
                    all_scores_time_lagged[stock][model],
                    bootstrap=True
                )
                for model in all_scores_time_lagged[stock].keys()
            } for stock in all_scores_time_lagged
        }
        print(summary_stats_time_lagged)

        plotting.summary_plot(
            summary_stats_time_lagged,
            save_path=f"{plot_dir}/summary_stats_time_lagged.png"
        )

        # COMPARISON PLOTS: bar plots and spider plots for time-lagged
        print("[*] Plotting time-lagged comparison plots")
        data_time_lagged = _scores_to_df(all_scores_time_lagged)
        for stock in data_time_lagged.stock.unique():
            for metric in data_time_lagged.metric.unique():
                print(f"[*] Plotting {stock} {metric} time-lagged bar plots")
                plotting.loss_bars(
                    data_time_lagged,
                    stock,
                    metric,
                    save_path=f"{plot_dir}/bar_time_lagged_{stock}_{metric}.png"
                )
                print(f"[*] Plotting {stock} {metric} time-lagged spider plots")
                plotting.spider_plot(
                    all_scores_time_lagged[stock],
                    metric,
                    title=f"{metric.capitalize()} Loss - Time-Lagged ({stock})",
                    plot_cis=False,
                    save_path=f"{plot_dir}/spider_time_lagged_{stock}_{metric}.png"
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
    
    if len(cond_files) & args.histograms > 0:
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
                        f"{plot_dir}/hist_time_lagged_{stock}_{model}_{var_eval}_{var_cond}.png",
                        bbox_inches="tight", dpi=300
                    )
                    plt.close()

    if len(context_files) > 0:
        # CONTEXTUAL score histograms and regime analysis
        print("[*] Plotting contextual regime histograms")
        for stock, score_stock in tqdm(all_dfs_context.items(), position=0, desc="Stock"):
            for model, score_model in tqdm(score_stock.items(), position=1, desc="Model", leave=False):
                for score_name, score_df in score_model.items():
                    print(f"[*] Plotting {stock} {model} contextual regimes for {score_name}")
                    
                    # Get unique regimes and plot each regime's distribution
                    regimes = sorted(score_df['group'].unique())
                    n_regimes = len(regimes)
                    
                    fig, axes = plt.subplots(1, n_regimes, figsize=(6*n_regimes, 5))
                    if n_regimes == 1:
                        axes = [axes]
                    
                    for ax, regime_id in zip(axes, regimes):
                        regime_data = score_df[score_df['group'] == regime_id]
                        
                        # Separate real and generated scores
                        real_scores = regime_data[regime_data['type'] == 'real']['score'].values
                        gen_scores = regime_data[regime_data['type'] == 'generated']['score'].values
                        
                        # Compute histogram bins
                        all_scores = np.concatenate([real_scores, gen_scores])
                        bins = np.histogram_bin_edges(all_scores, bins=15)
                        
                        # Plot histograms
                        ax.hist(real_scores, bins=bins, alpha=0.6, label='Real', 
                               color='#1f77b4', edgecolor='black')
                        ax.hist(gen_scores, bins=bins, alpha=0.6, label='Generated', 
                               color='#ff7f0e', edgecolor='black')
                        
                        ax.set_title(f'Regime {regime_id}', fontsize=12)
                        ax.set_xlabel('Score')
                        ax.set_ylabel('Frequency')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                    
                    plt.suptitle(f'{stock} {model} - {score_name} (Contextual Regimes)', 
                                fontsize=14)
                    plt.tight_layout()
                    plt.savefig(
                        f"{plot_dir}/hist_context_{stock}_{model}_{score_name.replace(' | ', '_')}.png",
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
    parser.add_argument("--histograms", action="store_true", default=False,
                        help="Plot histograms of scores")
    args = parser.parse_args()

    run_plotting(args,args.score_dir, args.plot_dir,args.model_name)
    if args.show_plots:
        plt.show()
