from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import argparse
from datetime import datetime
import scoring
import plotting
from run_bench import load_results


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
                for metric_name, metric_val in metric_score.items():
                    mean, ci, _ = metric_val
                    rows.append((stock, model, metric_name, score_name, mean, ci[0], ci[1]))
    return pd.DataFrame(
        rows,
        columns=['stock', 'model', 'metric', 'score', 'mean', 'ci_low', 'ci_high']
    )


def run_plotting(
    score_dir: str,
    plot_dir: str,
) -> None:
    # load all saved stats
    uncond_files = sorted(glob(score_dir + "/scores_uncond_*.pkl"))
    cond_files = sorted(glob(score_dir + "/scores_cond_*.pkl"))
    div_files = sorted(glob(score_dir + "/scores_div_*.pkl"))

    # load all scores
    all_scores_uncond, all_dfs_uncond = _load_all_scores(uncond_files)
    all_scores_cond, all_dfs_cond = _load_all_scores(cond_files)
    all_scores_div, all_dfs_div = _load_all_scores(div_files)

    # divergence plots
    print("[*] Plotting divergence plots")
    for stock, score_stock in tqdm(all_scores_div.items(), position=0, desc="Stock"):
        for model, score_model in tqdm(score_stock.items(), position=1, desc="Model", leave=False):
            plot_fns = {
                # TODO: add horizon_length to get_div_plot_fn from file name
                score_name: plotting.get_div_plot_fn(score_)
                    for score_name, score_ in score_model.items()
            }
        plotting.hist_subplots(
            plot_fns,
            figsize=(10, 25),
            suptile=f"L1 Divergence {stock} {model}",
            save_path=f'{plot_dir}/divergence_{stock}_{model}.png'
        )

    # plot individual histograms of scores
    print("[*] Plotting histograms")
    for stock, score_stock in tqdm(all_dfs_uncond.items(), position=0, desc="Stock"):
        for model, score_model in tqdm(score_stock.items(), position=1, desc="Model", leave=False):
            plot_fns = {
                score_name: plotting.get_plot_fn(score_df)
                    for score_name, score_df in score_model.items()
            }

            plotting.hist_subplots(
                plot_fns,
                figsize=(10, 25),
                suptile=f"{stock} {model}",
                save_path=f"{plot_dir}/hist_{stock}_{model}.png",
            )

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
    plotting.summary_plot(
        summary_stats,
        save_path=f"{plot_dir}/summary_stats_comp.png"
    )

    print("[*] Plotting comparison plots")
    # Bar plot of unconditional scores comparing all models
    data = _scores_to_df(all_scores_uncond)
    for stock in data.stock.unique():
        for metric in data.metric.unique():
            plotting.loss_bars(
                data,
                stock,
                metric,
                save_path=f"{plot_dir}/bar_{stock}_{metric}.png"
            )

            plotting.spider_plot(
                all_scores_uncond[stock],
                metric,
                title=f"{metric.capitalize()} Loss ({stock})",
                plot_cis=False,
                save_path=f"{plot_dir}/spider_{stock}_{metric}.png"
            )
    print("[*] Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--score_dir", type=str)
    parser.add_argument("--plot_dir", default="images", type=str)
    parser.add_argument("--show_plots", action="store_true")
    args = parser.parse_args()

    run_plotting(args.score_dir, args.plot_dir)
    if args.show_plots:
        plt.show()
