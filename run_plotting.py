from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from glob import glob
import argparse
from datetime import datetime
import scoring
import plotting
from run_bench import load_results


def _load_all_scores(files):
    all_scores = {}
    for f in files:
        stock = f.rsplit("/", 1)[-1].split("_")[2]
        model = f.rsplit("/", 1)[-1].split("_")[3]
        scores, scores_dfs = load_results(f)
        if stock not in all_scores:
            all_scores[stock] = {}
        all_scores[stock][model] = scores
    return all_scores


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
    uncond_files = glob(score_dir + "/scores_uncond_*.pkl")
    cond_files = glob(score_dir + "/scores_cond_*.pkl")

    # load all scores
    all_scores_uncond = _load_all_scores(uncond_files)
    all_scores_cond = _load_all_scores(cond_files)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--score_dir", type=str)
    parser.add_argument("--plot_dir", default="images", type=str)
    args = parser.parse_args()

    run_plotting(args.score_dir, args.plot_dir)
    plt.show()
