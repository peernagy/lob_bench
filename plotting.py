from typing import Callable, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from partitioning import flatten


def hist(
        real,
        gen,
        title: str= '',
        xlabel: str= '',
        ylabel: str= 'density',
        *,
        bins: Optional[int|str|list[float]] = 'auto',
        binwidth: Optional[float] = None,
        ax: Optional[plt.Axes] = None,
    ):
    """
    """
    real = pd.DataFrame(flatten(real))
    real['type'] = 'real'
    gen = pd.DataFrame(flatten(gen))
    gen['type'] = 'generated'
    data = pd.concat([real, gen], axis=0)
    data.columns = ['x', 'type']

    # display(data)
    sns.histplot(
        data, x='x', hue='type', #kde=True,
        alpha=0.5, stat='density', multiple='layer', line_kws={'linewidth': 2},
        common_bins=True, common_norm=False, binwidth=binwidth, bins=bins,
        ax=ax
    )
    sns.move_legend(
        ax, "lower center",
        bbox_to_anchor=(.5, 1.1), ncol=3, title=None, frameon=False,
    )
    # sns.histplot(gen, color='red', alpha=0.5, label='Generated', ax=plt.gca())
    # plt.legend()
    _finish_plot(title, xlabel, ylabel, ax)
    # if not isinstance(bins, str):
    #     print(f'bins: {bins}')


def facet_grid_hist(
        score_df: pd.DataFrame,
        var_eval: str = '',
        var_cond: str = '',
        # ylabel: str= 'density',
        filter_groups_below_weight: Optional[float] = None,
        bins = 'auto',
        binwidth: Optional[float] = None,
    ):
    """
    """

    score_df = score_df.copy()
    # add group weight to filter on
    score_df['weight'] = score_df.groupby('group').transform('count')['score'] / len(score_df)
    if filter_groups_below_weight is not None:
        score_df = score_df[score_df['weight'] >= filter_groups_below_weight]

    g = sns.FacetGrid(
        score_df, row="group",
        sharex=True, sharey=False,
        aspect=4, height=1
    )

    # TODO: remove this to make it general
    # x_ranges = [
    #     ()
    # ]

    for i, (group, ax) in enumerate(g.axes_dict.items()):
        df_ = score_df.loc[score_df["group"] == group]
        # weight = len(df_) / len(score_df)
        weight = df_.weight.iloc[0]
        print('weigth:', weight)
        sns.histplot(
            data=df_, x='score', hue='type',
            stat='density', common_bins=True, common_norm=False,
            ax=ax, alpha=0.5, bins=bins,
            #bins=100,
            # TODO: for other data
            binwidth=binwidth
        )
        ax.set_title(
            f'{var_cond} $ \\in$ [{df_.score_cond.min():.2e}, {df_.score_cond.max():.2e}] (w={weight:.2f})')
        # remove legend
        ax.get_legend().remove()
        ax.set_xlabel(var_eval)
        ax.set_ylabel('')
        # ax.set_xlim((100, 600))  # spread
        ax.set_xlim((0, 1500))  # total volume
    
    plt.suptitle(f'({var_eval} | {var_cond}) histograms', fontweight='bold')
    plt.tight_layout()

def scatter(
        score_df: pd.DataFrame,
        title: str= '',
        xlabel: str= 'Group',
        ylabel: str= 'Score',
        ax: Optional[plt.Axes] = None,
    ):
    """
    """
    sns.scatterplot(data=score_df, x='group', y='score', hue='type')
    _finish_plot(title, xlabel, ylabel, ax)


def line(
        score_df: pd.DataFrame,
        title: str= '',
        xlabel: str= 'Group',
        ylabel: str= 'Score',
        ax: Optional[plt.Axes] = None,
    ):
    """
    """
    sns.lineplot(
        data=score_df, x='group', y='score', hue='type',
        errorbar=("pi", 0.95)
    )
    _finish_plot(title, xlabel, ylabel, ax)


def error_divergence_plot(
        loss_horizons: list[tuple[float, np.ndarray]],
        horizon_length: int,
        title: str= '',
        xlabel: str= 'Prediction Horizon',
        ylabel: str= 'Error',
        ax: Optional[plt.Axes] = None,
    ):
    """
    """
    labels = [f"{period}-{period+horizon_length}"
        for period in range(
            0,
            horizon_length*len(loss_horizons),
            horizon_length
        )
    ]
    l1s = np.array([l[0] for l in loss_horizons])
    cis = np.array([l[1] for l in loss_horizons]).T

    sns.lineplot(
        x=labels,
        y=l1s,
        ax=ax
    )
    if ax is None:
        ax = plt
    ax.errorbar(
        x=labels, y=cis.mean(axis=0), yerr=np.diff(cis, axis=0) / 2,
        fmt='none')
    _finish_plot(title, xlabel, ylabel, ax)


def hist_subplots(
        plot_fns: dict[str, Callable[[str, plt.Axes], None]],
        figsize: Optional[tuple[int, int]] = None,
    ):
    """
    """
    fig, axs = plt.subplots(np.ceil(len(plot_fns) / 2).astype(int), 2, figsize=figsize)
    axs = axs.reshape(-1)

    # x_ranges = [
    #     (100, 1000),
    #     (-1, 1),
    #     (-22, 8),
    #     (5, 25),
    #     (0, 20000),
    #     (0, 20000),
    #     (0, 100000),
    #     (0, 100000),
    #     (0, 1000),
    #     (0, 1000),
    #     (0, 1000),
    #     (0, 1000),
    #     (1, 10),
    #     (1, 10),
    #     (1, 10),
    #     (1, 10),
    # ]

    for i, (name, fn) in enumerate(plot_fns.items()):
        fn(name, axs[i])
        # axs[i].set_xlim(x_ranges[i])
        if i > 0:
            legend = axs[i].get_legend()
            if legend:
                legend.remove()

    lines_labels = [ax.get_legend_handles_labels() for ax in axs]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels)

    plt.tight_layout()


def _finish_plot(
        title: str = '',
        xlabel: str = 'Group',
        ylabel: str = 'Score',
        ax: Optional[plt.Axes] = None,
    ):
    """
    """
    if ax is not None:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    else:
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)