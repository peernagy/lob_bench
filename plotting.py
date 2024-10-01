from typing import Callable, Optional, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
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
    print("plotting.py: n_Bins to use in histogram",len(bins))
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


def spider_plot(
    scores_models: dict[str, Any],
    stock: str,
    metric_str: str,
    title: str = '',
    plot_cis: bool = False,
) -> go.Figure:

    assert len(scores_models) <= 4, "Only max 4 models can be compared at once"

    default_cols = [
        "rgba(0.12156863, 0.46666667, 0.70588235, 0.2)",
        "rgba(1.0, 0.59607843, 0.2, 0.2)",
        "rgba(0.16862745, 0.78039216, 0.45098039, 0.2)",
        "rgba(0.90196078, 0.25098039, 0.25098039, 0.2)"
    ]

    fig = go.Figure()
    for i, (model, scores) in enumerate(scores_models.items()):

        # labels = [s.replace("_", "<br>") for s in list(scores.keys())]
        labels = []
        for s in scores.keys():
            s = s.split("_")
            ln_break_at = len(s) // 2
            s = " ".join(s[:ln_break_at]) + "<br>" + " ".join(s[ln_break_at:])
            labels.append(s)

        # pal = sns.color_palette("rocket", len(scores))
        losses = np.array([s[metric_str][0]for s in scores.values()])
        cis = np.array([s[metric_str][1]for s in scores.values()]).T

        fig.add_trace(go.Scatterpolar(
            r=-losses,
            theta=labels,
            mode='lines',
            fill='toself',
            # fillcolor='rgba(31, 119, 180, 0.2)',  # Set opacity to 0.2
            fillcolor=default_cols[i],
            name=model,
            # line=dict(color='rgba(31, 119, 180, 1)'),
            line=dict(color=default_cols[i]),
        ))
        if plot_cis:
            fig.add_trace(go.Scatterpolar(
                r=-cis[0],
                theta=labels,
                mode='lines',  # Add this line to remove markers
                name='CI lower',
                line=dict(color=default_cols[i], width=0),
                showlegend=False,
            ))
            fig.add_trace(go.Scatterpolar(
                r=-cis[1],
                theta=labels,
                mode='lines',  # Add this line to remove markers
                fill='tonext',
                fillcolor=default_cols[i],
                name='CI upper',
                line=dict(color=default_cols[i], width=0),
                showlegend=False,
            ))

    fig.update_layout(
        width=600,
        height=500,
        polar=dict(
            radialaxis=dict(
                visible=True,
                dtick=0.2,
                range=[-1.2, 0]
        )),
        showlegend=True,
        title=f'{title} ({stock})',
        # margin=dict(l=20, r=20, t=40, b=40),
        margin=dict(l=100, r=100, t=100, b=100),
        legend=dict(
            orientation='h',  # Horizontal orientation
            yanchor='bottom',  # Anchor the legend to the bottom
            y=-0.4,  # Position it below the plot
            xanchor='center',  # Center the legend
            x=0.5  # Position it horizontally centered
        ),
        font=dict(size=18),
    )

    fig.write_image(f"images/{metric_str}_spiderplt_{stock}.png")
    return fig


def summary_plot(
    summary_stats_models: dict[str, dict[str, Any]],
    stock: str,
    xranges: Optional[list[tuple[float, float]]] = None,
):
    # collect all unique stats used
    stat_names = []
    for _, summary_stats in summary_stats_models.items():
        stats_names = list(summary_stats.keys())
        for stat_name in stats_names:
            if stat_name not in stat_names:
                stat_names.append(stat_name)

    fix, axs = plt.subplots(2, 1, figsize=(5, 3))

    # error statistics (e.g. L1, Wasserstein)
    for i_stat, loss_metric in enumerate(stat_names):
        # plt.figure(figsize=(5,2))
        # models: e.g. genAI vs benchmark
        for i_model, (model, summary_stats) in enumerate(summary_stats_models.items()):
            if loss_metric in summary_stats:
                scatter_vals = summary_stats[loss_metric]
                scatter_x = [val[0] for val in scatter_vals]
                cis = np.array([val[1] for val in scatter_vals])
                axs[i_stat].scatter(
                    scatter_x,
                    ['mean', 'median', 'IQM'],
                    color=f'C{i_model}',
                    marker='x',
                    label=model,
                )
                # add errorbars
                axs[i_stat].errorbar(
                    x=cis.mean(axis=1),
                    y=['mean', 'median', 'IQM'],
                    xerr=np.diff(cis, axis=1).T[0],
                    fmt='none',
                    color=f'C{i_model}',
                )

        # set xrange
        # plt.xlim(*xranges[i])
        axs[i_stat].set_title(
            f"{stock} Model Summary ({loss_metric})",
            fontweight='bold'
        )
        # plt.legend()

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
    plt.tight_layout(pad=2.0)
    # plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.2)
    plt.savefig(
        f'images/summary_stats_{loss_metric}_{stock}.png',
        dpi=300, bbox_inches='tight'
    )


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