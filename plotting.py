from typing import Callable, Iterable, Optional, Any
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
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


def facet_grid_hist(
    score_df: pd.DataFrame,
    var_eval: str = '',
    var_cond: str = '',
    # ylabel: str= 'density',
    filter_groups_below_weight: Optional[float] = None,
    bins = 'auto',
    binwidth: Optional[float] = None,
    stock: str = "",
    model: str = "",
) -> None:
    score_df = score_df.copy()
    # add group weight to filter on
    score_df['weight'] = score_df.groupby('group').transform('count')['score'] / len(score_df)
    if filter_groups_below_weight is not None:
        score_df = score_df[score_df['weight'] >= filter_groups_below_weight]

    if (binwidth is None) and (bins == 'auto'):
        xmin = score_df.score.min()
        xmax = score_df.score.max()
        score_range = xmax - xmin
        unique_vals = np.sort(score_df.score.unique())
        n_unique = len(unique_vals)
        if n_unique < 30:
            min_diff = pd.Series(unique_vals).diff().min()
            binwidth = min_diff if min_diff > 0 else 1
            # bins = score_range / binwidth
            bins = np.arange(xmin, xmax+binwidth, binwidth)
        else:
            # apply FD rule by group and take mean FD rule over all groups
            binwidth = np.nanmean(
                score_df.groupby('group').apply(
                    lambda x: 2 * (x.score.quantile(0.75) - x.score.quantile(0.25)) / len(x) ** (1/3)
                ).values,
                dtype=np.float32
            )
            if binwidth == 0:
                n_bins = int(np.ceil(np.sqrt(score_df.groupby('group')['score'].count().mean())))
                binwidth = score_range / n_bins
            else:
                n_bins = np.ceil(score_range / binwidth).astype(int)
            bins = np.arange(xmin, xmax+binwidth, binwidth)

        if binwidth > score_range:
            binwidth = score_range
            bins = np.array([xmin, xmax+binwidth])
        if (
            np.isnan(binwidth)
            or np.isinf(binwidth)
            or binwidth == 0
            or n_unique < 2
        ):
            if n_unique < 30:
                binwidth = (xmax - xmin) / n_unique
                bins = np.arange(xmin, xmax+binwidth, binwidth)
            else:
                binwidth = None
                bins = 'auto'
        # if n_unique < 30:
        #     # binwidth = np.ceil(min_diff).astype(int)
        #     bins = n_unique
        #     binwidth = (score_df.score.max() - score_df.score.min()) / n_unique
        #     # bins = np.ceil(n_bins).astype(int)

    g = sns.FacetGrid(
        score_df, row="group",
        sharex=True, sharey=False,
        aspect=4, height=1
    )

    groups_to_drop = set()
    # remove outliers at .. x IQR or std from the mean
    # out_multiple = 4
    for i, (group, ax) in enumerate(g.axes_dict.items()):
        df_ = score_df.loc[score_df["group"] == group]
        # weight = len(df_) / len(score_df)
        weight = df_.weight.iloc[0]

        title = f'{var_cond} $ \\in$ [{df_.score_cond.min():.2e}, {df_.score_cond.max():.2e}] (w={weight:.2f})'
        ax.set_title(title)

        std = df_.score.std()
        iqr = df_.score.quantile(0.75) - df_.score.quantile(0.25)
        out_measure = iqr if iqr > 0 else std
        median = np.median(df_.score)
        # df_ = df_[
        #     (df_.score > median - out_multiple*out_measure)
        #     & (df_.score < median + out_multiple*out_measure)
        # ]

        n_unique_group = len(df_.score.unique())
        # don't plot empty groups
        if n_unique_group == 1:
            val = df_.score.iloc[0]
            # plot single bars manually for real and gen
            if "real" in df_.type.values:
                ax.hist(
                    val,
                    bins=[val, val+binwidth],
                    alpha=0.5,
                    edgecolor="black",
                    color="C0",
                )
            if "generated" in df_.type.values:
                ax.hist(
                    val,
                    bins=[val, val+binwidth],
                    alpha=0.5,
                    edgecolor="black",
                    color="C1",
                )
            ax.set_xlabel(var_eval)
            ax.set_ylabel("")
            continue
        elif n_unique_group == 0:
            groups_to_drop.add(group)
            continue

        sns.histplot(
            data=df_, x='score', hue='type',
            stat='density', common_bins=True, common_norm=False,
            ax=ax, alpha=0.5, bins=bins,
            # binwidth=binwidth,
            edgecolor="black",
            palette={"real": "C0", "generated": "C1"},
        )
        ax.set_xlabel(var_eval)
        ax.set_ylabel('')
        ax.get_legend().remove()

    # drop empy plots
    for group in groups_to_drop:
        g.axes_dict.pop(group)

    out_multiple = 2.5
    median, std = np.median(score_df.score), score_df.score.std()
    iqr = score_df.score.quantile(0.75) - score_df.score.quantile(0.25)
    xmax = score_df.score.max()
    if iqr == 0:
        lo = score_df.score.min()
        # use 99 quantile for hi
        hi = score_df.score.quantile(0.99)
    else:
        out_measure = iqr
        lo = np.maximum(median - out_multiple*out_measure, score_df.score.min())
        hi_outlier = median + out_multiple*out_measure
        hi_outlier = np.ceil(hi_outlier / binwidth) * binwidth
        hi_bin_add = binwidth if binwidth is not None else 0
        hi = np.minimum(hi_outlier, xmax + hi_bin_add)
    for ax in g.axes.flat:
        ax.set_xlim((lo, hi))

    suptitle = f'({var_eval} | {var_cond}) histograms ({stock}) [{model}]'
    plt.suptitle(suptitle, fontweight='bold')
    plt.legend(
        [
            Line2D([0], [0], color='C0', lw=4, alpha=0.5),
            Line2D([0], [0], color='C1', lw=4, alpha=0.5),
        ],
        ["real", "generated"],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=2,
        bbox_transform=ax.figure.transFigure,
    )
    plt.tight_layout()


def scatter(
    score_df: pd.DataFrame,
    title: str= '',
    xlabel: str= 'Group',
    ylabel: str= 'Score',
    ax: Optional[plt.Axes] = None,
):
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
    model_name: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    color: Optional[str] = None,
    baseline_errors: Optional[list[float]] = None,
):
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
        ax=ax,
        color=color,
        label=model_name,
    )
    if baseline_errors is not None:
        sns.lineplot(
            x=labels,
            y=baseline_errors,
            ax=ax,
            color='black',
            label=r'99% CI $\hat{D}[real|real]$',
            linestyle=':',
        )
    plt.legend(model_name, loc="lower right")
    if ax is None:
        ax = plt
    ax.errorbar(
        x=labels,
        y=cis.mean(axis=0),
        yerr=np.diff(cis, axis=0) / 2,
        fmt='none',
        color=color,
    )
    ax.tick_params(axis='both', which='major', labelsize=14)
    _finish_plot(title, xlabel, ylabel, ax)


def hist_subplots(
    plot_fns: dict[str, Callable[[str, plt.Axes], None]],
    axs: Optional[Iterable[plt.Axes]] = None,
    figsize: Optional[tuple[int, int]] = None,
    suptile: Optional[str] = None,
    save_path: Optional[str] = None,
    plot_legend: bool = True,
) -> Iterable[plt.Axes]:
    """
    """
    if axs is None:
        fig, axs = plt.subplots(np.ceil(len(plot_fns) / 2).astype(int), 2, figsize=figsize)
    axs = axs.reshape(-1)

    for i, (name, fn) in enumerate(plot_fns.items()):
        fn(name, axs[i])
        # axs[i].set_xlim(x_ranges[i])
        if (i == 0) and plot_legend:
            # move legend location to the right
            axs[i].legend(
                loc='upper center',
                bbox_to_anchor=(0.5, 1.45),
                ncol=3,
            )
        if i > 0:
            legend = axs[i].get_legend()
            if legend:
                legend.remove()

    # lines_labels = [ax.get_legend_handles_labels() for ax in axs]
    # lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    # # fig.legend(lines, labels)
    # plt.legend(
    #     lines,
    #     labels,
    #     loc='upper center',
    #     bbox_to_anchor=(0.5, -1.5),
    #     ncol=2
    # )
    if suptile is not None:
        plt.suptitle(suptile, fontsize=16, fontweight='bold', y=1.002)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(
            save_path,
            dpi=300, bbox_inches='tight'
        )
    return axs


def spider_plot(
    scores_models: dict[str, Any],
    metric_str: str,
    title: str = '',
    plot_cis: bool = False,
    save_path: Optional[str] = None,
) -> go.Figure:

    assert len(scores_models) <= 4, "Only max 4 models can be compared at once"

    default_colors = [
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
            # s = s.split("_")
            # ln_break_at = len(s) // 2
            # s = " ".join(s[:ln_break_at]) + "<br>" + " ".join(s[ln_break_at:])
            # labels.append(s)
            labels.append(s.replace("_", " "))

        # pal = sns.color_palette("rocket", len(scores))
        losses = np.array([s[metric_str][0]for s in scores.values()])
        cis = np.array([s[metric_str][1]for s in scores.values()]).T

        fig.add_trace(go.Scatterpolar(
            r=-losses,
            theta=labels,
            mode='lines',
            fill='toself',
            # fillcolor='rgba(31, 119, 180, 0.2)',  # Set opacity to 0.2
            fillcolor=default_colors[i],
            name=model,
            # line=dict(color='rgba(31, 119, 180, 1)'),
            line=dict(color=default_colors[i]),
        ))
        if plot_cis:
            fig.add_trace(go.Scatterpolar(
                r=-cis[0],
                theta=labels,
                mode='lines',  # Add this line to remove markers
                name='CI lower',
                line=dict(color=default_colors[i], width=0),
                showlegend=False,
            ))
            fig.add_trace(go.Scatterpolar(
                r=-cis[1],
                theta=labels,
                mode='lines',  # Add this line to remove markers
                fill='tonext',
                fillcolor=default_colors[i],
                name='CI upper',
                line=dict(color=default_colors[i], width=0),
                showlegend=False,
            ))

    # get minimum score to set the range
    max_loss = max([
        s[metric_str][0] for scores in scores_models.values()
            for s in scores.values()
    ])
    fig.update_layout(
        width=600,
        height=400,
        polar=dict(
            angularaxis=dict(
                rotation=180 - 180/len(labels),
                direction='clockwise'  # Set the rotation direction (clockwise or counterclockwise)
            ),
            radialaxis=dict(
                visible=True,
                dtick=0.5,
                range=[-max_loss*1.1, 0],
                # angle=90,
                # tickangle=90,
            ),
        ),
        showlegend=True,
        title=f'{title}',
        margin=dict(l=200, r=200, t=50, b=50),
        legend=dict(
            orientation='h',  # Horizontal orientation
            yanchor='bottom',  # Anchor the legend to the bottom
            y=-0.2,#-0.4,  # Position it below the plot
            xanchor='center',  # Center the legend
            x=0.5,  # Position it horizontally centered
        ),
        font=dict(size=18),
    )

    # fig.write_image(f"images/spiderplt_{stock}_{metric_str}.png")
    if save_path is not None:
        fig.write_image(save_path, scale=3)
    return fig


def summary_plot(
    summary_stats_stocks: dict[str, dict[str, Any]],
    # stock: str,
    xranges: Optional[list[tuple[float, float]]] = None,
    save_path: Optional[str] = None,
):
    # collect all unique stats used
    n_stocks = len(summary_stats_stocks)
    stat_names = []
    for stock, summary_stats_models in summary_stats_stocks.items():
        for _, summary_stats in summary_stats_models.items():
            stats_names = list(summary_stats.keys())
            for stat_name in stats_names:
                if stat_name not in stat_names:
                    stat_names.append(stat_name)
    n_stats = len(stat_names)
    fig, axs = plt.subplots(n_stats * n_stocks, 1, figsize=(5, (2+(n_stocks+1)*n_stats)/2.5))
    plt.subplots_adjust(hspace=0.6)  # Set some space for all

    def _get_ax(i_stat, i_stock):
        return axs[i_stat * n_stocks + i_stock]

    for i_stock, (stock, summary_stats_models) in enumerate(summary_stats_stocks.items()):
        # error statistics (e.g. L1, Wasserstein)
        for i_stat, loss_metric in enumerate(stat_names):
            # plt.figure(figsize=(5,2))
            # models: e.g. genAI vs benchmark
            for i_model, (model, summary_stats) in enumerate(summary_stats_models.items()):
                if loss_metric in summary_stats:
                    scatter_vals = summary_stats[loss_metric]
                    scatter_x = np.array([val[0] for val in scatter_vals])
                    cis = np.array([val[1] for val in scatter_vals])
                    ax = _get_ax(i_stat, i_stock)
                    if np.isnan(scatter_x).all():
                        continue

                    ax.scatter(
                        scatter_x,
                        ['mean', 'median', 'IQM'],
                        marker='x',
                        color=f"C{i_model}",
                        label=model,
                    )
                    # add errorbars
                    ax.errorbar(
                        x=cis.mean(axis=1),
                        y=['mean', 'median', 'IQM'],
                        xerr=np.diff(cis, axis=1).T[0],
                        fmt='none',
                        color=f'C{i_model}',
                    )
                    ax.set_ylim(-1, 3)

                    # Add text to the left of the y-axis in the first subplot
                    ax.text(
                        -0.25, 0.5, stock, fontsize=12,
                        ha='center', va='center',
                        transform=ax.transAxes
                    )

            # set xrange
            # plt.xlim(*xranges[i])
            _get_ax(i_stat, i_stock).set_title(
                loss_metric.capitalize(),
                fontweight='bold',
                loc='left', #pad=-20,
            )

    # set x-axis limits
    for i_stat in range(n_stats):
        if xranges is None:
            # get min and max x limits for all stocks
            xlims = [_get_ax(i_stat, i_stock) for i_stock in range(n_stocks)]
            min_x, max_x = (
                min([ax.get_xlim()[0] for ax in xlims]),
                max([ax.get_xlim()[1] for ax in xlims])
            )
        else:
            min_x, max_x = xranges[i_stat]

        for i_stock in range(n_stocks):
            _get_ax(i_stat, i_stock).set_xlim(min_x, max_x)


        for i_stock in range(1, n_stocks):
            ax_ref = _get_ax(i_stat, i_stock-1)
            ax = _get_ax(i_stat, i_stock)

            # remove x-axis labels for all but the last stock
            ax.set_xticklabels([])

            # remove space between subplots for the same stat
            ax.set_position([
                ax.get_position().x0,
                ax_ref.get_position().y0 + ax_ref.get_position().height,
                ax.get_position().width,
                ax.get_position().height
            ])

    # Collect handles and labels for the combined legend
    handles, labels = [], []
    seen_labels = set()
    for ax in axs.flatten():
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in seen_labels:
                handles.append(handle)
                labels.append(label)
                seen_labels.add(label)
    plt.legend(
        handles, labels,
        loc='upper center',
        bbox_to_anchor=(0.5, -1.5),
        ncol=3
    )

    if save_path is not None:
        plt.savefig(
            save_path,
            dpi=300, bbox_inches='tight'
        )


def loss_bars(
    data: pd.DataFrame,
    stock: str,
    metric: str,
    save_path: Optional[str] = None,
) -> plt.Axes:
    data_ = data.loc[(data.metric==metric) & (data.stock==stock)].copy()
    n_models = len(data_.model.unique())

    data_.score = data_.score.str.replace('_', ' ').str.capitalize()

    fig = plt.figure(figsize=(6, 3.6))
    # BAR PLOT
    ax = sns.barplot(
        data=data_,
        x="score",
        y="mean",
        hue="model",
    )
    # ERROR BARS
    # get x-coords of bars only (without the legend elements)
    x_coords = [p.get_x() + 0.5*p.get_width() for p in ax.patches][:-n_models]
    y_coords = data_.loc[:, ["ci_low", "ci_high"]].dropna().mean(axis=1)
    y_err = data_.loc[:, ["ci_low", "ci_high"]].dropna().diff(axis=1).iloc[:,1] / 2
    ax.errorbar(
        x=x_coords,
        y=y_coords,
        yerr=y_err,
        fmt="none",
        c="k",
        elinewidth=3
    )
    plt.title(
        f'{metric.capitalize()} Loss ({stock})',
        fontsize=16,
        fontweight='bold'
    )
    plt.ylabel(f'{metric.capitalize()} Loss', fontsize=14)
    plt.xlabel('')
    # Customize tick labels
    _ = plt.xticks(rotation=90)
    ax.tick_params(axis='x', labelsize=14)  # X tick labels font size
    ax.tick_params(axis='y', labelsize=14)  # Y tick labels font size

    if save_path is not None:
        plt.savefig(
            save_path,
            dpi=300, bbox_inches='tight'
        )
    return ax


def _finish_plot(
        title: str = '',
        xlabel: str = 'Group',
        ylabel: str = 'Score',
        ax: Optional[plt.Axes] = None,
    ):
    """
    """
    if ax is not None:
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_title(title, fontsize=18)
    else:
        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.title(title, fontsize=18)


def get_plot_fn_uncond(score_df: pd.DataFrame) -> Callable[[str, plt.Axes], None]:
    unique_scores = score_df.score.unique()
    if unique_scores.shape[0] < 80:
        # discrete = True
        min_diff = pd.Series(
            score_df.score.unique()
        ).sort_values().diff().min()
        binwidth = min_diff if min_diff > 0 else 1
    else:
        # discrete = False
        binwidth = None

    def _score_hist_plot(name: str, ax: plt.Axes) -> None:
        # mean, std = score_df.score.mean(), score_df.score.std()
        # xmin = max(mean - 3*std, score_df.score.min())
        # xmax = min(mean + 3*std, score_df.score.max())
        # get outliers based on quantiles
        q1, q3 = score_df.score.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower_bound = q1 - 2 * iqr
        upper_bound = q3 + 2 * iqr
        xmin = max(score_df.score.min(), lower_bound)
        xmax = min(score_df.score.max(), upper_bound)
        if xmin == xmax:
            xmin, xmax = score_df.score.quantile([0.01, 0.99])
        sns.histplot(
            score_df,
            x="score",
            hue="type",
            stat="density",
            common_bins=True,
            ax=ax,
            # discrete=discrete,
            binwidth=binwidth,
            legend=True,
        )
        ax.set_title(name.capitalize().replace("_", " "), fontsize=18)
        ax.set_xlabel("")
        ax.set_xlim(xmin, xmax)
        ax.set_ylabel("Density", fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=14)

    return _score_hist_plot


# def get_plot_fn_cond(
#     score_df: pd.DataFrame
# ) -> Callable[[str, str, int, float], None]:
#     def _plot_fn(
#         var_eval: str,
#         var_cond: str,
#         bins: int,
#         binwidth: float,
#     ) -> None:
#         return facet_grid_hist(
#             score_df,
#             var_eval=var_eval,
#             var_cond=var_cond,
#             filter_groups_below_weight=0.01,
#             bins=bins,
#             binwidth=binwidth,
#         )
#   return _plot_fn


def get_div_plot_fn(
    scores: list[tuple[float, np.ndarray]],
    horizon_length: int,
    color: str = None,
    model_name: Optional[str] = None,
    # FIXME: types
    baseline_errors: Optional[list[float]] = None,
) -> Callable[[str, plt.Axes], None]:
    def _div_plot_fn(title, ax):
        error_divergence_plot(
            scores,
            horizon_length,
            title=title,
            xlabel="Prediction Horizon [messages]",
            ylabel="L1 score",
            model_name=model_name,
            ax=ax,
            color=color,
            baseline_errors=baseline_errors,
        )
    return _div_plot_fn
