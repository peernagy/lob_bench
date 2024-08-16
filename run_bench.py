#3rd Party Packages
import numpy as np
import pandas as pd
import jax.numpy as jnp
import glob

#Lob-Bench Modules
import data_loading
import eval
import partitioning
import impact
import plotting
import metrics
import scoring
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns

print(plt)

def run_bench(model,stock,data_dir):
    # loader = data_loading.Simple_Loader(real_data_path='/data1/sascha/data/generated_data/GOOG/data_real_small',
    #                                     gen_data_path='/data1/sascha/data/generated_data/GOOG/data_gen_small',
    #                                     cond_data_path='/data1/sascha/data/generated_data/GOOG/data_cond_small')
    loader = data_loading.Simple_Loader(real_data_path=data_dir+'data_real',
                                        gen_data_path=data_dir+'data_gen',
                                        cond_data_path=data_dir+'data_cond')
    for s in loader:
        s.materialize()

    print(len(loader))
    scoring_config_dict = {
    "spread": {
        "fn": lambda m, b: eval.spread(m, b).values,#.mean(),
        "discrete": True,
    },
    "orderbook_imbalance": {
        "fn": lambda m, b: eval.orderbook_imbalance(m, b).values,
    },
    
    # TIMES (log scale)
    "log_inter_arrival_time": {
        # "fn": lambda m, b: eval.inter_arrival_time(m).values.astype(float),
        "fn": lambda m, b: np.log(eval.inter_arrival_time(m).replace({0: 1e-9}).values.astype(float)),
    },
    "log_time_to_cancel": {
        # "fn": lambda m, b: eval.time_to_cancel(m).values.astype(float),
        "fn": lambda m, b: np.log(eval.time_to_cancel(m).replace({0: 1e-9}).values.astype(float)),
    },
    
    # VOLUMES:
    "ask_volume_touch": {
        "fn": lambda m, b: eval.l1_volume(m, b).ask_vol.values,
    },
    "bid_volume_touch": {
        "fn": lambda m, b: eval.l1_volume(m, b).bid_vol.values,
    },
    "ask_volume": {
        "fn": lambda m, b: eval.total_volume(m, b, 10).ask_vol_10.values,
    },
    "bid_volume": {
        "fn": lambda m, b: eval.total_volume(m, b, 10).bid_vol_10.values,
    },

    # DEPTHS:
    "limit_ask_order_depth": {
        "fn": lambda m, b: eval.limit_order_depth(m, b)[0].values,
    },
    "limit_bid_order_depth": {
        "fn": lambda m, b: eval.limit_order_depth(m, b)[1].values,
    },
    "ask_cancellation_depth": {
        "fn": lambda m, b: eval.cancellation_depth(m, b)[0].values,
    },
    "bid_cancellation_depth": {
        "fn": lambda m, b: eval.cancellation_depth(m, b)[1].values,
    },

    # LEVELS:
    "limit_ask_order_levels": {
        "fn": lambda m, b: eval.limit_order_levels(m, b)[0].values,
        "discrete": True,
    },
    "limit_bid_order_levels": {
        "fn": lambda m, b: eval.limit_order_levels(m, b)[1].values,
        "discrete": True,
    },
    "ask_cancellation_levels": {
        "fn": lambda m, b: eval.cancel_order_levels(m, b)[0].values,
        "discrete": True,
    },
    "bid_cancellation_levels": {
        "fn": lambda m, b: eval.cancel_order_levels(m, b)[1].values,
        "discrete": True,
    },
    }   
    scoring_config_dict_cond = {
    ######################## CONDITIONAL SCORING ########################
    "ask_volume | spread": {
        "eval": {
            "fn": lambda m, b: eval.l1_volume(m, b).ask_vol.values,
        },
        "cond": {
            "fn": lambda m, b: eval.spread(m, b).values,
            "discrete": True,
        }
    },
    "spread | time": {
        "eval": {
            "fn": lambda m, b: eval.spread(m, b).values,
        },
        "cond": {
            "fn": lambda m, b: eval.time_of_day(m).values,
            # group by hour of the day (start of sequence)
            "thresholds": np.linspace(0, 24*60*60, 24),
        }
    },
    "spread | volatility": {
        "eval": {
            "fn": lambda m, b: eval.spread(m, b).values,
            "discrete": True,
        },
        "cond": {
            "fn": lambda m, b: [eval.volatility(m,b,'0.1s')] * len(m),
        }
    }
    }
    metrics_fns_dict={'l1': metrics.l1_by_group,
                    'wasserstein': metrics.wasserstein,
                    }

    scores, score_dfs, plot_fns = scoring.run_benchmark(
    loader,
    scoring_config_dict,
    default_metric=metrics_fns_dict
    )
    summary_stats=scoring.summary_stats(scores, bootstrap=True)

    for loss_metric in summary_stats.keys():
        scatter_vals = summary_stats[loss_metric]
        scatter_x = [val[0] for val in scatter_vals]
        cis = np.array([val[1] for val in scatter_vals])
        plt.figure(figsize=(5,2))
        plt.scatter(scatter_x, ['mean', 'median', 'IQM'])
        # add errorbars
        plt.errorbar(
            x=cis.mean(axis=1),
            y=['mean', 'median', 'IQM'],
            xerr=np.diff(cis, axis=1).T[0],
            fmt='none'
        )
        plt.title(f"{stock} Model Summary ({loss_metric})", fontweight='bold')
        plt.savefig(f'images/summary_stats_{loss_metric}_{stock}_{model}.png', dpi=300, bbox_inches='tight')

    labels = list(scores.keys())
    # pal = sns.color_palette("rocket", len(scores))
    losses = np.array([s['wasserstein'][0]for s in scores.values()])
    cis = np.array([s['wasserstein'][1]for s in scores.values()]).T

    plt.figure(figsize=(8,6))
    b = sns.barplot(
        x=labels,
        y=losses,
        hue=losses,
        legend='brief'
        # palette=np.array(pal[::-1])[rank]
    )
    plt.errorbar( # ([0.01]*len(labels), [0.03]*len(labels))
        x=labels, y=cis.mean(axis=0), yerr=np.diff(cis, axis=0)/2,
        fmt='none', elinewidth=3)
    _ = plt.xticks(rotation=90)

    plt.title('Wasserstein Loss of Generated Data Distributions')
    plt.ylabel('Wasserstein Loss')

    plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()

    # save fig as png
    plt.savefig(f'images/bar_plot_comparison_wasserstein_{stock}_{model}.png')

    labels = list(scores.keys())
    # pal = sns.color_palette("rocket", len(scores))
    losses = np.array([s['wasserstein'][0]for s in scores.values()])
    cis = np.array([s['wasserstein'][1]for s in scores.values()]).T

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=-losses,
        theta=labels,
        mode='lines',
        fill='toself',
        fillcolor='rgba(31, 119, 180, 0.2)',  # Set opacity to 0.5
        name='CI upper',
        line=dict(color='rgba(31, 119, 180, 1)'),
    ))
    fig.add_trace(go.Scatterpolar(
        r=-cis[0],
        theta=labels,
        mode='lines',  # Add this line to remove markers
        name='CI lower',
        line=dict(color='rgba(31, 119, 180, 0.2)', width=0),
        # fill='tonext',
    ))
    fig.add_trace(go.Scatterpolar(
        r=-cis[1],
        theta=labels,
        mode='lines',  # Add this line to remove markers
        fill='tonext',
        fillcolor='rgba(31, 119, 180, 0.5)',
        name='CI upper',
        line=dict(color='rgb(31, 119, 180)', width=0),
    ))

    fig.update_layout(
    polar=dict(
        radialaxis=dict(
        visible=True,
        range=[-0.8, 0]
        )),
    showlegend=False,
    title='Wasserstein Distances',
    margin=dict(l=20, r=20, t=40, b=40),
    )

    fig.write_image(f"images/wasserstein_comparison_{stock}_{model}.png")
    fig.show()
    # L1 Loss


    labels = list(scores.keys())
    # pal = sns.color_palette("rocket", len(scores))
    losses = np.array([s['l1'][0]for s in scores.values()])
    cis = np.array([s['l1'][1]for s in scores.values()]).T

    plt.figure(figsize=(8,6))
    b = sns.barplot(
        x=labels,
        y=losses,
        hue=losses,
        legend='brief'
        # palette=np.array(pal[::-1])[rank]
    )
    plt.errorbar( # ([0.01]*len(labels), [0.03]*len(labels))
        x=labels, y=cis.mean(axis=0), yerr=np.diff(cis, axis=0)/2,
        fmt='none', elinewidth=3)
    _ = plt.xticks(rotation=90)

    plt.title('L1 Loss of Generated Data Distributions')
    plt.ylabel('L1 Loss')

    plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()

    # save fig as png
    plt.savefig(f'images/bar_plot_comparison_l1_{stock}_{model}.png')


    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=1-losses,
        theta=labels,
        mode='lines',
        fill='toself',
        fillcolor='rgba(31, 119, 180, 0.2)',  # Set opacity to 0.5
        name='CI upper',
        line=dict(color='rgba(31, 119, 180, 1)'),
    ))
    fig.add_trace(go.Scatterpolar(
        r=1-cis[0],
        theta=labels,
        mode='lines',  # Add this line to remove markers
        name='CI lower',
        line=dict(color='rgba(31, 119, 180, 0.2)', width=0),
        # fill='tonext',
    ))
    fig.add_trace(go.Scatterpolar(
        r=1-cis[1],
        theta=labels,
        mode='lines',  # Add this line to remove markers
        fill='tonext',
        fillcolor='rgba(31, 119, 180, 0.5)',
        name='CI upper',
        line=dict(color='rgb(31, 119, 180)', width=0),
    ))

    fig.update_layout(
    polar=dict(
        radialaxis=dict(
        visible=True,
        range=[0.5, 1]
        )),
    showlegend=False,
    title='L1 Scores',
    margin=dict(l=20, r=20, t=40, b=40),
    )

    fig.write_image(f"images/l1_comparison_{stock}_{model}.png")
    fig.show()
    # Plot all histograms in single subplot
    plotting.hist_subplots(plot_fns, (10, 25))
    plt.suptitle(f'{stock} Data Distributions', fontweight='bold')
    plt.savefig(f'images/hist_subplots_{stock}_{model}.png', dpi=300, bbox_inches='tight')


    scores_, score_dfs_, plot_fns_ = scoring.run_benchmark(
        loader, scoring_config_dict, metrics.l1_by_group,
        divergence=True)

    plotting.hist_subplots(plot_fns_, (10, 25))
    plt.suptitle(f'L1 Divergence ({stock})', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'images/metric_divergence_{stock}_{model}.png', dpi=300, bbox_inches='tight')

    impact.impact_compare(loader,ticker=stock,model=model)

if __name__ == "__main__":
    import argparse
    #physical_devices = tf.config.list_physical_devices('GPU')
    #tf.config.experimental.set_memory_growth(physical_devices[0], True)
    #tf.config.experimental.set_visible_devices([], "GPU")

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="S5_Large",
                        help="Model type")
    parser.add_argument("--stock", type=str, default="GOOG",
                        help="Stock ticker")
    parser.add_argument("--data_dir", type=str, default="/data1/sascha/data/generated_data/GOOG",
                        help="Location of gen/real/cond data")

    args = parser.parse_args()
    print(f"Running benchmarks for model {args.model} and stock {args.stock}. \n Should be equivalent behaviour to tutorial notebook.")


    run_bench(args.model, args.stock, args.data_dir)