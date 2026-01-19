from typing import Any,Union
import numpy as np
import pandas as pd
import pathlib
from glob import glob
import pickle
import gzip
import argparse
from datetime import datetime

import data_loading
import scoring
import eval
import metrics


import time

###################### UNCONDITIONAL SCORING ########################
DEFAULT_METRICS = {
    'l1': metrics.l1_by_group,
    'wasserstein': metrics.wasserstein,
}


DEFAULT_SCORING_CONFIG = {
    "spread": {
        "fn": lambda m, b: eval.spread(m, b).values,
        "discrete": True,
    },
    "orderbook_imbalance": {
        "fn": lambda m, b: eval.orderbook_imbalance(m, b).values,
    },

    #  TIMES (log scale)
    "log_inter_arrival_time": {
        "fn": lambda m, b: np.log(
            eval.inter_arrival_time(m)
            .replace({0: 1e-9}).values.astype(float)
        ),
    },
    "log_time_to_cancel": {
        "fn": lambda m, b: np.log(
            eval.time_to_cancel(m)
            .dt.total_seconds()
            .replace({0: 1e-9})
            .values.astype(float)
        ),
    },

    # VOLUMES:
    "ask_volume_touch": {
        "fn": lambda m, b: eval.l1_volume(m, b).ask_vol.values,
    },
    "bid_volume_touch": {
        "fn": lambda m, b: eval.l1_volume(m, b).bid_vol.values,
    },
    "ask_volume": {
        "fn": lambda m, b: eval.total_volume(m, b, 10)
        .ask_vol_10.values,
    },
    "bid_volume": {
        "fn": lambda m, b: eval.total_volume(m, b, 10)
        .bid_vol_10.values,
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

    # TRADES
    "vol_per_min": {
        "fn": lambda m, b: eval.volume_per_minute(m, b).values,
    },
    "ofi": {
        "fn": lambda m, b: eval.orderflow_imbalance(m, b).values,
    },
    "ofi_up": {
        "fn": lambda m, b: eval.orderflow_imbalance_cond_tick(m, b, 1).values,
    },
    "ofi_stay": {
        "fn": lambda m, b: eval.orderflow_imbalance_cond_tick(m, b, 0).values,
    },
    "ofi_down": {
        "fn": lambda m, b: eval.orderflow_imbalance_cond_tick(m, b, -1).values,
    },
}


######################## CONDITIONAL SCORING ########################
DEFAULT_SCORING_CONFIG_COND = {
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

DEFAULT_SCORING_CONFIG_CONDEXT = {
    "ask_volume | spread": {
        "eval": {
            "fn": lambda m, b: eval.l1_volume(m, b).ask_vol.values,
        },
        "cond": {
            "fn": lambda m, b: eval.spread(m, b).values,
            "discrete": True,
        }
    },
}


DEFAULT_SCORING_CONFIG_CONTEXT = {
    "ask_volume | spread": {
        "fn": lambda m, b: eval.l1_volume(m, b).ask_vol.values,
        "context_fn": lambda m, b: eval.spread(m, b).values,
        "context_config": {
            "discrete": True,
        }
    },
    "bid_volume_touch | spread": {
        "fn": lambda m, b: eval.l1_volume(m, b).bid_vol.values,
        "context_fn": lambda m, b: eval.spread(m, b).values,
        "context_config": {
            "discrete": True,
        }
    },
}


def save_results(scores, scores_dfs, save_path, protocol=-1):
    # make sure the folder exists
    folder_path = save_path.rsplit("/", 1)[0]
    pathlib.Path(folder_path).mkdir(parents=True, exist_ok=True)
    # save tuple as pickle
    with gzip.open(save_path, 'wb') as f:
        tup = (scores, scores_dfs)
        pickle.dump(tup, f, protocol)


def load_results(save_path):
    with gzip.open(save_path, 'rb') as f:
        tup = pickle.load(f)
    return tup


def run_benchmark(
    args: argparse.Namespace,
    scoring_config: dict[str, Any] = None,
    scoring_config_cond: dict[str, Any] = None,
    scoring_config_context: dict[str, Any] = None,
    metric_config: dict[str, Any] = None,
) -> None:
    
    print("*** \tThe assumed file structure for files in this package is:\n"
                "***\t \t {DATA_DIR}/{MODEL}/{STOCK}/data_*\n"
                "***\twhereby DATA_DIR is passed as an argument when launching the script\n"
                "***\t{MODEL}s and {STOCK}s may either be passed as a str or a list of str\n"
                "***\tThe script will iterate over all combinations of models and stocks\n")

    if scoring_config is None:
        scoring_config = DEFAULT_SCORING_CONFIG
    if scoring_config_cond is None:
        scoring_config_cond = DEFAULT_SCORING_CONFIG_COND
    if scoring_config_context is None:
        scoring_config_context = DEFAULT_SCORING_CONFIG_CONTEXT
    if metric_config is None:
        metric_config = DEFAULT_METRICS

    if isinstance(args.stock, str):
        args.stock = [args.stock]
    if isinstance(args.model_name, str):
        args.model_name = [args.model_name]
    if isinstance(args.time_period, str):
        args.time_period = [args.time_period]
    if isinstance(args.model_version, str):
        args.model_version = [args.model_version]

    if args.model_version is None:
        args.model_version = [None]

    for stock in args.stock:
        for model_name in args.model_name:
            for time_period in args.time_period:
                for mv in args.model_version:
                    if mv is not None:
                        stock_model_path = f"{args.data_dir}/{model_name}/{stock}/{time_period}/{mv}"
                    else:
                        stock_model_path = f"{args.data_dir}/{model_name}/{stock}/{time_period}"

                    print(f"[*] Loading generated data from {stock_model_path}")
                    loader = data_loading.Simple_Loader(
                        stock_model_path  + "/data_real", 
                        stock_model_path  + "/data_gen", 
                        stock_model_path  + "/data_cond", 
                    )

                    # materialize all sequences, so we keep them in memory
                    # for multiple accesses
                    for s in loader:
                        s.materialize()

                    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    # NEW: Contextual Scoring
                    if True:
                        print("[*] Running contextual scoring:")
                        scores_context, score_dfs_context, plot_fns_context = scoring.run_benchmark(
                            loader,
                            scoring_config_context,
                            default_metric=metric_config,
                            contextual=True
                        )
                        print("[*] Saving contextual results...")
                        save_results(
                            scores_context,
                            score_dfs_context,
                            args.save_dir+"/scores"
                            + f"/scores_context_{stock}_{model_name}_{time_str}.pkl"
                        )
                        print("... done")

                    # if (not args.cond_only) and (not args.div_only):
                    #     print("[*] Running unconditional scoring")
                    #     scores, score_dfs, plot_fns = scoring.run_benchmark(
                    #         loader,
                    #         scoring_config,
                    #         # default_metric=metrics.l1_by_group
                    #         # default_metric=metrics.wasserstein
                    #         default_metric=metric_config
                    #     )
                    #     print("[*] Saving results...")
                    #     save_results(
                    #         scores,
                    #         score_dfs,
                    #         args.save_dir+"/scores"
                    #         + f"/scores_uncond_{stock}_{model_name}_{str(mv)}_{time_str}.pkl"
                    #     )
                    #     print("... done")

                    # if (not args.uncond_only) and (not args.div_only):
                    #     print("[*] Running conditional scoring")
                    #     scores_cond, score_dfs_cond, plot_fns_cond = scoring.run_benchmark(
                    #         loader,
                    #         scoring_config_cond,
                    #         # default_metric=metrics.l1_by_group
                    #         # default_metric=metrics.wasserstein
                    #         default_metric=metric_config
                    #     )
                    #     print("[*] Saving results...")
                    #     save_results(
                    #         scores_cond,
                    #         score_dfs_cond,
                    #         args.save_dir+"/scores"
                    #         + f"/scores_cond_{stock}_{model_name}_{time_str}.pkl"
                    #     )
                    #     print("...done")

                    # if (not args.cond_only) and (not args.uncond_only):
                    #     print("[*] Running divergence scoring")
                    #     scores_, score_dfs_, plot_fns_ = scoring.run_benchmark(
                    #         loader, scoring_config, metrics.l1_by_group,
                    #         divergence_horizon=args.divergence_horizon,
                    #         divergence=True
                    #     )
                    #     print("[*] Saving results...")
                    #     save_results(
                    #         scores_,
                    #         score_dfs_,
                    #         args.save_dir+"/scores"
                    #         + f"/scores_div_{stock}_{model_name}_"
                    #         + f"{args.divergence_horizon}_{time_str}.pkl"
                    #     )

                    # if args.div_error_bounds:
                    #     print("[*] Calculating divergence lower bounds...")
                    #     baseline_errors_by_score = scoring.calc_baseline_errors_by_score(
                    #         score_dfs_,
                    #         metrics.l1_by_group
                    #     )
                    #     print("[*] Saving baseline errors...")
                    #     save_results(
                    #         baseline_errors_by_score,
                    #         None,
                    #         args.save_dir+"/scores"
                    #         + f"/scores_div_{stock}_REAL_"
                    #         + f"{args.divergence_horizon}_{time_str}.pkl"
                    #     )

                    print("[*] Done")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock", nargs='+', default="GOOG")
    parser.add_argument("--time_period", nargs='+', default="2023_Jan")
    parser.add_argument("--model_version", nargs='+', default=None)
    parser.add_argument("--data_dir", default="./data/evalsequences", type=str)
    parser.add_argument("--save_dir", type=str,default="./results")
    parser.add_argument("--model_name", nargs='+', default="s5_main")
    parser.add_argument("--uncond_only", action="store_true")
    parser.add_argument("--cond_only", action="store_true")
    parser.add_argument("--div_only", action="store_true")
    parser.add_argument("--div_error_bounds", action="store_true")
    parser.add_argument("--divergence_horizon", type=int, default=100)
    args = parser.parse_args()

    assert not (args.uncond_only and args.cond_only), \
        "Cannot specify both uncond_only and cond_only as args"

    assert not (args.div_error_bounds and (args.uncond_only or args.cond_only)), \
        "Cannot calculate divergence error bounds without running divergence scoring"
    t0=time.time()
    run_benchmark(args)
    t1=time.time()
    print("Finished Run, time (s) is:", t1-t0)
