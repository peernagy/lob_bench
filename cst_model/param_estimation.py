import sys
import matplotlib.pyplot as plt
import glob
import pickle
import jax
import jax.numpy as jnp
from typing import Any, Iterable
from scipy.optimize import curve_fit
import pandas as pd

import cst
from cst import CSTParams

# FIXME: this is a hack to import from parent dir
# add parent dir to path
import sys
sys.path.append("..")
from lob_bench import data_loading


def pickle_params(params: dict[str, Any], path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(params, f)


def load_params(path: str) -> dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def estimate_params(
    book_data: pd.DataFrame,
    message_data: pd.DataFrame,
    tick_size: int,
    num_ticks: int,
) -> dict[str, Any]:
    # entire time duration
    total_time = float(message_data.time.iloc[-1] - message_data.time.iloc[0])
    # TODO: consider grouping executions at same time stamp?
    # number of executions / market orders
    mo_count = message_data[message_data.event_type == 4].shape[0]

    # average size of limit orders
    lo_size = message_data.loc[message_data.event_type == 1, "size"].mean()
    # average size of market orders
    mo_size = message_data.loc[message_data.event_type == 4, "size"].mean()
    # average size of cancellations
    co_size = message_data.loc[message_data.event_type.isin((2, 3)), "size"].mean()
    # print("Sl:", lo_size, "Sm:", mo_size, "Sc:", co_size)

    mean_spread = ((book_data.iloc[:, 0] - book_data.iloc[:, 2]) / tick_size).mean()

    # market order rate (sizes relative to limit orders)
    mo_mu = mo_count / total_time * mo_size / lo_size

    # TODO: calculate difference between message price and (opposite) best price
    #       to get depth of message --> twice, per side of book
    #       then, value_counts of depths to get distribution of depths
    #       and divide by T to get rates for LOs and COs

    lo_count, co_count = _get_counts_by_depth(book_data, message_data, tick_size, num_ticks)
    # fit power law to LO rates
    lo_sol = _fit_power_law(lo_count / total_time, plot_fit=False)

    # convert all lobster book data to jax array of volumes at depths
    book_jnp = jax.jit(jax.vmap(cst.init_book, in_axes=(0, None)), static_argnums=(1,))(
        jnp.array(book_data.values),
        CSTParams(
            tick_size=tick_size,
            num_ticks=num_ticks,
        )
    )
    # calculate Q_i from paper (mean volume at depth) averaged over bid and ask sides
    mean_size_at_depth = (book_jnp.asks.mean(axis=0) + book_jnp.bids.mean(axis=0)) / 2
    # print("mean_size_at_depth", mean_size_at_depth)

    co_theta = co_count / (mean_size_at_depth * total_time) * (co_size / lo_size)
    co_theta = co_theta.at[jnp.isnan(co_theta)].set(0)
    # print("co_theta", co_theta)

    params_dict = {
        "total_time": total_time,
        "num_events": message_data.shape[0],
        "mean_spread": mean_spread,
        "lo_size": lo_size,
        "mo_size": mo_size,
        "co_size": co_size,
        "lo_lambda": lo_count / total_time,
        "mo_count": mo_count,
        "mo_mu": mo_mu,
        "lo_count": lo_count,
        "co_count": co_count,
        "q_i": mean_size_at_depth,
        "co_theta": co_theta,
        "tick_size": tick_size,
        "num_ticks": num_ticks,
        "lo_k": lo_sol[0],
        "lo_alpha": lo_sol[1],
    }
    return params_dict


def _fit_power_law(y: jax.Array, plot_fit: bool = False) -> jax.Array:
    i = jnp.arange(1, y.shape[0] + 1)
    fit_params, pcov = curve_fit(cst.lo_rate, i, y)
    # print("LO fit", lo_sol)

    if plot_fit:
        plt.plot(i, y, label="LO")
        plt.plot(i, cst.lo_rate(i, *fit_params), label="LO fit")
        plt.title("Limit Order Counts")
        plt.xlabel("Depth")
        plt.ylabel("Count")
        plt.legend()
        plt.show()
    return fit_params


def estimate_data_file(
    orderbook_path: str,
    tick_size: int = 100,
    num_ticks: int = 500,
    save: bool = True,
):
    b_df = data_loading.load_book_df(orderbook_path)
    m_df = data_loading.load_message_df(orderbook_path.replace("orderbook", "message"))
    params = estimate_params(b_df, m_df, tick_size, num_ticks)
    if save:
        save_path = orderbook_path.replace("orderbook", "params").replace(".csv", ".pkl")
        print("Saving params to", save_path)
        pickle_params(params, save_path)
    return params


def estimate_from_data_files(
    data_path: str,
    tick_size: int = 100,
    num_ticks: int = 500,
):
    aggr_params = []
    book_files = glob.glob(data_path + "/*orderbook*.csv")
    for book_file in book_files:
        print("Estimating", book_file)
        aggr_params.append(estimate_data_file(book_file, tick_size, num_ticks, save=True))
    return aggregate_params(aggr_params, save_path=data_path + "/aggregated_params.pkl")


def aggregate_params(
    param_iter: Iterable[dict[str, Any]],
    save_path: str = None,
) -> dict[str, Any]:
    def _weighted_aggr(param_iter, field_name, aggr_params):
        return (
            jnp.mean(jnp.array([d[field_name] * d["num_events"] for d in param_iter]))
            / aggr_params["num_events"]
        )

    aggr = {
        "total_time": jnp.sum(jnp.array([d["total_time"] for d in param_iter])),
        "num_events": jnp.sum(jnp.array([d["num_events"] for d in param_iter])),
        # must be constant across all files: take from first
        "tick_size": param_iter[0]["tick_size"],
        "num_ticks": param_iter[0]["num_ticks"],
    }
    # weight aggregate by number of events
    aggr["mean_spread"] = _weighted_aggr(param_iter, "mean_spread", aggr)
    aggr["lo_size"] = _weighted_aggr(param_iter, "lo_size", aggr)
    aggr["mo_size"] = _weighted_aggr(param_iter, "mo_size", aggr)
    aggr["co_size"] = _weighted_aggr(param_iter, "co_size", aggr)
    aggr["mo_count"] = jnp.sum(jnp.array([d["mo_count"] for d in param_iter]))
    aggr["mo_mu"] = aggr["mo_count"] / aggr["total_time"] * aggr["mo_size"] / aggr["lo_size"]
    aggr["lo_count"] = jnp.stack([d["lo_count"] for d in param_iter], axis=0).sum(axis=0)
    aggr["co_count"] = jnp.stack([d["co_count"] for d in param_iter], axis=0).sum(axis=0)
    aggr["lo_lambda"] = aggr["lo_count"] / aggr["total_time"]
    aggr["q_i"] = _weighted_aggr(param_iter, "q_i", aggr)
    aggr["co_theta"] = (
        aggr["co_count"] / (aggr["q_i"] * aggr["total_time"])
        * (aggr["co_size"] / aggr["lo_size"])
    )

    lo_sol = _fit_power_law(aggr["lo_count"] / aggr["total_time"], plot_fit=False)
    aggr["lo_k"] = lo_sol[0]
    aggr["lo_alpha"] = lo_sol[1]

    if save_path is not None:
        print("Saving aggregated params to", save_path)
        pickle_params(aggr, save_path)

    return aggr


def _get_counts_by_depth(
    book_data: pd.DataFrame,
    message_data: pd.DataFrame,
    tick_size: int,
    num_ticks: int,
) -> tuple[jax.Array, jax.Array]:
    ask_counts = _get_counts_by_side(book_data, message_data, tick_size, is_bid=False)
    bid_counts = _get_counts_by_side(book_data, message_data, tick_size, is_bid=True)
    counts = pd.merge(
        ask_counts, bid_counts, left_index=True, right_index=True, how="outer"
    ).fillna(0)
    counts["lo_count"] = counts["lo_count_bid"] + counts["lo_count_ask"]
    counts["co_count"] = counts["co_count_bid"] + counts["co_count_ask"]

    # counts[["lo_count", "co_count"]].hist(bins=20, range=(0,20))
    # plt.show()

    lo_count = jnp.zeros(num_ticks, dtype=jnp.int32)
    lo_count = (
        lo_count
        .at[counts.lo_count.index.values - 1]
        .set(counts.lo_count.values.astype(jnp.int32))
    )

    co_count = jnp.zeros(num_ticks, dtype=jnp.int32)
    co_count = (
        co_count
        .at[counts.co_count.index.values - 1]
        .set(counts.co_count.values.astype(jnp.int32))
    )

    return lo_count, co_count


def _get_counts_by_side(
    book_data: pd.DataFrame,
    message_data: pd.DataFrame,
    tick_size: int,
    is_bid: bool,
) -> pd.Series:
    if is_bid:
        direction = 1
        p_ref_col_idx = 0
        quant_col_idx = 3
    else:
        direction = -1
        p_ref_col_idx = 2
        quant_col_idx = 1

    b_depth = book_data[message_data.direction == direction].copy()
    m_depth = message_data[message_data.direction == direction].copy()
    m_depth.price = (-direction) * (
        m_depth.price - b_depth.iloc[:, p_ref_col_idx]
    ) // tick_size

    b_depth = pd.concat([m_depth.price, b_depth.iloc[:, quant_col_idx::4]], axis=1)

    # limit orders
    lo_count = (
        m_depth
        .loc[m_depth.event_type == 1, "price"]
        .value_counts(sort=False)
        .sort_index()
    )
    lo_count.name = "lo_count_" + ("bid" if is_bid else "ask")
    # cancel orders
    co_count = (
        m_depth
        .loc[m_depth.event_type.isin((2, 3)), "price"]
        .value_counts(sort=False)
        .sort_index()
    )
    co_count.name = "co_count_" + ("bid" if is_bid else "ask")

    counts = pd.merge(
        lo_count, co_count, left_index=True, right_index=True, how="outer"
    ).fillna(0)

    return counts


if __name__ == "__main__":
    # estimate params for a single file
    # m_df = data_loading.load_message_df("data/GOOG_2012-06-21_34200000_57600000_message_10.csv")
    # b_df = data_loading.load_book_df("data/GOOG_2012-06-21_34200000_57600000_orderbook_10.csv")
    # print(m_df)
    # estimate_params(b_df, m_df, tick_size=100)

    # estimate on all files in a directory
    # estimate_from_data_files("data/")
    aggr_params = estimate_from_data_files("data/_data_dwn_32_210__AVXL_2021-01-01_2021-01-31_10")
    model_params, lo_lambda, co_theta = cst.init_params(aggr_params)
    print(model_params)
    print("lo_lambda", lo_lambda)
    print("co_theta", co_theta)
