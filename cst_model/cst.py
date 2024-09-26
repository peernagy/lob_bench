""" Cont Stoikov Talreja model of limit order book dynamics
"""

import pandas as pd
import jax
import jax.numpy as jnp
import chex
import matplotlib.pyplot as plt
from typing import Any
from scipy.optimize import curve_fit

# FIXME: this is a hack to import from parent dir
# add parent dir to path
import sys
sys.path.append("..")
from lob_bench import data_loading


@chex.dataclass(frozen=True)
class CSTParams:
    qty: int = 1
    tick_size: int = 100
    num_ticks: int = 500
    # limit order params
    lo_k: float = 1.0
    lo_alpha: float = 0.5
    # market order params
    mo_mu: float = 10.0


@chex.dataclass
class Book:
    best_ask: jax.Array
    best_bid: jax.Array
    asks: jax.Array
    bids: jax.Array
    time: jax.Array


@chex.dataclass
class Message:
    time: jax.Array
    event_type: jax.Array
    # oid: jax.Array
    size: jax.Array
    price: jax.Array
    direction: jax.Array


# def estimate_params(
#     book_data: pd.DataFrame,
#     message_data: pd.DataFrame,
#     tick_size: int = 100,
#     num_ticks: int = 500,
# ) -> dict[str, Any]:
#     # entire time duration
#     total_time = float(message_data.time.iloc[-1] - message_data.time.iloc[0])
#     # TODO: consider grouping executions at same time stamp?
#     # number of executions / market orders
#     mo_count = message_data[message_data.event_type == 4].shape[0]

#     # average size of limit orders
#     lo_size = message_data.loc[message_data.event_type == 1, "size"].mean()
#     # average size of market orders
#     mo_size = message_data.loc[message_data.event_type == 4, "size"].mean()
#     # average size of cancellations
#     co_size = message_data.loc[message_data.event_type.isin((2, 3)), "size"].mean()
#     print("Sl:", lo_size, "Sm:", mo_size, "Sc:", co_size)

#     mean_spread = ((book_data.iloc[:, 0] - book_data.iloc[:, 2]) / tick_size).mean()
#     print("Mean spread:", mean_spread)

#     # market order rate (sizes relative to limit orders)
#     mo_mu = mo_count / total_time * mo_size / lo_size

#     # TODO: calculate difference between message price and (opposite) best price
#     #       to get depth of message --> twice, per side of book
#     #       then, value_counts of depths to get distribution of depths
#     #       and divide by T to get rates for LOs and COs

#     lo_count, co_count = _get_counts_by_depth(book_data, message_data, tick_size, num_ticks)
#     print('lo_count', lo_count)
#     print('co_count', co_count)
#     i = jnp.arange(1, num_ticks+1)
#     lo_sol, pcov = curve_fit(lo_rate, i, lo_count)
#     print("LO fit", lo_sol)
#     plt.plot(i, lo_count, label="LO")
#     plt.plot(i, lo_rate(i, *lo_sol), label="LO fit")
#     plt.title("Limit Order Counts")
#     plt.xlabel("Depth")
#     plt.ylabel("Count")
#     plt.legend()
#     plt.show()

#     # convert all lobster book data to jax array of volumes at depths
#     book_jnp = jax.jit(jax.vmap(init_book, in_axes=(0, None)), static_argnums=(1,))(
#         jnp.array(book_data.values),
#         CSTParams(
#             tick_size=tick_size,
#             num_ticks=num_ticks,
#         )
#     )
#     # calculate Q_i from paper (mean volume at depth) averaged over bid and ask sides
#     mean_size_at_depth = (book_jnp.asks.mean(axis=0) + book_jnp.bids.mean(axis=0)) / 2
#     print("mean_size_at_depth", mean_size_at_depth)

#     co_theta = co_count / (mean_size_at_depth * total_time) * (co_size / lo_size)
#     co_theta = co_theta.at[jnp.isnan(co_theta)].set(0)
#     print("co_theta", co_theta)

#     params_dict = {
#         "total_time": total_time,
#         "num_events": message_data.shape[0],
#         "mean_spread": mean_spread,
#         "lo_size": lo_size,
#         "mo_size": mo_size,
#         "co_size": co_size,
#         "mo_count": mo_count,
#         "mo_mu": mo_mu,
#         "lo_counts": lo_count,
#         "co_counts": co_count,
#         "q_i": mean_size_at_depth,
#         "co_theta": co_theta,
#     }
#     return params_dict


# def _powerlaw(i, k, alpha):
#     return k * i ** (-alpha)


# def _get_counts_by_depth(
#     book_data: pd.DataFrame,
#     message_data: pd.DataFrame,
#     tick_size: int,
#     num_ticks: int,
# ) -> tuple[jax.Array, jax.Array]:
#     ask_counts = _get_counts_by_side(book_data, message_data, tick_size, is_bid=False)
#     bid_counts = _get_counts_by_side(book_data, message_data, tick_size, is_bid=True)
#     counts = pd.merge(
#         ask_counts, bid_counts, left_index=True, right_index=True, how="outer"
#     ).fillna(0)
#     counts["lo_count"] = counts["lo_count_bid"] + counts["lo_count_ask"]
#     counts["co_count"] = counts["co_count_bid"] + counts["co_count_ask"]

#     # counts[["lo_count", "co_count"]].hist(bins=20, range=(0,20))
#     # plt.show()

#     lo_count = jnp.zeros(num_ticks, dtype=jnp.int32)
#     lo_count = lo_count.at[counts.lo_count.index.values - 1].set(counts.lo_count.values)

#     co_count = jnp.zeros(num_ticks, dtype=jnp.int32)
#     co_count = co_count.at[counts.co_count.index.values - 1].set(counts.co_count.values)

#     return lo_count, co_count


# def _get_counts_by_side(
#     book_data: pd.DataFrame,
#     message_data: pd.DataFrame,
#     tick_size: int,
#     is_bid: bool,
# ) -> pd.Series:
#     if is_bid:
#         direction = 1
#         p_ref_col_idx = 0
#         quant_col_idx = 3
#     else:
#         direction = -1
#         p_ref_col_idx = 2
#         quant_col_idx = 1

#     b_depth = book_data[message_data.direction == direction].copy()
#     m_depth = message_data[message_data.direction == direction].copy()
#     m_depth.price = (-direction) * (
#         m_depth.price - b_depth.iloc[:, p_ref_col_idx]
#     ) // tick_size

#     b_depth = pd.concat([m_depth.price, b_depth.iloc[:, quant_col_idx::4]], axis=1)
#     print(b_depth)

#     # limit orders
#     lo_counts = (
#         m_depth
#         .loc[m_depth.event_type == 1, "price"]
#         .value_counts(sort=False)
#         .sort_index()
#     )
#     lo_counts.name = "lo_count_" + ("bid" if is_bid else "ask")
#     # cancel orders
#     co_counts = (
#         m_depth
#         .loc[m_depth.event_type.isin((2, 3)), "price"]
#         .value_counts(sort=False)
#         .sort_index()
#     )
#     co_counts.name = "co_count_" + ("bid" if is_bid else "ask")

#     counts = pd.merge(
#         lo_counts, co_counts, left_index=True, right_index=True, how="outer"
#     ).fillna(0)

#     return counts


def lo_rate(depth_i, k, alpha):
    return k / (depth_i ** alpha)


def get_event_base_rates(
    params: CSTParams,
    cancel_rates: jax.Array,
    lo_rates: jax.Array = None,
) -> jax.Array:
    # using powerlaw if no other empirical rates are provided
    if lo_rates is None:
        i_arr = jnp.arange(1, params.num_ticks+1)
        lo_rates = lo_rate(i_arr, params.lo_k, params.lo_alpha)
    rates = jnp.hstack([
        lo_rates, lo_rates,
        params.mo_mu, params.mo_mu,
        cancel_rates, cancel_rates
    ])
    return rates


def get_event_rates(base_rates: jax.Array, book: Book) -> jax.Array:
    n = book.asks.shape[0]
    rates = base_rates.copy()
    rates = rates.at[-2*n:-n].set(rates[-2*n:-n] * book.asks)
    rates = rates.at[-n:].set(rates[-n:] * book.bids)
    return rates


def sample_tau(rates: jax.Array, rng: jax.Array) -> jax.Array:
    r0 = jnp.sum(rates)
    tau = 1/r0 * jnp.log(1/jax.random.uniform(rng))
    return tau


def sample_event(rates: jax.Array, rng: jax.Array) -> int:
    rates_cum = jnp.cumsum(rates)
    r = jax.random.uniform(rng) * rates_cum[-1]
    event = jnp.argwhere(rates_cum > r, size=1)[0,0]
    return event


def init_book(l2_book_lobster: jax.Array, params: CSTParams) -> Book:
    l2_book_lobster = l2_book_lobster.reshape(-1, 4)
    best_ask = l2_book_lobster[0, 0]
    best_bid = l2_book_lobster[0, 2]
    # normalize prices relative to opposite best price
    ask_p_idx = ((l2_book_lobster[:, 0] - best_bid) // params.tick_size).astype(jnp.int32) - 1
    bid_p_idx = ((l2_book_lobster[:, 2] - best_ask) // (-params.tick_size)).astype(jnp.int32) - 1

    asks = jnp.zeros(params.num_ticks)
    bids = jnp.zeros(params.num_ticks)
    asks = asks.at[ask_p_idx].set(l2_book_lobster[:, 1])
    bids = bids.at[bid_p_idx].set(l2_book_lobster[:, 3])

    return Book(
        best_ask=best_ask,
        best_bid=best_bid,
        asks=asks,
        bids=bids,
        time=0.
    )


def get_l2_book(book: Book, params: CSTParams, n_lvls: int) -> jax.Array:
    ask_idx = jnp.argwhere(book.asks, size=n_lvls, fill_value=jnp.nan).squeeze()
    ask_p = (book.best_bid + (ask_idx + 1) * params.tick_size).astype(jnp.int32)
    ask_p = jnp.where(ask_p == 0, 999999999, ask_p)
    ask_q = jnp.where(jnp.isnan(ask_idx), 0, book.asks[ask_idx.astype(jnp.int32)]).astype(jnp.int32)

    bid_idx = jnp.argwhere(book.bids, size=n_lvls, fill_value=jnp.nan).squeeze()
    bid_p = (book.best_ask - (bid_idx + 1) * params.tick_size).astype(jnp.int32)
    bid_p = jnp.where(bid_p == 0, -999999999, bid_p)
    bid_q = jnp.where(jnp.isnan(bid_idx), 0, book.bids[bid_idx.astype(jnp.int32)]).astype(jnp.int32)

    l2 = jnp.stack([ask_p, ask_q, bid_p, bid_q], axis=1)
    return l2, book.time


def l2_book_to_pandas(l2_book: jax.Array) -> pd.DataFrame:
    columns = [
        s for l in range(1, l2_book.shape[1] // 4 + 1)
            for s in (f"ask_p_{l}", f"ask_q_{l}", f"bid_p_{l}", f"bid_q_{l}")
    ]
    return pd.DataFrame(l2_book, columns=columns)


def step_book(
    book: Book,
    base_rates: jax.Array,
    params: CSTParams,
    rng: jax.Array,
) -> Book:
    rng, rng_tau, rng_event = jax.random.split(rng, 3)
    rates = get_event_rates(base_rates, book)
    tau = sample_tau(rates, rng_tau)
    event = sample_event(rates, rng_event)
    print("EVENT:", event, "TAU:", tau)
    book, message = _apply_event(book, tau, event, params)
    return book, message, rng


def make_step_book_scannable(n_lvls: int):
    def _step_book_scannable(carry, _):
        book, base_rates, params, rng = carry
        book, message, rng = step_book(book, base_rates, params, rng)
        l2, time = get_l2_book(book, params, n_lvls)
        return (book, base_rates, params, rng), (message, l2)

    return _step_book_scannable


def _apply_event(
    book: Book,
    tau: jax.Array,
    event: int,
    params: CSTParams
) -> Book:
    with jax.ensure_compile_time_eval():
        # indices where event types change
        event_group_thresh = jnp.array([
            0,
            params.num_ticks, params.num_ticks,
            1, 1,
            params.num_ticks, params.num_ticks,
        ]).cumsum()

    event_type = (jnp.argwhere(event_group_thresh > event, size=1)[0] - 1).squeeze()

    depth_i = jax.lax.select(
        # market orders need to find correct depth
        event_type // 2 == 1,
        # find current best price level index
        (book.best_ask - book.best_bid) // params.tick_size - 1,
        # depth starts at 0 for each new event type
        (event - event_group_thresh[event_type])
    )

    # all events above LOs remove liquidity (MO, CO) --> negative quantity
    qty = (params.qty * (-2 * (event_type >= 2) + 1)).squeeze()
    # odd event types pertain to the bid side
    bid_side = (event_type % 2).squeeze()
    print('depth_i:', depth_i, 'qty:', qty, 'bid_side:', bid_side)

    book = _change_vol(book, depth_i, qty, bid_side, params)
    book.time += tau

    message = _get_message(
        book.time,
        event_type,
        depth_i,
        book.best_ask,
        book.best_bid,
        qty,
        bid_side,
        params
    )
    return book, message


def _get_message(
    time: int,
    event_type: int,
    depth_i: int,
    best_ask: int,
    best_bid: int,
    qty: int,
    bid_side: bool,
    params: CSTParams,
):
    # event type: 0,1 -> type1, 2,3 -> type4, 4,5 -> type3
    lobster_type = jax.lax.switch(
        event_type // 2,
        (lambda : 1,
        lambda : 4,
        lambda : 3),
    )
    # price
    price = jax.lax.select(
        bid_side,
        best_ask - (depth_i + 1) * params.tick_size,
        best_bid + (depth_i + 1) * params.tick_size,
    )
    # scale from {0,1} to {-1,1}
    direction = bid_side * 2 - 1
    return Message(
        time=time,
        event_type=lobster_type,
        size=qty,
        price=price,
        direction=direction
    )


def _change_vol(
    book: Book,
    depth_i: int,
    qty: int,
    bid_side: bool,
    params: CSTParams,
) -> Book:
    vols, vols_opp, p_ref, p_ref_idx = _get_book_sides(book, bid_side, params)
    # change volume at depth_i
    vols = vols.at[depth_i].set(jnp.maximum(0, vols[depth_i] + qty))
    p_ref_idx_new = jnp.argwhere(vols > 0, size=1)[0, 0]
    print("p_ref_idx", p_ref_idx, "p_ref_idx_new", p_ref_idx_new)
    p_delta = (p_ref_idx_new - p_ref_idx).astype(jnp.int32)
    # update reference price by moving it to the new best price
    p_ref = p_ref + p_delta * (-2*bid_side + 1) * params.tick_size
    # shift the opposite side of the book in accordance with the new reference price
    vols_opp = jnp.roll(vols_opp, p_delta)
    print('p_delta:', p_delta)

    vols_opp = jax.lax.select(
        p_delta >= 0,
        # vols_opp.at[:p_delta].set(0),
        # vols_opp.at[p_delta:].set(0),
        # jax.lax.dynamic_update_slice(vols_opp, jnp.zeros(p_delta), 0),
        # jax.lax.dynamic_update_slice(vols_opp, jnp.zeros(p_delta), -p_delta),
        jnp.where(jnp.arange(vols_opp.shape[0]) < p_delta, 0, vols_opp),
        jnp.where(jnp.arange(vols_opp.shape[0]) >= vols_opp.shape[0] + p_delta, 0, vols_opp),
    )

    asks = jax.lax.select(bid_side, vols_opp, vols)
    bids = jax.lax.select(bid_side, vols, vols_opp)
    best_ask = jax.lax.select(bid_side, book.best_ask, p_ref)
    best_bid = jax.lax.select(bid_side, p_ref, book.best_bid)
    return Book(
        best_ask=best_ask,
        best_bid=best_bid,
        asks=asks,
        bids=bids,
        time=book.time
    )


def _get_book_sides(
    book: Book,
    bid_side: bool,
    params: CSTParams,
) -> tuple[jax.Array, jax.Array, float, int]:
    # idx position of current reference price
    # e.g. for bid change, where best_bid is in the array
    p_ref_idx = (book.best_ask - book.best_bid) // params.tick_size - 1
    # if bid_side:
    #     vols, vols_opp = book.bids, book.asks
    #     p_ref = book.best_bid
    # else:
    #     vols, vols_opp = book.asks, book.bids
    #     p_ref = book.best_ask
    vols = jax.lax.select(bid_side, book.bids, book.asks)
    vols_opp = jax.lax.select(bid_side, book.asks, book.bids)
    p_ref = jax.lax.select(bid_side, book.best_bid, book.best_ask)
    return vols, vols_opp, p_ref, p_ref_idx


if __name__ == "__main__":
    # with jax.disable_jit():

    params = CSTParams()
    # book = init_book(
    #     jnp.array([100_0100, 1, 100_0000, 2, 101_1000, 3, 98_0000, 4, 0, 0, 0, 0]),
    #     params
    # )
    # print("INITIAL BOOK")
    # print(book)
    # print(get_l2_book(book, params, 5))

    # co_theta = jnp.ones(params.num_ticks) * 0.1
    # base_rates = get_event_base_rates(params, co_theta)
    # print("base_rates", base_rates)

    # rng = jax.random.PRNGKey(0)

    # # SCAN VERSION
    # c, (msgs, l2_books) = jax.lax.scan(
    #     make_step_book_scannable(5),
    #     (book, base_rates, params, rng),
    #     length=100,
    # )
    # print(msgs)
    # print(l2_books)

    # FOR LOOP VERSION
    # for i in range(100):
    #     book, message, rng = step_book(book, base_rates, params, rng)
    #     print(message)
    #     print("BOOK AFTER STEP", i)
    #     print(get_l2_book(book, params, 5))
    #     # print(book)
    #     print()

    m_df = data_loading.load_message_df("data/GOOG_2012-06-21_34200000_57600000_message_10.csv")
    b_df = data_loading.load_book_df("data/GOOG_2012-06-21_34200000_57600000_orderbook_10.csv")
    print(m_df)
    estimate_params(b_df, m_df, tick_size=100)
