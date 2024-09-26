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


def init_params(params_dict: dict[str, Any]) -> CSTParams:
    params = CSTParams(
        qty=jnp.round(params_dict["lo_size"]).astype(jnp.int32),
        tick_size=params_dict["tick_size"],
        num_ticks=params_dict["num_ticks"],
        lo_k=params_dict["lo_k"],
        lo_alpha=params_dict["lo_alpha"],
        mo_mu=params_dict["mo_mu"],
    )
    return params, params_dict["lo_lambda"], params_dict["co_theta"]


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
        return (book, base_rates, params, rng), (message, l2.flatten())

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
    # print('depth_i:', depth_i, 'qty:', qty, 'bid_side:', bid_side)

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
    # print("p_ref_idx", p_ref_idx, "p_ref_idx_new", p_ref_idx_new)
    p_delta = (p_ref_idx_new - p_ref_idx).astype(jnp.int32)
    # update reference price by moving it to the new best price
    p_ref = p_ref + p_delta * (-2*bid_side + 1) * params.tick_size
    # shift the opposite side of the book in accordance with the new reference price
    vols_opp = jnp.roll(vols_opp, p_delta)
    # print('p_delta:', p_delta)

    vols_opp = jax.lax.select(
        p_delta >= 0,
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
    vols = jax.lax.select(bid_side, book.bids, book.asks)
    vols_opp = jax.lax.select(bid_side, book.asks, book.bids)
    p_ref = jax.lax.select(bid_side, book.best_bid, book.best_ask)
    return vols, vols_opp, p_ref, p_ref_idx


if __name__ == "__main__":
    from param_estimation import load_params

    # with jax.disable_jit():

    aggr_params_dict = load_params(
        "data/_data_dwn_32_210__AVXL_2021-01-01_2021-01-31_10/aggregated_params.pkl"
    )
    params, lo_lambda, co_theta = init_params(aggr_params_dict)
    print("model_params", params)
    print("lo_lambda", lo_lambda)
    print("co_theta", co_theta)

    # plt.plot(jnp.arange(500), lo_lambda)
    # plt.plot(jnp.arange(500), co_theta)
    # plt.show()

    lobster_book = data_loading.load_book_df(
        "data/_data_dwn_32_210__AVXL_2021-01-01_2021-01-31_10/"
        "AVXL_2021-01-04_34200000_57600000_orderbook_10.csv"
    )

    # params = CSTParams()
    book = init_book(
        # jnp.array([100_0100, 1, 100_0000, 2, 101_1000, 3, 98_0000, 4, 0, 0, 0, 0]),
        lobster_book.iloc[0].values,
        params
    )
    print("INITIAL BOOK")
    # print(book)
    print(get_l2_book(book, params, 5))

    # co_theta = jnp.ones(params.num_ticks) * 0.1
    base_rates = get_event_base_rates(
        params,
        cancel_rates=co_theta,
        lo_rates=lo_lambda,
    )
    print("base_rates", base_rates)
    rng = jax.random.PRNGKey(0)

    # SCAN VERSION
    c, (msgs, l2_books) = jax.lax.scan(
        make_step_book_scannable(10),
        (book, base_rates, params, rng),
        length=100,
    )
    print(msgs)
    # print(l2_books)
    print(l2_book_to_pandas(l2_books))

    # FOR LOOP VERSION (no scan)
    # for i in range(100):
    #     book, message, rng = step_book(book, base_rates, params, rng)
    #     print(message)
    #     print("BOOK AFTER STEP", i)
    #     print(get_l2_book(book, params, 5))
    #     # print(book)
    #     print()
