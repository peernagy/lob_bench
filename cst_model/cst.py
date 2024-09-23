""" Cont Stoikov Talreja model of limit order book dynamics
"""

import jax
import jax.numpy as jnp
import chex


@chex.dataclass
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


def lo_rate(depth_i, k, alpha):
    return k / (depth_i ** alpha)


def get_event_base_rates(
    params: CSTParams,
    cancel_rates: jax.Array
) -> jax.Array:
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
    ask_p = book.best_bid + (ask_idx + 1) * params.tick_size
    ask_q = book.asks[ask_idx.astype(jnp.int32)].at[jnp.isnan(ask_idx)].set(0)

    bid_idx = jnp.argwhere(book.bids, size=n_lvls, fill_value=jnp.nan).squeeze()
    bid_p = book.best_ask - (bid_idx + 1) * params.tick_size
    bid_q = book.bids[bid_idx.astype(jnp.int32)].at[jnp.isnan(bid_idx)].set(0)

    l2 = jnp.stack([ask_p, ask_q, bid_p, bid_q], axis=1).astype(jnp.int32)
    return l2, book.time


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
    return _apply_event(book, tau, event, params), rng


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

    # TODO: remove prints
    # print('event_type:', event_type)
    if event_type == 0:
        print('LO_ASK')
    elif event_type == 1:
        print('LO_BID')
    elif event_type == 2:
        print('MO_BUY')
    elif event_type == 3:
        print('MO_SELL')
    elif event_type == 4:
        print('CO_ASK')
    elif event_type == 5:
        print('CO_BID')

    # depth starts at 0 for each new event type
    depth_i = (event - event_group_thresh[event_type])
    # all events above LOs remove liquidity (MO, CO) --> negative quantity
    qty = (params.qty * (-2 * (event_type >= 2) + 1)).squeeze()
    # odd event types pertain to the bid side
    bid_side = (event_type % 2).squeeze()
    print('depth_i:', depth_i, 'qty:', qty, 'bid_side:', bid_side)
    book = _change_vol(book, depth_i, qty, bid_side, params)
    # print('125', book)
    book.time += tau
    return book


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
        vols_opp.at[:p_delta].set(0),
        vols_opp.at[p_delta:].set(0),
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
    if bid_side:
        vols, vols_opp = book.bids, book.asks
        p_ref = book.best_bid
    else:
        vols, vols_opp = book.asks, book.bids
        p_ref = book.best_ask
    return vols, vols_opp, p_ref, p_ref_idx


if __name__ == "__main__":
    # with jax.disable_jit():

    params = CSTParams()
    book = init_book(
        jnp.array([100_0100, 1, 100_0000, 2, 101_1000, 3, 98_0000, 4, 0, 0, 0, 0]),
        params
    )
    print("INITIAL BOOK")
    print(book)
    print(get_l2_book(book, params, 5))

    co_theta = jnp.ones(params.num_ticks) * 0.1
    base_rates = get_event_base_rates(params, co_theta)
    print("base_rates", base_rates)

    rng = jax.random.PRNGKey(0)
    for i in range(100):
        book, rng = step_book(book, base_rates, params, rng)
        print("BOOK AFTER STEP", i)
        print(get_l2_book(book, params, 5))
        # print(book)
        print()
