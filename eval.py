"""
Evaluate realism of generated LOB data (in LOBSTER format).

------------------------------------------------------------
LOBSTER FORMAT

Message fields:

Time (sec), Event Type, Order ID, Size, Price, Direction
------------------------------------------------------------

All evaluation should be based on providing
    (1) the (unique) real message sequences and
    (2) n samples of generated message sequences.
Optionally, the LOB state at each time point can be provided.
If this is not provided, the JAX-LOB simulator is used to generate the L2 states of the book.

* Distributional evaluation (single message forward / teacher forcing):
    - cross-entropy of generated messages.
    - use tokenization scheme to encode messages?
    X issues: calculated using sampled data --> could have 0 empirical probability if not sampled.

    - Divergence of LOB state over time --> only condition once and generate as long as possible / for whole day

* Trajectory-level evaluation
    - Discrimator to distinguish real from generated data

* "Stylized facts" & matching data characteristics:
    x how to define conditioning set? If we re-condition on real data after a few seconds, distributions will match real data more closely because they are not fully generated.
      --> check if we can generate full day in principle
    x should we do this all jaxified or e.g. with pandas?
      ---> could e.g. be vmapped over different periods? problem: time windows have heterogeneous lengths
      ---> use pandas for most of this. Only use jax if can make good use of parallelization and jitting with fixed sizes.

    - time to first fill for new limit orders (ignore cancelled orders)
    - time to cancel / modification (ignore filled orders)
    - per time of day (define periods, e.g. 5 minute windows):
        NOTE: use pandas for this. use time periods contained in data only --> can call multiple times for different periods if needed.
        NOTE: calc. for every new state; for distributional / hist measures, weight observations by time since last observation.
        . ask and bid volume on first level
        . aggregated bid and ask volume (NOTE: need to specify number of levels)
        . depth of limit orders (--> filter to only limit orders) (NOTE: no need to weight by time since last observation)
        . cancellation depths (NOTE: no need to weight by time since last observation)
        . spread
        . mid-price or return distribution (NOTE: either moments or distributional distance metric)
    - autocorrelation of returns (NOTE: use pandas to calculate m-minute returns, e.g. m=1)
        . 1-lag 1-minute ac coefficient
        . 1-10 lag r2 acf

* Conditional predictive performance:
    - Correlation of predicted and realized mid-price returns.

* Price impact measures
    - Micro impact of individual orders can be compared to real data

* Evaluation on Downstream Tasks --> how well do learned policies generalize to other models?

"""

from datetime import timedelta
from typing import Optional, Union
import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset
import statsmodels.api as sm


def mean_per_interval(
        series: pd.Series,
        period: Union[DateOffset, timedelta, str] = '5Min'
    ):
    """
    Calculate mean of series in given interval.
    """

    # dt_start = pd.to_datetime('1970-01-01')
    # origin arg in resample only works properly with DateTimeIndex
    # series.index = series.index + dt_start
    means = series.resample(
        period, label='right', closed='left', origin='start_day'
    ).mean()
    # series.index = series.index - dt_start
    return means

def mid_price(
        messages: pd.DataFrame,
        book: pd.DataFrame
    ) -> pd.Series:
    """
    Calculate mid-price of book.
    """
    mid = (book.iloc[:, 0] + book.iloc[:, 2]) / 2
    mid.index = messages.time
    return mid

def spread(
        messages: pd.DataFrame,
        book: pd.DataFrame
    ) -> pd.Series:
    """
    Calculate spread of book.
    """
    spread = book.iloc[:, 0] - book.iloc[:, 2]
    spread.index = messages.time
    spread.name = 'spread'
    return spread

def mid_returns(
        messages: pd.DataFrame,
        book: pd.DataFrame,
        interval: Union[DateOffset, timedelta, str] = '1min'
    ) -> pd.Series:
    """
    Calculate mid-price log returns in given interval.
    NOTE: mid-prices are resampled with the last value of the interval to not look ahead.
    """
    mid = mid_price(messages, book)
    mid.index = messages.time
    mid = mid.resample(
        interval, label='right', closed='left', origin='start_day'
    ).last()

    # mid = mid.pct_change()
    ln_p = np.log(mid)
    ret = ln_p - ln_p.shift(1)
    ret.name = 'mid_returns_' + interval

    return ret

def volatility(
        messages: pd.DataFrame,
        book: pd.DataFrame,
        interval: Union[DateOffset, timedelta, str] = '1min',
    ) -> pd.Series:
    """
    """
    r = mid_returns(messages, book, interval)
    std = r.std()
    if np.isnan(std):
        return 0.
    return std

def autocorr(
        returns: pd.Series,
        lags: int = 1,
        alpha: Optional[float] = None,
    ) -> np.array:
    """
    """
    # return returns.autocorr(lags)
    return sm.tsa.acf(
        returns.dropna(),
        nlags=lags,
        fft=True,
        alpha=alpha
    )

def time_of_day(messages: pd.DataFrame) -> pd.Series:
    """
    Get time of day in seconds.
    """
    sod = messages.time.iloc[0].replace(hour=0, minute=0, second=0, microsecond=0)
    return (messages.time - sod).dt.total_seconds()

def start_date_time(messages: pd.DataFrame) -> pd.Timestamp:
    """
    Get start time of data.
    """
    return messages.time.iloc[0]

def start_time(messages: pd.DataFrame) -> pd.Timestamp:
    """
    Get start time of data.
    """
    dt = start_date_time(messages)
    return (dt - dt.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()

def inter_arrival_time(messages: pd.DataFrame) -> pd.Series:
    """
    Calculate inter-arrival time of messages in milliseconds.
    """
    return messages.time.diff().iloc[1:].dt.total_seconds() * 1000

def time_to_first_fill(messages: pd.DataFrame) -> pd.Series:
    """
    Calculate time to first fill for new limit orders.

    CAVE: only orders which are executed are considered. Open orders and cancellations are ignored.
    Hence, the actual time to fill is underestimated.
    """
    # filter for new orders and executions
    orders = messages[(messages.event_type == 1) | (messages.event_type == 4)]

    # filter order to only include messages related to new orders
    new_ids = orders[orders.event_type == 1].order_id.unique()
    orders = orders[orders.order_id.isin(new_ids)]

    del_t = orders.groupby('order_id').apply(
        # take new order (type 1) and first execution (type 4) if it exists
        # and calculate time difference
        lambda x: x.iloc[:2].time.diff().iloc[-1],
        include_groups = False
    # get rid of nans from orders that were not executed
    ).dropna()
    return del_t

def time_to_cancel(messages: pd.DataFrame) -> pd.Series:
    """
    Calculate time to cancel or first modification for limit orders.

    CAVE: only orders which are cancelled are considered. Open orders and executions are ignored.
    """
    # filter for new orders, cancellations, modifications
    orders = messages[
        (messages.event_type == 1) |
        (messages.event_type == 2) |
        (messages.event_type == 3)
    ]

    # filter order to only include messages related to new orders
    new_ids = orders[orders.event_type == 1].order_id.unique()
    orders = orders[orders.order_id.isin(new_ids)]

    del_t = orders.groupby('order_id').apply(
        # take new order (type 1) and first cancellation (type 2 or 3) if it exists
        # and calculate time difference
        lambda x: x.iloc[:2].time.diff().iloc[-1],
        include_groups=False
    # get rid of nans from orders that were not cancelled
    ).dropna()
    # always return a Series with a timedelta type
    if len(del_t) == 0:
        return pd.to_timedelta(pd.Series())
    return del_t

def total_volume(messages: pd.DataFrame, book: pd.DataFrame, n_levels: int) -> pd.Series:
    """
    Calculate aggregated bid and ask volume on first n levels of book.
    Returns a single dataframe with summed ask and bid volume of first n_levels,
    and a "time weight", to weigh observations by the amount of time present in the data.
    """
    assert n_levels > 0, "Number of levels must be positive."
    assert n_levels <= book.shape[1] // 4, "Number of levels exceeds book depth."

    ask_vol = book.iloc[:, 1: 4*n_levels: 4].sum(axis=1)
    bid_vol = book.iloc[:, 3: 4*n_levels: 4].sum(axis=1)
    df = pd.concat([ask_vol, bid_vol], axis=1)
    df.columns = ['ask_vol_' + str(n_levels), 'bid_vol_' + str(n_levels)]
    df['time_weight'] = messages.time.diff() / (messages.time.iloc[-1] - messages.time.iloc[0])
    df.index = messages.time

    return df

def l1_volume(messages: pd.DataFrame, book: pd.DataFrame) -> pd.Series:
    """
    Calculate ask and bid volume on first level of book.
    Returns a single dataframe with best ask and best bid volume,
    and a "time weight", to weigh observations by the amount of time present in the data.
    """
    df = book.iloc[:, [1, 3]].copy()
    df.columns = ['ask_vol', 'bid_vol']
    df['time_weight'] = messages.time.diff() / (messages.time.iloc[-1] - messages.time.iloc[0])
    df.index = messages.time
    return df

def _order_depth(
        messages: pd.DataFrame,
        book: pd.DataFrame,
        event_types: tuple[int],
    ) -> tuple[pd.Series, pd.Series]:
    """
    Calculate depth of given order types. Depth is defined as the absolute distance to the mid-price.
    Returns two series: ask and bid depth.
    """
    order_prices = messages[messages.event_type.isin(event_types)].price
    mid_prices = (book.iloc[:, 0] + book.iloc[:, 2]) / 2

    depth = order_prices - mid_prices
    depth.index = messages.time
    ask_depth = depth[depth > 0]
    bid_depth = depth[depth < 0]
    return ask_depth, bid_depth.abs()

def limit_order_depth(messages: pd.DataFrame, book: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Calculate depth (level) of new limit order events.
    Returns ask and bid depth Series.
    """
    ask_depth, bid_depth = _order_depth(messages, book, (1,))
    ask_depth.name = 'ask_limit_depth'
    bid_depth.name = 'bid_limit_depth'
    return ask_depth, bid_depth

def cancellation_depth(messages: pd.DataFrame, book: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Calculate depth of limit order cancellation or modifications.
    """
    ask_depth, bid_depth =  _order_depth(messages, book, (2, 3))
    ask_depth.name = 'ask_cancel_depth'
    bid_depth.name = 'bid_cancel_depth'
    return ask_depth, bid_depth

def _order_levels(
        messages: pd.DataFrame,
        book: pd.DataFrame,
        event_types: tuple[int],
        verify_orders: bool = False,
    ) -> tuple[pd.Series, pd.Series]:
    """
    Get levels of given order types.
    Returns two Series: ask and bid levels.
    """
    if (2 in event_types) or (3 in event_types):
        assert not 1 in event_types, \
            "Order levels for cancellations and modifications refer to the previous book state and are hence not compatible with new orders."
        # look at previous book state for cancellations and modifications
        book = book.shift(1).iloc[1:]
        messages = messages.iloc[1:]

    order_prices = messages[messages.event_type.isin(event_types)]
    level_prices = book.loc[order_prices.index, 0::2]
    order_prices.index = order_prices.time
    order_prices = order_prices.price
    # print('order_prices', order_prices.shape, order_prices)
    # print('level_prices', level_prices.shape, level_prices)
    lvl_idx = np.argwhere((order_prices.values == level_prices.values.T).T)
    if verify_orders:
        assert lvl_idx.shape[0] == order_prices.shape[0], f"Not all order prices found in book. ({lvl_idx.shape[0]} != {order_prices.shape[0]})"
    # lvl_idx = pd.Series(lvl_idx[:, 0], index=lvl_idx[:, 1])
    # print('lvl_idx', lvl_idx)
    # lvl_idx = pd.Series(lvl_idx[:, 1], index=order_prices.index)
    lvl_idx = pd.Series(lvl_idx[:, 1], index=lvl_idx[:, 0])
    # display(lvl_idx)
    # print('index', order_prices.index)

    bid_lvl_mask = (lvl_idx % 2 == 1)
    bid_levels = lvl_idx.loc[bid_lvl_mask]
    bid_levels = (bid_levels + 1) // 2
    ask_levels = lvl_idx.loc[~bid_lvl_mask]
    ask_levels = ask_levels // 2 + 1

    return ask_levels, bid_levels

def limit_order_levels(messages: pd.DataFrame, book: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Get levels of new limit orders.
    Returns ask and bid levels Series.
    """
    ask_levels, bid_levels = _order_levels(messages, book, (1,))
    ask_levels.name = 'ask_limit_level'
    bid_levels.name = 'bid_limit_level'
    return ask_levels, bid_levels

def cancel_order_levels(messages: pd.DataFrame, book: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Get levels of limit order cancellations and modifications.
    Returns ask and bid levels Series.
    """
    ask_levels, bid_levels = _order_levels(messages, book, (2, 3))
    ask_levels.name = 'ask_cancel_level'
    bid_levels.name = 'bid_cancel_level'
    return ask_levels, bid_levels

def orderbook_imbalance(messages: pd.DataFrame, book: pd.DataFrame) -> pd.Series:
    """
    Calculate orderbook imbalance on the first level of the book (best prices).
    """
    ask_vol = book.iloc[:, 1]
    bid_vol = book.iloc[:, 3]
    imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol)
    imbalance.index = messages.time
    return imbalance

def volume_per_minute(messages: pd.DataFrame, book: pd.DataFrame) -> pd.Series:
    """
    Calculate executed order size per minute by summing volume in
    1-second intervals and multiplying by 60.
    This is calculated based on execution messages (event_type == 4),
    irrespective of changes in the book.
    Edge periods are dropped if they are shorter than 0.1s,
    otherwise the volume is scaled proportionally to a full second.
    """

    min_period_microsec = 100_000

     # Filter for executed orders
    trades = messages[messages.event_type == 4].copy()

    # Group by 1-second intervals
    trades['second'] = trades['time'].dt.floor('1s')
    vol_per_second = trades.groupby('second')['size'].sum().astype(float)

    if not vol_per_second.empty:
        first, last = vol_per_second.index[0], vol_per_second.index[-1]
        first_start = trades.loc[trades['second'] == first, 'time'].iloc[0]
        last_end = trades.loc[trades['second'] == last, 'time'].iloc[-1]
        # drop period if < 0.1s

        # single second period
        if len(vol_per_second) == 1:
            # multiple trades in a single second
            if first_start != last_end:
                period = (last_end.microsecond - first_start.microsecond)
                if period < min_period_microsec:
                    vol_per_second = vol_per_second.iloc[0:0]
                else:
                    vol_per_second.iloc[0] *= 1_000_000 / period

        # multiple second periods
        else:
            # first second period is too short
            if first_start.microsecond < (1_000_000 - min_period_microsec):
                # drop first period
                vol_per_second = vol_per_second.iloc[1:]
            else:
                vol_per_second.iloc[0] /= 1 - (first_start.microsecond / 1_000_000)

            # last second period is too short
            if last_end.microsecond < min_period_microsec:
                # drop last period
                vol_per_second = vol_per_second.iloc[:-1]
            else:
                vol_per_second.iloc[-1] *= 1_000_000 / last_end.microsecond

    # Convert to per-minute volume
    return vol_per_second.multiply(60)

def orderflow_imbalance(messages: pd.DataFrame, book: pd.DataFrame, n_window=100) -> pd.Series:
    """
    Calculate orderflow imbalance.
    """
    lvl1 = book.iloc[:, :4]
    imb = pd.DataFrame(columns=['ask_delta', 'bid_delta', 'imbalance'])
    imb["ask_delta"] = lvl1.iloc[:, 0].diff().values[1:]
    imb["bid_delta"] = lvl1.iloc[:, 2].diff().values[1:]
    # +(p_bid(t) >= p_bid(t-1)) * q_bid(t) - (p_bid(t) <= p_bid(t-1)) * q_bid(t-1)
    # -(p_ask(t) <= p_ask(t-1)) * q_ask(t) + (p_ask(t) >= p_ask(t-1)) * q_ask(t-1)
    imb["imbalance"] = (
        (imb.bid_delta >= 0) * lvl1.values[1:, 3]
        - (imb.bid_delta <= 0) * lvl1.values[:-1, 3]
        - (imb.ask_delta <= 0) * lvl1.values[1:, 1]
        + (imb.ask_delta >= 0) * lvl1.values[:-1, 1]
    )
    # rolling average over n_window
    return imb["imbalance"].rolling(n_window).mean().iloc[n_window-1:]

def orderflow_imbalance_cond_tick(
    messages: pd.DataFrame,
    book: pd.DataFrame,
    tick_sign: int,
    n_window = 100,
) -> pd.Series:
    assert tick_sign in {-1, 0, 1}, "tick_sign must be -1, 0, or 1."
    # imb length: seq_len - n_window
    imb = orderflow_imbalance(messages, book, n_window)
    mid_price_diff = mid_price(messages, book).diff().iloc[1:]
    df = pd.DataFrame({
        "imb_prev": imb.values[:-1],
        "mid_price_diff": mid_price_diff.values[n_window:],
        # "direction": np.sign(mid_price_diff.values),
    })
    df["direction"] = np.sign(df["mid_price_diff"])
    return df.loc[df["direction"] == tick_sign, "imb_prev"]

def compute_3d_book_changes(messages: pd.DataFrame, book: pd.DataFrame) -> pd.Series:
    """
    Compute changes in book state with every message, expressed as (change in mid-price, change at which relative price, change in volume).
    """
    mid = mid_price(messages, book)

    # calculate changes in book state
    mid_diff = mid.diff()
    mid_diff.name = 'mid_change'
    # display(mid_diff)
    # display(messages.price)

    messages = messages.copy()
    messages.index = mid.index
    messages.price = messages.price - mid
    # most message types remove volume from the book...
    messages['size'] *= -1
    # except for new orders, which add volume:
    messages.loc[messages.event_type == 1, 'size'] *= -1

    return pd.concat([mid_diff, messages.price, messages['size']], axis=1)

def compute_3d_book_groups(
        book_3d: pd.DataFrame,
        n_bins_per_dim: int,
    ) -> pd.DataFrame:
    """
    """
    book_3d_groups = pd.DataFrame()
    book_3d_groups['mid_change'], mid_change_bins = pd.qcut(book_3d['mid_change'], n_bins_per_dim, duplicates='drop', retbins=True)
    book_3d_groups['price'], price_bins = pd.qcut(book_3d['price'], n_bins_per_dim, duplicates='drop', retbins=True)
    book_3d_groups['size'], size_bins = pd.qcut(book_3d['size'], n_bins_per_dim, duplicates='drop', retbins=True)
    return book_3d_groups, [mid_change_bins, price_bins, size_bins]
