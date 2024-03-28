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

    dt_start = pd.to_datetime('1970-01-01')
    # origin arg in resample only works properly with DateTimeIndex
    series.index = series.index + dt_start
    means = series.resample(
        period, label='right', closed='left', origin='start_day'
    ).mean()
    series.index = series.index - dt_start
    return means

def mid_price(
        messages: pd.DataFrame,
        book: pd.DataFrame
    ) -> pd.Series:
    """
    Calculate mid-price of book.
    """
    mid = (book.iloc[:, 0] + book.iloc[:, 2]) / 2
    mid.index = pd.to_timedelta(messages.time.astype(float), unit='s')
    return mid

def spread(
        messages: pd.DataFrame,
        book: pd.DataFrame
    ) -> pd.Series:
    """
    Calculate spread of book.
    """
    spread = book.iloc[:, 0] - book.iloc[:, 2]
    spread.index = pd.to_timedelta(messages.time.astype(float), unit='s')
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
    mid.index = pd.to_datetime(messages.time.astype(float), unit='s')
    mid = mid.resample(
        interval, label='right', closed='left', origin='start_day'
    ).last()
    # convert datetime to timedelta since midnight
    mid.index = pd.to_timedelta(mid.index - mid.index[0].normalize())

    # mid = mid.pct_change()
    ln_p = np.log(mid)
    ret = ln_p - ln_p.shift(1)

    return ret

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
    ).dropna().astype(float)
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
    ).dropna().astype(float)
    return del_t

def total_volume(messages: pd.DataFrame, book: pd.DataFrame, n_levels: int) -> pd.Series:
    """
    Calculate aggregated bid and ask volume on first n levels of book.
    Returns a single dataframe with summed ask and bid volume of first n_levels,
    and a "time weight", to weigh observations by the amount of time present in the data.
    """
    assert n_levels > 0, "Number of levels must be positive."
    assert n_levels <= book.shape[1] // 4, "Number of levels exceeds book depth."

    ask_vol = book.iloc[:, 1: 2*n_levels: 2].sum(axis=1)
    bid_vol = book.iloc[:, 0: 2*n_levels: 2].sum(axis=1)
    df = pd.concat([ask_vol, bid_vol], axis=1)
    df.columns = ['ask_vol', 'bid_vol']
    df['time_weight'] = messages.time.diff() / (messages.time.iloc[-1] - messages.time.iloc[0])
    df.index = pd.to_timedelta(messages.time.astype(float), unit='s')

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
    df.index = pd.to_timedelta(messages.time.astype(float), unit='s')
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
    depth.index = pd.to_timedelta(messages.time.astype(float), unit='s')
    ask_depth = depth[depth > 0]
    bid_depth = depth[depth < 0]
    return ask_depth, bid_depth.abs()

def limit_order_depth(messages: pd.DataFrame, book: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Calculate depth (level) of new limit order events.
    Returns ask and bid depth Series.
    """
    return _order_depth(messages, book, (1,))

def cancellation_depth(messages: pd.DataFrame, book: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Calculate depth of limit order cancellation or modifications.
    """
    return _order_depth(messages, book, (2, 3))

def _order_levels(
        messages: pd.DataFrame,
        book: pd.DataFrame,
        event_types: tuple[int],
    ) -> tuple[pd.Series, pd.Series]:
    """
    Get levels of given order types.
    Returns two Series: ask and bid levels.
    """
    order_prices = messages[messages.event_type.isin(event_types)]
    level_prices = book.loc[order_prices.index, 0::2]
    order_prices.index = pd.to_timedelta(order_prices.time.astype(float), unit='s')
    order_prices = order_prices.price
    lvl_idx = np.argwhere((order_prices.values == level_prices.values.T).T)
    assert lvl_idx.shape[0] == order_prices.shape[0], "Not all order prices found in book."
    # lvl_idx = pd.Series(lvl_idx[:, 0], index=lvl_idx[:, 1])
    # print('lvl_idx', lvl_idx)
    lvl_idx = pd.Series(lvl_idx[:, 1], index=order_prices.index)
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
    return _order_levels(messages, book, (1,))

def cancel_order_levels(messages: pd.DataFrame, book: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Get levels of limit order cancellations and modifications.
    Returns ask and bid levels Series.
    """
    return _order_levels(messages, book, (2, 3))
