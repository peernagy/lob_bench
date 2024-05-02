"""
Functions to estimate market impact of different events produced by the model, and compared to real data behaviour.
"""
from typing import Optional
import numpy as np
import pandas as pd


def get_impact_message(
        price: int,
        is_buy_message: bool,
        size: int,
        t_prev: np.datetime64,
        t_next: Optional[float] = None,
    ) -> np.array:
    """
    Returns an LOBSTER 'impact' message (limit order), which should be 
    appended to the input or conditioning sequence of the generative model.
    The time stamp is uniformly sampled between t_prev (incl.) and t_next (excl.).
    If t_next is None, the same time as t_prev is used.
    """
    pass

def get_cleanup_message(
        impact_message: np.array,
        wait_messages: pd.DataFrame,
    ) -> Optional[np.array]:
    """
    Returns a limit order to 'clean up' remaining unexecuted volume of a 
    passive impact message. If the impact message was fully executed, None
    is returned.

    :param impact_message: The impact message to clean up.
    :param wait_messages: All messages in the "wait period" between the impact
                          message and the cleanup message.
    :return: A cleanup message or None.
    """
    pass

# TODO: Bouchaud Aggregator Model