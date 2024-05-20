"""
Functions to estimate market impact of different events produced by the model, and compared to real data behaviour.
"""
from typing import Optional
import numpy as np
import pandas as pd
from decimal import Decimal
from typing import Tuple

def filter_sequence_for_impact(message_sequence : pd.DataFrame,
                                book_sequence: pd.DataFrame,
                                ordersize: Tuple[int,int],
                                orderside: int,
                                timestamp :Tuple[str,str] = None,
                                sequence_length:int=100,):
    """Given an input sequence of LOBSTER messages and L2 orderbook data, of undetermined length. 
    Aims to find specific impact events, as defined by: order quantity, order side, limit or execution, potentially time.

            Parameters:
            message_sequence : Messages from a single day in a DF
            book_sequence: L2 Book states from a single day in a DF
            ordersize: Tuple with min and max order size (quant) to consider for impact orders (min,max)
            orderside: int,
            timestamps :Optional Tuple with min and max time to consider for impact orders (min,max)
            timestamp_below: str = None,
            sequence_length:int=100,
                
        Returns:
                Messages that follow an impact event. 
                Book states that follow an impact event. 
    
    """
    if timestamp is None:
        impact_msgs=message_sequence.loc[(message_sequence['size']>ordersize[0])
                                         &(message_sequence['size']<ordersize[1])
                                         &(message_sequence['direction']==orderside)]
    else:
        message_sequence['time']=message_sequence['time'].map(Decimal)
        print((message_sequence['time']<=Decimal(timestamp[1]))
                & (message_sequence['time']>=Decimal(timestamp[0])))
        impact_msgs=message_sequence.loc[(message_sequence['size']>=ordersize[0])
                                        &(message_sequence['size']<=ordersize[1])
                                        & (message_sequence['direction']==orderside)
                                        & (message_sequence['time']<=Decimal(timestamp[1]))
                                        & (message_sequence['time']>=Decimal(timestamp[0]))]
    sequence_indices=[]
    for i in impact_msgs.index:
        if (i+sequence_length) > len(message_sequence):
            break
        sequence_indices=sequence_indices+list(range(i,i+sequence_length,1))
    filtered_messages=message_sequence.loc[sequence_indices]
    filtered_books=book_sequence.loc[sequence_indices]
    return filtered_messages,filtered_books



 
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