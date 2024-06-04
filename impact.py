"""
Functions to estimate market impact of different events produced by the model, and compared to real data behaviour.
"""
from typing import Optional
import numpy as np
import pandas as pd
from decimal import Decimal
from typing import Tuple
from functools import partial
import scipy.stats as st

from collections import namedtuple
ConfidenceInterval = namedtuple("ConfidenceInterval", ["low", "high"])

def filter_sequence_for_impact(message_sequence : pd.DataFrame,
                                book_sequence: pd.DataFrame,
                                ordersize: Tuple[int,int],
                                orderside: int,
                                ordertype:int,
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
                                         &(message_sequence['event_type']<ordertype)
                                         &(message_sequence['direction']==orderside)]
    else:
        message_sequence['time']=message_sequence['time'].map(Decimal)
        print((message_sequence['time']<=Decimal(timestamp[1]))
                & (message_sequence['time']>=Decimal(timestamp[0])))
        impact_msgs=message_sequence.loc[(message_sequence['size']>=ordersize[0])
                                        &(message_sequence['size']<=ordersize[1])
                                        &(message_sequence['event_type']<ordertype)
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



def filter_touch_events(m_df,b_df):
    indeces=(~b_df[[0, 1, 2,3]].diff().eq(0).all(axis=1) & m_df['event_type'].isin([1,2,3,4]))
    m_df=m_df.loc[indeces]
    b_df=b_df.loc[indeces]
    #FIXME: Should really ensure that the market orders within a ms are collapsed into one. pg.1400 of paper.
    big_df=pd.concat([m_df,b_df],axis=1)
    return big_df

def classify_event_types(big_df :pd.DataFrame):
    big_df['askprice_diff']=big_df[0].diff()
    big_df['bidprice_diff']=big_df[2].diff()


    def epsilon_fun(series: pd.Series):
         if series['event_type']==1:
             return {'eps':series['direction'],'s':series['direction']*-1}
         elif series['event_type']==4:
             return {'eps':series['direction']*-1,'s':series['direction']*-1}
         else:
             return {'eps':series['direction']*-1,'s':series['direction']*-1}

    def classification_fun(series : pd.Series):
        """Incoming series merged"""

        """
        MO_0: market order. Type 4 and volume doesn't match entire price level-> prices unchanged
        MO_1: market order type 4, price is changed
        CA_0: type 2 or 3 order, prices unchanged in book
        CA_1: """
        
        if series['event_type']==4:
            if ((series['askprice_diff']!=0)| 
               (series['bidprice_diff']!=0)):
                return "MO_1"
            else:
                return "MO_0"

        elif ((series['event_type']==2) |
              (series['event_type']==3)):
            if ((series['askprice_diff']!=0)| 
               (series['bidprice_diff']!=0)):
                return "CA_1"
            else:
                return "CA_0"
        elif series['event_type']==1 :
            if ((series['askprice_diff']!=0)| 
               (series['bidprice_diff']!=0)):
                return "LO_1"
            else:
                return "LO_0"
        else: 
            raise ValueError("Event type is none of 1, 2, 3 or 4. Ensure filtering is done properly beforehand.")
            
    big_df['bouchaud_event']=big_df.apply(classification_fun,axis=1,)
    eps=big_df.apply(epsilon_fun,axis='columns',result_type='expand')
    df=pd.concat([big_df,eps],axis='columns')
    df['midprice']=((df[0]+df[2])/2).astype(int)
    return df


def sign_autocorr(lag:int,sign:str, sequence:pd.DataFrame):
    """Calculates autocorrelation of adjusted sign 's' which is sign 'eps' except for LOs which are reveresed"
    """

    """if MO, CA, s = eps
        elif LO s = -eps"""

    """ should decay with lag^-0.7"""
    mult=sequence[sign]*sequence[sign].shift(periods=-lag)
    return mult



def event_cross_corr(event1, event2, lag, sequence:pd.DataFrame):
    """Equation 13 from the paper.
    """
    series=(sequence['bouchaud_event'].eq(event1)
            * sequence['eps']
            * sequence['bouchaud_event'].shift(periods=-lag).eq(event2)
            * sequence['eps'].shift(periods=-lag))
    num=series.mean()
    Pi_1,count1=Prob_pi(event1,sequence)
    Pi_2,count2=Prob_pi(event2,sequence)
    C_pi1_pi2=series / (Pi_1 * Pi_2)
    return C_pi1_pi2

def Prob_pi(event,sequence:pd.DataFrame):
    """Page 1400 from the bouchaud paper"""
    count=sequence['bouchaud_event'].eq(event)
    return count.mean(),count.sum()

def u_event_corr(event1, event2,lag,  sequence:pd.DataFrame):
    """Equation 14 from the bouchaud paper.
    """
    series=(sequence['bouchaud_event'].eq(event1)
            * sequence['bouchaud_event'].shift(periods=-lag).eq(event2))
    num=series.mean()
    PI_pi1_pi2=series / (Prob_pi(event1,sequence)
                     * Prob_pi(event2,sequence)) - 1
    return PI_pi1_pi2

def response_func(event, lag, sequence:pd.DataFrame):
    price_changes=(sequence['midprice'].shift(periods=-(lag-1))-sequence['midprice'].shift(periods=+1))*sequence['eps']
    price_changes=price_changes/100.0
    responses=price_changes[sequence['bouchaud_event'].eq(event)]
    return responses

def apply_to_seqs(fun,CI,args,sequences:list):
    """Fun must return a scalar"""
    metrics=pd.concat([fun(*args,seq) for seq in sequences],axis=0)
    array_metrics=np.array(metrics[~np.isnan(metrics)])
    unique=np.unique(array_metrics).shape[0]
    if unique>1:
        res=st.bootstrap((array_metrics,),np.mean,confidence_level=0.90,n_resamples=1000,method='basic')
        ci=res.confidence_interval
    else:
        ci=ConfidenceInterval(0,0)
    return metrics.mean(),ci


#Skip, or TODO at some other point. 
def matrix_A(sequence):
    """equation 19 from paper"""
    return 0




 
#USe the below for macro impact generative loop. 


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



