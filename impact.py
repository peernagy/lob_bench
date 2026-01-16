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
from data_loading import Simple_Loader
import datetime
import math
from matplotlib import pyplot as plt
import data_loading
import pickle
from collections import Counter
from pathlib import Path
import argparse
import time
import os
import tqdm

from collections import namedtuple
ConfidenceInterval = namedtuple("ConfidenceInterval", ["low", "high"])

def _filter_sequence_for_impact(message_sequence : pd.DataFrame,
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

def _filter_touch_events(m_df,b_df, realid=0,date=''):
    indeces=(~b_df[[0, 1, 2,3]].eq(0).any(axis=1) & ~b_df[[0, 1, 2,3]].diff().eq(0).all(axis=1) & m_df['event_type'].isin([1,2,3,4]) & m_df['direction'].isin([-1,1]))
    # if not indeces.any():
    #     print(f"For this sequence {date}_{realid}, there are no valid touch events")
    m_df=m_df.loc[indeces]
    b_df=b_df.loc[indeces]
    #FIXME: Should really ensure that the market orders within a ms are collapsed into one. pg.1400 of paper.
    big_df=pd.concat([m_df,b_df],axis=1)
    return big_df

def _filter_market_orders(m_df,b_df):
    indeces=(~b_df[[0, 1, 2,3]].eq(0).any(axis=1) & ~b_df[[0, 1, 2,3]].diff().eq(0).all(axis=1) & m_df['event_type'].isin([4]) & m_df['direction'].isin([-1,1]))
    m_df=m_df.loc[indeces]
    b_df=b_df.loc[indeces]
    big_df=pd.concat([m_df,b_df],axis=1)
    return big_df


def _classify_MO_only(big_df :pd.DataFrame):
    big_df['askprice_diff']=big_df[0].diff()
    big_df['bidprice_diff']=big_df[2].diff()


    def epsilon_fun(series: pd.Series):
         if series['event_type']==4:
             return {'eps':series['direction']*-1,'s':series['direction']*-1}
         else:
             raise ValueError("Event type is none of 4. Ensure filtering is done properly beforehand.")

    def classification_fun(series : pd.Series):
        """Incoming series merged"""

        """
        MO_0: market order. Type 4 and volume doesn't match entire price level-> prices unchanged
        MO_1: market order type 4, price is changed
        """
        
        if series['event_type']==4:
            return "MO"
        else: 
            raise ValueError("Event type is none of 4. Ensure filtering is done properly beforehand.")
            
    big_df['bouchaud_event']=big_df.apply(classification_fun,axis=1,)
    eps=big_df.apply(epsilon_fun,axis='columns',result_type='expand')
    df=pd.concat([big_df,eps],axis='columns')
    df['midprice']=((df[0]+df[2])/2).astype(int)
    df=merge_MOs(df)
    return df

def _classify_event_types(big_df :pd.DataFrame):
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
    df=merge_MOs(df)
    return df

def _sign_autocorr(lag:int,sign:str, sequence:pd.DataFrame):
    """Calculates autocorrelation of adjusted sign 's' which is sign 'eps' except for LOs which are reveresed"
    """

    """if MO, CA, s = eps
        elif LO s = -eps"""

    """ should decay with lag^-0.7"""
    mult=sequence[sign]*sequence[sign].shift(periods=-lag)
    return mult

def _event_cross_corr(event1, event2, lag, sequence:pd.DataFrame):
    """Equation 13 from the paper.
    """
    series=(sequence['bouchaud_event'].eq(event1)
            * sequence['eps']
            * sequence['bouchaud_event'].shift(periods=-lag).eq(event2)
            * sequence['eps'].shift(periods=-lag))
    num=series.mean()
    Pi_1,count1=_prob_pi(event1,sequence)
    Pi_2,count2=_prob_pi(event2,sequence)
    C_pi1_pi2=series / (Pi_1 * Pi_2)
    return C_pi1_pi2

def _prob_pi(event,sequence:pd.DataFrame):
    """Page 1400 from the bouchaud paper"""
    count=sequence['bouchaud_event'].eq(event)
    return count.mean(),count.sum()

def _u_event_corr(event1, event2,lag,  sequence:pd.DataFrame):
    """Equation 14 from the bouchaud paper.
    """
    series=(sequence['bouchaud_event'].eq(event1)
            * sequence['bouchaud_event'].shift(periods=-lag).eq(event2))
    num=series.mean()
    PI_pi1_pi2=series / (_prob_pi(event1,sequence)
                     * _prob_pi(event2,sequence)) - 1
    return PI_pi1_pi2

def _response_func(event, lag, ticksize, sequence:pd.DataFrame):
    price_changes=(sequence['midprice'].shift(periods=-(lag-1))-sequence['midprice'].shift(periods=+1))*sequence['eps']
    price_changes=price_changes/ticksize
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

def merge_MOs(test_df:pd.DataFrame):
    gb=test_df.groupby(
        ((test_df['time'].diff()>datetime.timedelta(microseconds=1)) |
        (test_df['event_type']!=4) |
        (test_df['direction']!=test_df['direction'].shift(1)) |
        (test_df['price']!=test_df['price'].shift(1))).cumsum()) 
    cols=test_df.columns.difference(['size'])
    final=gb.agg({'size': "sum"}).join(gb[cols].last())
    return final


#Skip, or TODO at some other point. 
def matrix_A(sequence):
    """equation 19 from paper"""
    return 0

def score_sequence(m_df: pd.DataFrame,b_df : pd.DataFrame,fun,event:str ="MO",lag:int = 1,ticksize:int=100) -> pd.Series:
    if event in ["MO"]:
        filtered_df=_filter_market_orders(m_df,b_df)
        if not filtered_df.empty:
            classified_df=_classify_MO_only(filtered_df)
            metr=fun(event,lag,ticksize,classified_df)
            return metr
        else:
            return pd.Series(dtype=float)
    else:
        raise ValueError("Event type not recognised. Currently only MO supported.")



def _apply_to_Sequences(fun,CI,max_seq,args,loader:Simple_Loader):
    print(args[1],end="\r")
    metrics_real=[]
    metrics_gen=[]
    for i in tqdm.tqdm(range(0,min(max_seq,len(loader))), desc="Processing sequences"):
        filtered_df=_filter_touch_events(loader[i].m_real,loader[i].b_real,loader[i].real_id,loader[i].date)
        if not filtered_df.empty:
            real_df=_classify_event_types(filtered_df)
            metr=fun(*args,real_df)
            metrics_real.append(metr)

        for j,mgen in enumerate(loader[i].m_gen):
            filtered_df=_filter_touch_events(mgen,loader[i].b_gen[j],loader[i].real_id,loader[i].date)
            if not filtered_df.empty:
                gen_df=_classify_event_types(filtered_df)
                metr=fun(*args,gen_df)
                metrics_gen.append(metr)
    metrics_real=pd.concat(metrics_real,axis=0)
    metrics_gen=pd.concat(metrics_gen,axis=0)
    results=[]
    for metrics in [metrics_real,metrics_gen]:
        array_metrics=np.array(metrics[~np.isnan(metrics)])
        unique=np.unique(array_metrics).shape[0]
        if unique>1:
            res=st.bootstrap((array_metrics,),np.mean,confidence_level=CI,n_resamples=1000,method='basic')
            ci=res.confidence_interval
        else:
            ci=ConfidenceInterval(0,0)
        results.append((metrics.mean(),ci))
    return results[0],results[1]




def macro_impact_compare(max_seq,ticksize,Nbins,loader):
    data={'dm_real':[], 
          'p_pi_real':[],
          'dm_gen':[],
          'p_pi_gen':[]}

    for i in range(0,min(max_seq,len(loader))):
        dm, p_pi=_get_m_p(loader[i].m_real,loader[i].b_real,ticksize)
        data['dm_real'].append(dm)
        data['p_pi_real'].append(p_pi)
        for j,mgen in enumerate(loader[i].m_gen):
            dm, p_pi=_get_m_p(loader[i].m_gen[j],loader[i].b_gen[j],ticksize)
            data['dm_gen'].append(dm)
            data['p_pi_gen'].append(p_pi)
    df=pd.DataFrame(data)

    #use qcut if want uniform amounts. This gives uniform bins
    a,bins=pd.cut(df['p_pi_real'],Nbins,duplicates='drop',retbins=True,labels=False)

    M_i_r,P_i_r,M_i_g,P_i_g=[],[],[],[]

    for i in range(0,Nbins):
        M_i_r.append(df['dm_real'][a==i].mean())
        P_i_r.append(df['p_pi_real'][a==i].mean())
        M_i_g.append(df['dm_gen'][a==i].mean())
        P_i_g.append(df['p_pi_gen'][a==i].mean())

    return M_i_r,P_i_r,M_i_g,P_i_g

def macro_impact_analyse(tau:int,ticksize,Nbins,m_seqs,b_seqs):
    mb_seqs=[pd.concat([m,b],axis=1) for m,b in zip(m_seqs,b_seqs)]
    data={'dm':[], 
          'p_pi':[],}

    time_seqs=[]

    print("Concat Done")

    for df in mb_seqs:
        for i,df in df.groupby((df['time']-df['time'].iloc[0])//tau):
            dm, p_pi=_get_m_p_merged(df,ticksize)
            data['dm'].append(dm)
            data['p_pi'].append(p_pi)

    print("Time split & Calc done")

    df=pd.DataFrame(data)

    #use qcut if want uniform amounts. This gives uniform bins
    a,bins=pd.cut(df['p_pi'],Nbins,duplicates='drop',retbins=True,labels=False)
    #print("a",a.value_counts().sort_index())

    

    M_i_r,P_i_r=[],[]

    for i in range(0,Nbins):
        M_i_r.append(df['dm'][a==i].mean())
        P_i_r.append(df['p_pi'][a==i].mean())

    return M_i_r,P_i_r


    


def _get_m_p(m_df: pd.DataFrame,b_df : pd.DataFrame,ticksize:int =100):
    #Unsure about directions for sell and buy but doesn't matter cause abs diff and sum. 
    V_sell=m_df[((m_df['event_type']==4) &
            (m_df['direction']==1))]['size'].sum()
    V_buy=m_df[((m_df['event_type']==4) &
            (m_df['direction']==-1))]['size'].sum()
    den=(V_buy+V_sell)
    if den !=0:
        p=abs(V_sell-V_buy)/den
    else:
        p=0
    mid_init=(b_df[0].iloc[0]+b_df[2].iloc[0])/2
    mid_final=(b_df[0].iloc[-1]+b_df[2].iloc[-1])/2
    delta_mid=mid_final-mid_init
    return delta_mid/ticksize,p

def _get_m_p_merged(mb_df: pd.DataFrame,ticksize:int =100):
    #Unsure about directions for sell and buy but doesn't matter cause abs diff and sum. 
    V_sell=mb_df[((mb_df['event_type']==4) &
            (mb_df['direction']==1))]['size'].sum()
    V_buy=mb_df[((mb_df['event_type']==4) &
            (mb_df['direction']==-1))]['size'].sum()
    den=(V_buy+V_sell)
    if den !=0:
        p=abs(V_sell-V_buy)/den
    else:
        p=0
    mid_init=(mb_df[0].iloc[0]+mb_df[2].iloc[0])/2
    mid_final=(mb_df[0].iloc[-1]+mb_df[2].iloc[-1])/2
    delta_mid=mid_final-mid_init
    return delta_mid/ticksize,p

def impact_compare(loader:Simple_Loader,
                   ticker:str="BLANK",
                   model="BLANK",
                   ticksize=100,
                   save_dir='./'):
    x=(10** np.arange(0.3,2.4,step=0.1)).astype(int) # Lag l values
    [1]+list(x)
    events=['MO_0','MO_1','LO_0','LO_1','CA_0','CA_1'] # Bouchaud event types from 6 propagator model.
    ys_real=[]
    ys_gen=[]
    confidence_ints_real=[]
    confidence_ints_gen=[]
    diff={}

    for event in events:
        print("Calculating for event type: ", event)
        r,g=zip(*[_apply_to_Sequences(_response_func,0.99,8192,(event,i,ticksize),loader) for i in x])
        r_m,r_ci=zip(*r)
        g_m,g_ci=zip(*g)
        ys_real.append(np.array(r_m))
        ys_gen.append(np.array(g_m))
        confidence_ints_gen.append(g_ci)
        confidence_ints_real.append(r_ci)

        # Calculate sum of abs differences. 
        delta=np.sum(np.abs(np.array(r_m)-np.array(g_m)))
        diff[event]=delta
    abs_diffs=np.array(list(diff.values()))

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_tuple=(x,ys_real,ys_gen,confidence_ints_real,confidence_ints_gen,abs_diffs)
    save_impact_data(save_tuple,
                     save_dir=save_dir,
                     ticker=ticker,
                     model=model)

    return {"Mean_Diffs":np.mean(abs_diffs),"Saved_Tuple":save_tuple}

def impact_analyse(m_seqs,b_seqs):
    mb_seqs=[_filter_touch_events(m,b) for m,b in zip(m_seqs,b_seqs)]
    df_seqs_mos=[merge_MOs(df) for df in mb_seqs]
    df_seqs=[_classify_event_types(df) for df in df_seqs_mos]
    temp=df_seqs[0][0].diff().abs()
    ticksize=temp[temp>0].min()

    x=(10** np.arange(0,3,step=0.1)).astype(int)
    events=['MO_0','MO_1','LO_0','LO_1','CA_0','CA_1'] #'MO_0','MO_1'
    ys=[]
    confidence_ints=[]
    for event in events:
        print("Calculating for event type: ", event)
        mean,cis=zip(*[apply_to_seqs(_response_func,(event,i,ticksize),df_seqs) for i in x])
        ys.append(np.array(mean))
        confidence_ints.append(cis)

    plot_linlog(x,ys,confidence_ints,legend=events,
            title="Response functions real data",
            ylabel="P_response")


def save_impact_data(save_tuple, 
                    save_dir=".",
                    ticker="STOCK",
                    model="MODEL",
                    macro="micro"):
    save_string=f"/{macro}_impact_for_{ticker}_{model}.pickle"
    with open(save_dir+save_string, 'wb') as f:
        pickle.dump(save_tuple,f)

def load_impact_data(save_dir=".",
                    ticker="STOCK",
                    model="MODEL",
                    macro="micro"):
    save_string=f"/{macro}_impact_for_{ticker}_{model}.pickle"
    with open(save_dir+save_string, 'rb') as f:
        save_tuple=pickle.load(f)
    return save_tuple


#PLotting utilities
def plot_loglog(x,ys,errs,legend,colors=['b','r','g','c','m','y'],title="Title",loglog=None,ylabel="y_axis_replace",invert=False):
    fig = plt.figure()
    ax = plt.gca()
    ys_lims=(0,0.1)
    for i,y in enumerate(ys):
        #ax.scatter(x,y,marker='x',color=colors[i])
        ax.plot(x, y,'--',color=colors[i],lw=0.4,marker='x')
    if errs is not None:
            ax.fill_between(x,y-errs[i],y+errs[i],alpha=0.1,label='_nolegend_')
            ys_lims=(min(ys_lims[0],np.min(y)),max(ys_lims[1],np.max(y)))

    if loglog is not None:
        ax.loglog(x,np.power(x,-loglog),color='k')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.legend(legend)
    ax.set_title(title)
    ax.set_xlabel("Lag")
    ax.set_ylabel(ylabel)
    if errs is not None:
        ax.set_ylim(np.array(ys_lims)*1.2)
    if invert:
        ax.invert_yaxis()


def plot_linlog(x,ys,errs,legend,colors=['b','r','g','c','m','y'],title="Title",loglog=None,ylabel="y_axis_replace"):
    fig = plt.figure()
    ax = plt.gca()
    ys_lims=(-0.1,0.1)
    for i,y in enumerate(ys):
        #ax.scatter(x,y,marker='x',color=colors[i])
        print()
        ax.plot(x, y,'--',color=colors[i],lw=0.4,marker='x')
        ax.fill_between(x,np.array(errs[i])[:,0],np.array(errs[i])[:,1],alpha=0.1,label='_nolegend_')
        ys_lims=(min(ys_lims[0],np.min(y)),max(ys_lims[1],np.max(y)))
        

    if loglog is not None:
        ax.loglog(x,np.power(x,-loglog),color='k')
    ax.set_xscale('log')
    ax.legend(legend)
    ax.set_title(title)
    ax.set_xlabel("Lag")
    ax.set_ylabel(ylabel)
    ax.set_ylim(np.array(ys_lims)*1.2)


def plot_linlog_subplots(x,ys,errs,
                         legend,colors=['r','g','b'],
                         suptitle=None,titles=["Title"],
                         loglog=None,ylabel="y_axis_replace",
                         ticker:str="BLANK",model:str="BLANK",
                         save_dir='./'):
    assert len(ys)==len(errs)
    fig,axarr = plt.subplots(1,len(ys),sharey=False)
    plt.subplots_adjust(wspace=0.05)
    fig.set_figwidth(12)
    ys_lims=(-0.1,0.1)
    markers=['x','+']
    for a,ax in enumerate(axarr):
        for i,y in enumerate(ys[a]):
            c=colors[math.floor(i/2)]
            m=markers[i%2]
            #ax.scatter(x,y,marker='x',color=colors[i])
            # print(x,y)
            ax.plot(x[:-2], y[:-2],'--',color=c,lw=0.4,marker=m,)
            ax.fill_between(x[:-2],np.array(errs[a][i])[:-2,0],np.array(errs[a][i])[:-2,1],alpha=0.1,label='_nolegend_',color=c)
            ys_lims=(min(ys_lims[0],np.min(y)),max(ys_lims[1],np.max(y)))
            

        if loglog is not None:
            ax.loglog(x,np.power(x,-loglog),color='k')
        ax.set_xscale('log')
        ax.legend(legend)
        ax.set_title(titles[a],fontsize=16)
        ax.set_xlabel("Events lag (l)",fontsize=14)
    axarr[0].set_ylabel(ylabel,fontsize=14)
    for a in axarr:
        a.set_ylim(np.array(ys_lims)*1.2)
        # ax.tick_params(axis='x', labelsize=14)  # X tick labels font size
        # ax.tick_params(axis='y', labelsize=14)  # Y tick labels font size



    fig.suptitle(suptitle,fontsize=16,fontweight='bold')
    fig.savefig(save_dir+'/micro_impact_compare_'+ticker+'.png', dpi=fig.dpi)


def plot_macro_impact_subplots(p,m,
                         legend=[],colors=['r','g','b'],
                         suptitle=None,titles=["Title"],
                         loglog=None,ylabel="y_axis_replace",
                         ticker:str="BLANK",model:str="BLANK",
                         save_dir='./'):
    assert len(p)==len(m)
    fig,axarr = plt.subplots(1,len(p),sharey=False)
    plt.subplots_adjust(wspace=0.05)
    fig.set_figwidth(12)
    y_lims=(-100,100)
    markers=['x','+']

    for a,ax in enumerate(axarr):
        c='r'
        mark=markers[a%2]
        #ax.scatter(x,y,marker='x',color=colors[i])
        print("p is ",p)
        print(p[a])

        print("m is ",m)
        print(m[a])
        ax.plot(np.abs(np.array(p[a])),np.abs(np.array(m[a])), 'o', c='blue', alpha=0.5, markeredgecolor='none')

        ax.legend(legend)
        ax.set_title(titles[a],fontsize=16)
        ax.set_xlabel("Participation of Volume",fontsize=14)

    axarr[0].set_ylabel(ylabel,fontsize=14)
    for a in axarr:
        ax.set_ylim(np.array(y_lims)*1.2)
        # ax.tick_params(axis='x', labelsize=14)  # X tick labels font size
        # ax.tick_params(axis='y', labelsize=14)  # Y tick labels font size


    fig.suptitle(suptitle,fontsize=16,fontweight='bold')
    fig.savefig(save_dir+'/macro_impact_compare_'+ticker+'.png', dpi=fig.dpi)



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


def run_impact(args):

    print("*** \tThe assumed file structure for files in this package is:\n"
                "***\t \t {DATA_DIR}/{MODEL}/{STOCK}/data_*\n"
                "***\twhereby DATA_DIR is passed as an argument when launching the script\n"
                "***\t{MODEL}s and {STOCK}s may either be passed as a str or a list of str\n"
                "***\tThe script will iterate over all combinations of models and stocks\n")
    
    if not (args.micro_calculate or args.micro_plot or args.macro_calculate or args.macro_plot):
        print("You must activate the options micro_calculate, micro_plot, macro_calculate or macro_plot to run these options, by default they are all off.")
    
    if isinstance(args.stock, str):
        args.stock = [args.stock]
    if isinstance(args.model_name, str):
        args.model_name = [args.model_name]
    if isinstance(args.period, str):
        args.period = [args.period]


    if args.micro_calculate:
        for model in args.model_name:
            for ticker in args.stock:
                for p in args.period:
                    root_path = f"{args.data_dir}/{model}/{ticker}/{p}"
                    print(f"Loading data from {root_path}")
                    print(f"Evaluating impact for {model} Model for {ticker}")

                    loader = data_loading.Simple_Loader(
                            real_data_path= root_path+"/data_real",
                            gen_data_path= root_path+"/data_gen",
                            cond_data_path=root_path+"/data_cond",
                    )
                    res=impact_compare(loader,
                                    ticker=ticker,
                                    save_dir=args.save_dir+f"/impact/{ticker}/{model}",
                                    model=model)
                    print("Sum of differences of response function: ",res["Mean_Diffs"])
    if args.micro_plot:
        for ticker in args.stock:
            y_list=[]
            ci_list=[]
            for i,model in enumerate(args.model_name):
                for p in args.period:
                    root_path = f"{args.data_dir}/{model}/{ticker}/{p}"
                    print(f"Plotting impact for {model} Model for {ticker}")

                    save_path=args.save_dir+f"/impact/{ticker}/{model}"
                    (x,ys_real,ys_gen,confidence_ints_real,confidence_ints_gen,abs_diffs)=load_impact_data(save_dir=save_path,
                                                                                                            ticker=ticker,
                                                                                                            model=model)
                    print(f'The abs diffs are {abs_diffs} for {model} and {ticker}.')
                    print(f'The mean is {np.mean(abs_diffs)}.')

                    if i==0:
                        y_list.append(ys_real)
                        ci_list.append(confidence_ints_real)
                    y_list.append(ys_gen)
                    ci_list.append(confidence_ints_gen)

            plot_path=args.save_dir+"/plots"
            os.makedirs(plot_path,exist_ok=True)
            plot_linlog_subplots(x,y_list,ci_list,
                        legend=['MO_0','MO_1','LO_0','LO_1','CA_0','CA_1'],
                        suptitle="Tick-normalised microscopic response functions for stock ticker: "+ticker,
                        titles=["Real Data"]+[f"{mod}" for mod in args.model_name],
                        ylabel="Rπ (ticks)",
                        ticker=ticker,
                        model=model,
                        save_dir=plot_path)
            
    if args.macro_calculate:
        for model in args.model_name:
            for ticker in args.stock:
                root_path = f"{args.data_dir}/{model}/{ticker}"
                print(f"Loading data from {root_path}")
                print(f"Evaluating impact for {model} Model for {ticker}")

                loader = data_loading.Simple_Loader(
                        real_data_path= root_path+"/data_real",
                        gen_data_path= root_path+"/data_gen",
                        cond_data_path=root_path+"/data_cond",
                )
                res=macro_impact_compare(10000,100,10,loader)
                print(res)
                save_path=args.save_dir+f"/impact/{ticker}/{model}"
                save_impact_data(res,save_dir=save_path,ticker=ticker,model=model,macro="macro")
                print(f"Saved impact data for {model} and {ticker}.")

    if args.macro_plot:
        for ticker in args.stock:
            m_list=[]
            p_list=[]
            for i,model in enumerate(args.model_name):
                root_path = f"{args.data_dir}/{model}/{ticker}"
                print(f"Plotting impact for {model} Model for {ticker}")
                save_path=args.save_dir+f"/impact/{ticker}/{model}"
                res=load_impact_data(save_dir=save_path,
                                    ticker=ticker,
                                    model=model,macro="macro")
                (M_i_r,P_i_r,M_i_g,P_i_g)=res
                print(M_i_r[0].dtype)
                if i==0:
                    m_list.append(M_i_r)
                    p_list.append(P_i_r)
                m_list.append(M_i_g)
                p_list.append(P_i_g)

            plot_path=args.save_dir+"/plots"
            os.makedirs(plot_path,exist_ok=True)
            print("M_LIST IS",m_list)
            plot_macro_impact_subplots(p_list,m_list,
                        legend=[],
                        suptitle="Macro Impact graphs from (Vyetrenko, 2019) for stock : "+ticker,
                        titles=["Real Data Sequences"]+[f"{mod}" for mod in args.model_name],
                        ylabel="Average Midprice move (ticks)",
                        ticker=ticker,
                        model=model,
                        save_dir=plot_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock", nargs='+', default="GOOG")
    parser.add_argument("--period", nargs='+', default="2023_Jan")
    parser.add_argument("--data_dir", default="/home/myuser/data/evalsequences", type=str)
    parser.add_argument("--save_dir", type=str,default="/home/myuser/data/lobbench_scores")

    parser.add_argument("--model_name", nargs='+', default="s5_main")
    parser.add_argument("--micro_calculate", action="store_true")
    parser.add_argument("--micro_plot", action="store_true")
    parser.add_argument("--macro_calculate", action="store_true")
    parser.add_argument("--macro_plot", action="store_true",)
    args = parser.parse_args()


    t0=time.time()
    run_impact(args)
    t1=time.time()
    print("Finished Run, time (s) is:", t1-t0)


   