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

def _filter_touch_events(m_df,b_df):
    indeces=(~b_df[[0, 1, 2,3]].diff().eq(0).all(axis=1) & m_df['event_type'].isin([1,2,3,4]))
    m_df=m_df.loc[indeces]
    b_df=b_df.loc[indeces]
    #FIXME: Should really ensure that the market orders within a ms are collapsed into one. pg.1400 of paper.
    big_df=pd.concat([m_df,b_df],axis=1)
    return big_df

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
        ((test_df['time'].diff()>datetime.timedelta(milliseconds=1)) |
        (test_df['event_type']!=4) |
        (test_df['direction']!=test_df['direction'].shift(1))).cumsum())
    cols=test_df.columns.difference(['size'])
    final=gb.agg({'size': "sum"}).join(gb[cols].last())
    return final


#Skip, or TODO at some other point. 
def matrix_A(sequence):
    """equation 19 from paper"""
    return 0


def _apply_to_Sequences(fun,CI,max_seq,args,loader:Simple_Loader):
    print(args[1],end="\r")
    metrics_real=[]
    metrics_gen=[]
    for i in range(0,min(max_seq,len(loader))):
        real_df=_classify_event_types(
                        #merge_MOs(
                        _filter_touch_events(loader[i].m_real,loader[i].b_real))#)
        metr=fun(*args,real_df)
        metrics_real.append(metr)

        for j,mgen in enumerate(loader[i].m_gen):
            gen_df=_classify_event_types(
                        #merge_MOs(
                        _filter_touch_events(mgen,loader[i].b_gen[j]))#)
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

def impact_compare(loader:Simple_Loader,ticker:str="BLANK",model="BLANK",ticksize=100):
    x=(10** np.arange(0.3,2.4,step=0.1)).astype(int)
    events=['MO_0','MO_1','LO_0','LO_1','CA_0','CA_1'] #'MO_0','MO_1'
    ys_real=[]
    ys_gen=[]
    confidence_ints_real=[]
    confidence_ints_gen=[]
    for event in events:
        print("Calculating for event type: ", event)
        r,g=zip(*[_apply_to_Sequences(_response_func,0.99,1000,(event,i,ticksize),loader) for i in x])
        r_m,r_ci=zip(*r)
        g_m,g_ci=zip(*g)
        ys_real.append(np.array(r_m))
        ys_gen.append(np.array(g_m))
        confidence_ints_gen.append(g_ci)
        confidence_ints_real.append(r_ci)

    np.savez("ys_real_"+ticker,*ys_real)
    np.savez("ys_gen_"+ticker,*ys_gen)
    np.save("x_vals_"+ticker,x)


    with open('cis_real'+ticker+'_'+model+'.pickle', 'wb') as f:
        pickle.dump(confidence_ints_real,f)
    with open('cis_gen'+ticker+'_'+model+'.pickle', 'wb') as f:
        pickle.dump(confidence_ints_gen,f)
    
    plot_linlog_subplots(x,[ys_real,ys_gen],[confidence_ints_real,confidence_ints_gen],
                     legend=events,
                     suptitle="Tick-normalised microscopic response functions for stock ticker: "+ticker,
                     titles=["Real Data Sequences","Generated Data Sequences"],
                     ylabel="Rπ (ticks)",
                     ticker=ticker,
                     model=model)


    diff={}
    for i,event in enumerate(events):
        delta=np.sum(np.abs(ys_real[i]-ys_gen[i]))
        print("Sum of abs differences for ",event," events:",delta)
        diff[event]=delta
    abs_diffs=np.array(list(diff.values()))
    np.save("comparison_graph_"+ticker+"_"+model,abs_diffs,)
    return np.mean(abs_diffs)

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

def plot_linlog_subplots(x,ys,errs,legend,colors=['r','g','b'],suptitle=None,titles=["Title"],loglog=None,ylabel="y_axis_replace",ticker:str="BLANK",model:str="BLANK"):
    fig,axarr = plt.subplots(1,len(ys),sharey=True)
    plt.subplots_adjust(wspace=0.05)
    fig.set_figwidth(12)
    ys_lims=(-0.1,0.1)
    markers=['x','+']
    for a,ax in enumerate(axarr):
        for i,y in enumerate(ys[a]):
            c=colors[math.floor(i/2)]
            m=markers[i%2]
            #ax.scatter(x,y,marker='x',color=colors[i])
            ax.plot(x, y,'--',color=c,lw=0.4,marker=m,)
            ax.fill_between(x,np.array(errs[a][i])[:,0],np.array(errs[a][i])[:,1],alpha=0.1,label='_nolegend_',color=c)
            ys_lims=(min(ys_lims[0],np.min(y)),max(ys_lims[1],np.max(y)))
            

        if loglog is not None:
            ax.loglog(x,np.power(x,-loglog),color='k')
        ax.set_xscale('log')
        ax.legend(legend)
        ax.set_title(titles[a])
        ax.set_xlabel("Events lag (l)")
    axarr[0].set_ylabel(ylabel)
    for a in axarr:
        ax.set_ylim(np.array(ys_lims)*1.2)
    fig.suptitle(suptitle)
    fig.savefig('compare_'+ticker+'_'+model+'.png', dpi=fig.dpi)

 
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



if __name__ == '__main__':
    for root_path in ["/data1/sascha/data/GOOG/","/data1/sascha/data/INTC/"]:
        print("Loading data from "+root_path)
        ticker=str.split(root_path,"/")[-2]
        
        loader = data_loading.Simple_Loader(
                real_data_path= root_path+"data_real",
                gen_data_path= root_path+"data_gen",
                cond_data_path=root_path+"data_cond",
        )
        res=impact_compare(loader,ticker=ticker,ticksize=100)
        #df=macro_impact_data(10,100,loader)
        #print(df)
        print("Sum of differences of response function: ",res)