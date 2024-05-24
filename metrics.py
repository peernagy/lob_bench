import numpy as np
import pandas as pd
import scipy.stats as stats
from partitioning import flatten
import warnings

import scipy, time
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import stats, integrate

import scipy.stats
from sklearn.neighbors import KernelDensity

from scipy.spatial import cKDTree


def wasserstein(p, q):
    p, q = flatten(p), flatten(q)
    return stats.wasserstein_distance(p, q)

def l1_by_group(score_df: pd.DataFrame) -> float:
    """
    Takes a "score dataframe" with columns score (real numbers), group (+int), type ("real" or "generated")
    Returns the mean L1 distance between the number of scores in each group for the real and generated data.
    """
    if 'generated' not in score_df['type'].values:
        return 1.
    group_counts = score_df.groupby(['type', 'group']).count()
    group_counts = pd.merge(group_counts.loc['real'], group_counts.loc['generated'], on='group')
    group_counts /= group_counts.sum(axis=0)
    return (group_counts.score_x - group_counts.score_y).abs().sum() / 2.

def ob3Drepr(orderbook, row_index):
    '''
    Input is the orderbook, for example, `b_real`, `b_gene`, and the corresponding `row_index`. 
    Output is the 3D reconstruction of the row in the orderbook corresponding to `row_index`.
    delta_mid_price: delta mid-price
    index:           relative price level where change occurs,
    quant:           change in quantity at that level
    '''
    price_level = 10
    delta_mid_price = orderbook.iloc[row_index,-1]
    lb_array=orderbook.iloc[row_index-1,:4*price_level].values.reshape(2*price_level,2)
    b_array=orderbook.iloc[row_index,:4*price_level].values.reshape(2*price_level,2)
    lb_unique = np.array([x for x in lb_array if not any(np.array_equal(x, y) for y in b_array)])
    b_unique = np.array([x for x in b_array if not any(np.array_equal(x, y) for y in lb_array)])
    first_lb_unique=lb_unique[0];first_b_unique=b_unique[0]
    index = np.where(b_array[:, 0] == first_b_unique[0])[0][0]
    quant = np.where(first_b_unique[0] == first_lb_unique[0], first_b_unique[1] - first_lb_unique[1], first_b_unique[1])
    result = np.array([delta_mid_price,index,quant])
    return result

# TODO: Implement way to estimate KL based on samples from continuous distributions
#       e.g. using KDE, fitting densities (check empirical distribution shapes - normal?)
#       or directly using ECDF https://www.semanticscholar.org/paper/Kullback-Leibler-divergence-estimation-of-Pérez-Cruz/310974d3c141589a7800d737e5859b76676dcb5d?p2df
#       WHY? -> KL is undefined if the support of the two distributions is not the same, so we can't just use the discrete values of the histograms
#       (or would need to define the bins in a way that the support is the same, but this would underestimate the KL)



def kl_divergence_kde(a, b):
    """
    Calculate the Kullback-Leibler (KL) divergence between the KDEs of two datasets.

    Args:
        a (np.ndarray): First dataset of shape (N, K), where N is the number of samples and K is the number of features.
        b (np.ndarray): Second dataset of shape (M, K), where M is the number of samples and K is the number of features.

    Returns:
        float: The KL divergence between the two datasets.
    """
    N = a.shape[0]
    M = b.shape[0]
    K = a.shape[1]
    assert a.shape[1] == b.shape[1]    

    # Fit KDE models
    kde_a = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(a)
    kde_b = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(b)

    # Generate evaluation points based on the average number of samples
    num_points = int(np.mean([N, M]) * 10)
    # x = np.linspace(0, 1, 1000)
    x = np.linspace(0, 1, num_points)

    # Generate a grid of K-dimensional points for PDF evaluation
    grid = np.array(np.meshgrid(*([x] * K))).T.reshape(-1, K)
    

    log_pdf_a = kde_a.score_samples(grid)
    log_pdf_b = kde_b.score_samples(grid)

    pdf_a = np.exp(log_pdf_a)
    pdf_b = np.exp(log_pdf_b)

    # Add a small value to PDFs to avoid log(0)
    epsilon = 1e-10
    pdf_a += epsilon
    pdf_b += epsilon

    # Normalize PDFs
    pdf_a /= np.sum(pdf_a)
    pdf_b /= np.sum(pdf_b)

    # Calculate KL divergence
    kl_div = scipy.stats.entropy(pdf_a, pdf_b)

    # print("KL Divergence:", kl_div)
    return kl_div




def kl_divergence_PerezCruz(P, Q, eps=1e-11):
    '''
    Only work for one dimensional, e.g. P(N*1) and Q(M*1) as input
    
    Kullback-Leibler divergence estimation of continuous distributions
    Published in IEEE International Symposium… 6 July 2008 Mathematics
    F. Pérez-Cruz, Pedro E. Harunari, Ariel Yssou 
    
    Codes based on Section II of Fernando Pérez-Cruz's paper 
    "Kullback-Leibler Divergence Estimation of Continuous Distributions". 
    From two independent datasets of continuous variables, 
    the KLD (aka relative entropy) is estimated by the construction of 
    cumulative probability distributions and the comparison between their slopes at specific points.
    Estimating the probability distributions and directly evaluating KLD's definition
    leads to a biased estimation, whereas the present method leads to an unbiased estimation. 
    This is particularly important in practical applications due to the finitude of collected statistics.

    Func takes two datasets to estimate the relative entropy between their PDFs
    we use eps=10^-11, but it could be defined as < the minimal interval between data points.
    '''
    def cumcount_reduced(arr):
        '''Returns the step function value at each increment of the CDF'''
        sorted_arr = np.array(sorted(arr))
        counts = np.zeros(len(arr))
        
        rolling_count = 0
        for idx, elem in enumerate(sorted_arr):
            rolling_count += 1
            counts[idx] = rolling_count

        counts /= len(counts)
        counts -= (1 / (2 * len(counts)))

        return (sorted_arr, counts)
    P = sorted(P)
    Q = sorted(Q)
    
    P_positions, P_counts = cumcount_reduced(P)
    Q_positions, Q_counts = cumcount_reduced(Q)
    
    #definition of x_0 and x_{n+1}
    x_0 = np.min([P_positions[0], Q_positions[0]]) - 1
    P_positions = np.insert(P_positions, 0, [x_0])
    P_counts = np.insert(P_counts, 0, [0])
    Q_positions = np.insert(Q_positions, 0, [x_0])
    Q_counts = np.insert(Q_counts, 0, [0])
    
    x_np1 = np.max([P_positions[-1], Q_positions[-1]]) + 1
    P_positions = np.append(P_positions, [x_np1])
    P_counts = np.append(P_counts, [1])
    Q_positions = np.append(Q_positions, [x_np1])
    Q_counts = np.append(Q_counts, [1])
    
    f_P = interp1d(P_positions, P_counts)
    f_Q = interp1d(Q_positions, Q_counts) 
    
    X = P_positions[1:-2]
    values = (f_P(X) - f_P(X - eps)) / (f_Q(X) - f_Q(X - eps))
    filt = ((values != 0.) & ~(np.isinf(values)) & ~(np.isnan(values)))
    values_filter = values[filt]
    out = (np.sum(np.log(values_filter)) / len(values_filter)) - 1.

    return out



def kl_divergence_knn(X, Y):
    """
    Estimate the Kullback-Leibler divergence between two multivariate samples.

    Parameters
    ----------
    X : 2D numpy array (n, d)
        Samples from distribution P, which typically represents the true distribution.
    Y : 2D numpy array (m, d)
        Samples from distribution Q, which typically represents the approximate distribution.

    Returns
    -------
    out : float
        The estimated Kullback-Leibler divergence D(P||Q).
    """
    # Get important dimensions
    d = X.shape[1]  # number of dimensions, must be the same in X and Y
    n = X.shape[0]  # number of samples in X
    m = Y.shape[0]  # number of samples in Y

    # Get distances to nearest neighbors using cKDTree
    kdtree_X = cKDTree(X)
    kdtree_Y = cKDTree(Y)

    # Distances to the nearest neighbors within the same dataset
    r = kdtree_X.query(X, k=2, eps=0.01)[0][:, 1]  # distances to the 2nd closest neighbor
    s = kdtree_Y.query(X, k=1, eps=0.01)[0]  # distances to the closest neighbor

    # There is a mistake in the paper. In Eq. 14, the right side misses a negative sign on the first term of the right hand side.
    return - np.sum(np.log(r / s)) * d / n + np.log(m / (n - 1))

# TODO (Peer): use the wasserstein metric to compute differences between order book volume samples (copy from LOBS5 project)

if __name__=="__main__":
    # Example usage
    N = 1011
    M = 1234
    # M = 211
    K = 1
    a = np.random.rand(N, K)
    b = np.random.rand(M, K)
    kl_divergence_kde(a, b)
    kl_divergence_PerezCruz(a, b)
    


