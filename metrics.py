import numpy as np
import pandas as pd
import scipy.stats as stats
from partitioning import flatten


def wasserstein(p, q):
    p, q = flatten(p), flatten(q)
    return stats.wasserstein_distance(p, q)

def l1_by_group(score_df: pd.DataFrame) -> float:
    """
    Takes a "score dataframe" with columns score (real numbers), group (+int), type ("real" or "generated")
    Returns the mean L1 distance between the number of scores in each group for the real and generated data.
    """
    group_counts = score_df.groupby(['type', 'group']).count()
    group_counts = pd.merge(group_counts.loc['real'], group_counts.loc['generated'], on='group')
    return (group_counts.score_x - group_counts.score_y).abs().mean()

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

import scipy.stats
from sklearn.neighbors import KernelDensity

def kl_divergence(a, b):
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
kl_divergence(a, b)


# TODO (Peer): use the wasserstein metric to compute differences between order book volume samples (copy from LOBS5 project)
